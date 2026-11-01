"""LLM experiment plan proposer with schema enforcement, allowlist validation, and fallback."""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from automl.core.config import Settings, get_settings
from automl.core.exceptions import ExperimentPlanError
from automl.core.logging import get_logger
from automl.profiler.schemas import DatasetProfile, TaskType
from automl.planner.catalogue import (
    PLANNER_SYSTEM_PROMPT,
    format_compact_profile,
    get_supported_catalogue_description,
)
from automl.planner.provider import get_llm_provider
from automl.planner.schemas import (
    LLMExperimentProposal,
    PlanValidationResult,
    ProposedModelConfig,
)
from automl.trainer.engine import build_default_experiment_config
from automl.trainer.schemas import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_MODELS,
    REGRESSION_METRICS,
    REGRESSION_MODELS,
    ExperimentConfig,
    ModelCandidateConfig,
    ModelFamily,
    PreprocessingConfig,
)

logger = get_logger("planner.proposer")


def validate_experiment_proposal(
    proposal: LLMExperimentProposal,
    profile: DatasetProfile,
) -> PlanValidationResult:
    """Strictly validate an LLM experiment proposal against task, catalogue, and resource bounds.

    Args:
        proposal: The parsed LLM proposal.
        profile: DatasetProfile from Phase 1.

    Returns:
        PlanValidationResult: Result containing validity flag and diagnostic errors/warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if profile.target_profile is None:
        errors.append("Target profile is missing from dataset profile.")
        return PlanValidationResult(is_valid=False, errors=errors)

    task_type = profile.target_profile.detected_task_type

    # 1. Validate primary and secondary metrics against task
    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        if proposal.primary_metric not in CLASSIFICATION_METRICS:
            errors.append(
                f"Primary metric '{proposal.primary_metric.value}' is invalid for {task_type.value}. "
                f"Allowed metrics: {[m.value for m in CLASSIFICATION_METRICS]}"
            )
        for sec in proposal.secondary_metrics:
            if sec not in CLASSIFICATION_METRICS:
                errors.append(f"Secondary metric '{sec.value}' is invalid for classification.")
    else:
        if proposal.primary_metric not in REGRESSION_METRICS:
            errors.append(
                f"Primary metric '{proposal.primary_metric.value}' is invalid for {task_type.value}. "
                f"Allowed metrics: {[m.value for m in REGRESSION_METRICS]}"
            )
        for sec in proposal.secondary_metrics:
            if sec not in REGRESSION_METRICS:
                errors.append(f"Secondary metric '{sec.value}' is invalid for regression.")

    # 2. Validate candidate models count
    if not proposal.candidates:
        errors.append("LLM proposal contains 0 model candidates. At least 1 candidate is required.")
    elif len(proposal.candidates) > 10:
        errors.append(f"LLM proposal contains {len(proposal.candidates)} candidates, exceeding limit of 10.")

    # 3. Validate each model against allowlist and task type
    allowed_models = (
        CLASSIFICATION_MODELS
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        else REGRESSION_MODELS
    )

    has_baseline = False
    for idx, cand in enumerate(proposal.candidates):
        if cand.model_family not in allowed_models:
            errors.append(
                f"Candidate #{idx+1} '{cand.name}' uses model_family '{cand.model_family.value}' "
                f"which is not in allowlist for {task_type.value}."
            )

        if cand.is_baseline or cand.model_family in (ModelFamily.DUMMY_CLASSIFIER, ModelFamily.DUMMY_REGRESSOR):
            has_baseline = True

        # Check hyperparameters: must only contain primitive types, no code strings
        for param_key, param_val in cand.hyperparameters.items():
            if not isinstance(param_val, (int, float, str, bool, type(None))):
                errors.append(
                    f"Candidate '{cand.name}' hyperparameter '{param_key}' has unapproved type '{type(param_val).__name__}'. "
                    "Only primitive scalar values are allowed."
                )

    if not has_baseline:
        warnings.append("No baseline model designated in proposal. A baseline will be automatically injected.")

    is_valid = (len(errors) == 0)
    return PlanValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        validated_proposal=proposal if is_valid else None,
    )


def convert_proposal_to_experiment_config(
    proposal: LLMExperimentProposal,
    profile: DatasetProfile,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> ExperimentConfig:
    """Convert a validated LLM proposal into a standard ExperimentConfig for the ML engine."""
    target_name = profile.target_profile.name
    task_type = profile.target_profile.detected_task_type

    candidates: List[ModelCandidateConfig] = []
    has_baseline = False

    for cand in proposal.candidates:
        is_base = cand.is_baseline or cand.model_family in (
            ModelFamily.DUMMY_CLASSIFIER,
            ModelFamily.DUMMY_REGRESSOR,
        )
        if is_base:
            has_baseline = True
        candidates.append(
            ModelCandidateConfig(
                name=cand.name,
                model_family=cand.model_family,
                hyperparameters=cand.hyperparameters,
                is_baseline=is_base,
            )
        )

    # Automatically prepend baseline if missing
    if not has_baseline:
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            candidates.insert(
                0,
                ModelCandidateConfig(
                    name="Baseline (Majority Class)",
                    model_family=ModelFamily.DUMMY_CLASSIFIER,
                    hyperparameters={"strategy": "most_frequent"},
                    is_baseline=True,
                ),
            )
        else:
            candidates.insert(
                0,
                ModelCandidateConfig(
                    name="Baseline (Mean Predictor)",
                    model_family=ModelFamily.DUMMY_REGRESSOR,
                    hyperparameters={"strategy": "mean"},
                    is_baseline=True,
                ),
            )

    preprocessing = PreprocessingConfig(
        numeric_imputer=proposal.preprocessing.numeric_imputer,
        categorical_imputer=proposal.preprocessing.categorical_imputer,
        scaling=proposal.preprocessing.scaling,
    )

    return ExperimentConfig(
        target_column=target_name,
        task_type=task_type,
        primary_metric=proposal.primary_metric,
        secondary_metrics=proposal.secondary_metrics,
        test_size=test_size,
        cv_folds=cv_folds,
        random_seed=random_seed,
        candidates=candidates,
        preprocessing=preprocessing,
    )


def propose_experiment_plan(
    profile: DatasetProfile,
    llm: Optional[Any] = None,
    settings: Optional[Settings] = None,
    fallback_to_default: bool = True,
) -> Tuple[ExperimentConfig, Dict[str, Any]]:
    """Propose an experiment plan using an LLM with strict validation and deterministic fallback.

    Args:
        profile: DatasetProfile from Phase 1.
        llm: Optional LangChain chat model instance.
        settings: Application settings.
        fallback_to_default: Whether to fall back to the deterministic plan upon any error.

    Returns:
        Tuple[ExperimentConfig, Dict[str, Any]]: Validated experiment plan and metadata.
    """
    if profile.target_profile is None:
        raise ExperimentPlanError("Cannot generate plan: target profile is missing.")

    chat_model = get_llm_provider(settings=settings, custom_llm=llm)

    # 1. Fallback if no LLM provider or credentials available
    if chat_model is None:
        logger.info("LLM provider unavailable; falling back to deterministic plan.")
        default_plan = build_default_experiment_config(profile)
        metadata = {
            "source": "deterministic_fallback",
            "reason": "No LLM provider or credentials available.",
            "rationale": "Generated via deterministic heuristics based on dataset profile.",
        }
        return default_plan, metadata

    # 2. Build structured prompt and parser
    task_type = profile.target_profile.detected_task_type
    catalogue_desc = get_supported_catalogue_description(task_type)
    compact_profile = format_compact_profile(profile)

    parser = PydanticOutputParser(pydantic_object=LLMExperimentProposal)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        (
            "human",
            "Review the dataset profile and model catalogue below, then propose a controlled experiment plan.\n\n"
            "{catalogue_desc}\n\n"
            "{compact_profile}\n\n"
            "{format_instructions}\n"
        ),
    ])

    formatted_prompt = prompt.format_prompt(
        catalogue_desc=catalogue_desc,
        compact_profile=compact_profile,
        format_instructions=parser.get_format_instructions(),
    )

    # 3. Invoke LLM with error handling and fallback
    try:
        response = chat_model.invoke(formatted_prompt.to_messages())
        raw_content = response.content if hasattr(response, "content") else str(response)
        parsed_proposal = parser.parse(raw_content)

    except Exception as e:
        logger.warning(f"LLM plan generation or parsing failed: {str(e)}")
        if fallback_to_default:
            default_plan = build_default_experiment_config(profile)
            return default_plan, {
                "source": "deterministic_fallback",
                "reason": f"LLM invocation or schema parsing error: {str(e)}",
                "rationale": "Fallback to deterministic defaults due to LLM parsing error.",
            }
        raise ExperimentPlanError(f"LLM plan generation failed: {str(e)}") from e

    # 4. Strict allowlist and constraint validation
    validation_res = validate_experiment_proposal(parsed_proposal, profile)

    if not validation_res.is_valid:
        logger.warning(f"LLM proposed invalid plan: {validation_res.errors}")
        if fallback_to_default:
            default_plan = build_default_experiment_config(profile)
            return default_plan, {
                "source": "deterministic_fallback",
                "reason": f"Plan rejected due to constraint violations: {validation_res.errors}",
                "rejected_proposal": parsed_proposal.model_dump(),
                "validation_errors": validation_res.errors,
                "rationale": "Fallback to deterministic defaults following plan rejection.",
            }
        raise ExperimentPlanError(f"LLM plan rejected: {validation_res.errors}")

    # 5. Convert to valid ExperimentConfig
    config = convert_proposal_to_experiment_config(parsed_proposal, profile)
    metadata = {
        "source": "llm",
        "rationale": parsed_proposal.rationale,
        "data_hygiene_observations": parsed_proposal.data_hygiene_observations,
        "warnings": validation_res.warnings,
    }
    return config, metadata


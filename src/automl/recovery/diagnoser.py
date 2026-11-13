"""Adaptive failure diagnoser, root cause analyzer, and recovery rule engine."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from automl.core.logging import get_logger
from automl.planner.provider import get_llm_provider
from automl.recovery.schemas import (
    ActionType,
    FailureCategory,
    FailureDiagnosis,
    LLMDiagnosisRecommendation,
    RecoveryAction,
)
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    ImputationStrategy,
    ModelFamily,
    RunStatus,
    ScalingStrategy,
)

logger = get_logger("recovery.diagnoser")


# Allowed hyperparameter keys per model family to prevent arbitrary injection
ALLOWED_HYPERPARAMETERS: Dict[ModelFamily, Dict[str, Tuple[type, Any, Any]]] = {
    ModelFamily.DUMMY_CLASSIFIER: {
        "strategy": (str, None, None),
    },
    ModelFamily.DUMMY_REGRESSOR: {
        "strategy": (str, None, None),
    },
    ModelFamily.LOGISTIC_REGRESSION: {
        "max_iter": (int, 10, 10000),
        "C": (float, 1e-4, 1000.0),
        "solver": (str, None, None),
    },
    ModelFamily.RANDOM_FOREST_CLASSIFIER: {
        "n_estimators": (int, 5, 500),
        "max_depth": (int, 1, 50),
        "min_samples_split": (int, 2, 20),
    },
    ModelFamily.GRADIENT_BOOSTING_CLASSIFIER: {
        "n_estimators": (int, 5, 500),
        "max_depth": (int, 1, 20),
        "learning_rate": (float, 1e-4, 1.0),
    },
    ModelFamily.RIDGE_REGRESSION: {
        "alpha": (float, 1e-5, 1000.0),
    },
    ModelFamily.RANDOM_FOREST_REGRESSOR: {
        "n_estimators": (int, 5, 500),
        "max_depth": (int, 1, 50),
        "min_samples_split": (int, 2, 20),
    },
    ModelFamily.GRADIENT_BOOSTING_REGRESSOR: {
        "n_estimators": (int, 5, 500),
        "max_depth": (int, 1, 20),
        "learning_rate": (float, 1e-4, 1.0),
    },
}


DIAGNOSIS_SYSTEM_PROMPT = """You are an expert AutoML Failure Diagnostic and Recovery Assistant.
An automated machine learning experiment failed during training or cross-validation.
Analyze the error message and failed candidate details.
Classify the failure and recommend a safe, minimal parameter adjustment using the provided schema.

STRICT BOUNDARIES:
1. You MUST ONLY recommend adjustments from the standard allowlist: max_iter, C, alpha, n_estimators, max_depth.
2. You MUST NOT propose new model families, arbitrary code, or changes to held-out test splits.
3. You MUST output strictly conforming JSON matching the provided schema.
"""


def categorize_failure(
    error_msg: str,
    candidate_runs: Optional[List[CandidateRunResult]] = None,
    plan: Optional[ExperimentConfig] = None,
) -> Tuple[FailureCategory, str, bool, List[str]]:
    """Analyze error messages and candidate outcomes to diagnose root cause and recovery feasibility.

    Args:
        error_msg: Primary error string or exception message.
        candidate_runs: Optional list of individual candidate run results.
        plan: Optional current experiment configuration.

    Returns:
        Tuple[FailureCategory, str, bool, List[str]]:
            - FailureCategory
            - Root cause summary
            - is_recoverable boolean
            - List of affected candidate names
    """
    msg_lower = (error_msg or "").lower()
    affected_candidates: List[str] = []

    # Check candidate runs for candidate-specific failures
    if candidate_runs:
        for run in candidate_runs:
            if run.status == RunStatus.FAILED:
                affected_candidates.append(run.candidate_name)
                if run.error_message:
                    msg_lower += " " + run.error_message.lower()

    # 1. Convergence Errors
    convergence_patterns = [
        "convergence",
        "did not converge",
        "iterations reached limit",
        "max_iter reached",
        "lbfgs failed to converge",
        "line search failed",
        "stochastic gradient descent reached max",
        "convergencewarning",
    ]
    if any(p in msg_lower for p in convergence_patterns):
        return (
            FailureCategory.CONVERGENCE,
            "Optimization algorithm reached iteration ceiling before convergence was achieved.",
            True,
            affected_candidates,
        )

    # 2. Resource / Memory Errors
    resource_patterns = [
        "memoryerror",
        "out of memory",
        "allocate",
        "bytes for an array",
        "oom",
        "resource exhaustion",
        "cuda out of memory",
        "cannot allocate memory",
    ]
    if any(p in msg_lower for p in resource_patterns):
        return (
            FailureCategory.RESOURCE,
            "System memory or resource limits exceeded during candidate fitting or fold splitting.",
            True,
            affected_candidates,
        )

    # 3. Timeout Errors
    timeout_patterns = [
        "timeout",
        "timeouterror",
        "reached time limit",
        "timed out",
        "total time limit exceeded",
        "execution time exceeded",
        "deadline exceeded",
        "skipped due to experiment total time limit",
    ]
    if any(p in msg_lower for p in timeout_patterns):
        return (
            FailureCategory.TIMEOUT,
            "Candidate training or total experiment execution exceeded configured time ceiling.",
            True,
            affected_candidates,
        )

    # 4. Invalid Hyperparameters
    hp_patterns = [
        "invalid parameter",
        "unexpected keyword argument",
        "unknown parameter",
        "parameter of",
        "parameter must be",
        "must be a float",
        "must be an int",
        "must be a str among",
        "must be in the range",
        "c must be",
        "alpha must be",
        "penalty and solver",
        "min_samples_split must be",
        "hyperparameter",
        "solver not supported",
    ]
    if any(p in msg_lower for p in hp_patterns):
        return (
            FailureCategory.INVALID_HYPERPARAMETERS,
            "Candidate model configured with invalid hyperparameter values, unapproved types, or incompatible arguments.",
            True,
            affected_candidates,
        )

    # 5. Preprocessing / Data Errors
    data_patterns = [
        "input contains nan",
        "could not convert string to float",
        "found array with 0 sample",
        "0 sample(s)",
        "unknown label",
        "empty data",
        "y contains 1 class",
        "class count in training set",
    ]
    if any(p in msg_lower for p in data_patterns):
        is_recoverable = not ("0 sample" in msg_lower or "empty data" in msg_lower or "y contains 1 class" in msg_lower)
        return (
            FailureCategory.PREPROCESSING_DATA,
            "Dataset hygiene or feature encoding issue encountered during pipeline execution.",
            is_recoverable,
            affected_candidates,
        )

    # 6. Unsupported Configuration / Model
    unsupported_patterns = [
        "unsupported model",
        "not in allowlist",
        "model_family not supported",
        "task_type mismatch",
        "does not match detected task",
    ]
    if any(p in msg_lower for p in unsupported_patterns):
        return (
            FailureCategory.UNSUPPORTED_CONFIGURATION,
            "Experiment plan contains unallowlisted model families or incompatible task settings.",
            True,
            affected_candidates,
        )

    # 7. Unknown Runtime Error
    return (
        FailureCategory.UNKNOWN,
        f"Unclassified execution failure: {error_msg[:150] if error_msg else 'Unknown error'}",
        False,  # Unrecoverable by default to prevent blind retries
        affected_candidates,
    )


def generate_deterministic_recovery(
    diagnosis: FailureDiagnosis,
    plan: ExperimentConfig,
) -> List[RecoveryAction]:
    """Produce concrete, safe deterministic recovery actions based on failure diagnosis.

    Args:
        diagnosis: FailureDiagnosis report.
        plan: Current ExperimentConfig.

    Returns:
        List[RecoveryAction]: Prioritized, validated recovery actions.
    """
    actions: List[RecoveryAction] = []
    category = diagnosis.category

    # If unrecoverable, return termination action
    if not diagnosis.is_recoverable:
        return [
            RecoveryAction(
                action_type=ActionType.TERMINATE_UNRECOVERABLE,
                rationale=f"Failure category '{category.value}' is unrecoverable. Halting workflow safely.",
                is_safe=True,
            )
        ]

    # Target specific candidates if identified, else all candidates
    target_names = set(diagnosis.affected_candidates)
    target_candidates = [
        c for c in plan.candidates if not target_names or c.name in target_names
    ]
    if not target_candidates:
        target_candidates = plan.candidates

    # 1. Convergence Recovery
    if category == FailureCategory.CONVERGENCE:
        for cand in target_candidates:
            if cand.model_family == ModelFamily.LOGISTIC_REGRESSION:
                curr_iter = int(cand.hyperparameters.get("max_iter", 1000))
                new_iter = min(curr_iter * 2, 5000)
                actions.append(
                    RecoveryAction(
                        action_type=ActionType.ADJUST_HYPERPARAMETERS,
                        target_candidate=cand.name,
                        target_model_family=cand.model_family,
                        parameter_adjustments={"max_iter": new_iter},
                        rationale=f"Increase max_iter from {curr_iter} to {new_iter} to allow optimization convergence.",
                    )
                )
            elif cand.model_family in (
                ModelFamily.GRADIENT_BOOSTING_CLASSIFIER,
                ModelFamily.GRADIENT_BOOSTING_REGRESSOR,
            ):
                curr_lr = float(cand.hyperparameters.get("learning_rate", 0.1))
                new_lr = round(max(curr_lr * 0.5, 0.01), 4)
                actions.append(
                    RecoveryAction(
                        action_type=ActionType.ADJUST_HYPERPARAMETERS,
                        target_candidate=cand.name,
                        target_model_family=cand.model_family,
                        parameter_adjustments={"learning_rate": new_lr},
                        rationale=f"Reduce learning_rate to {new_lr} to stabilize gradient optimization.",
                    )
                )

    # 2. Resource Exhaustion Recovery
    elif category == FailureCategory.RESOURCE:
        for cand in target_candidates:
            adjustments: Dict[str, Any] = {}
            if "n_estimators" in cand.hyperparameters:
                curr_n = int(cand.hyperparameters["n_estimators"])
                adjustments["n_estimators"] = max(int(curr_n * 0.5), 15)
            if "max_depth" in cand.hyperparameters or cand.model_family in (
                ModelFamily.RANDOM_FOREST_CLASSIFIER,
                ModelFamily.RANDOM_FOREST_REGRESSOR,
            ):
                curr_depth = cand.hyperparameters.get("max_depth")
                adjustments["max_depth"] = 5 if (curr_depth is None or curr_depth > 5) else curr_depth

            if adjustments:
                actions.append(
                    RecoveryAction(
                        action_type=ActionType.REDUCE_RESOURCES,
                        target_candidate=cand.name,
                        target_model_family=cand.model_family,
                        parameter_adjustments=adjustments,
                        rationale="Halve tree count and cap max_depth to reduce memory consumption.",
                    )
                )

        # Also reduce CV folds if resource-constrained
        if plan.cv_folds > 2:
            actions.append(
                RecoveryAction(
                    action_type=ActionType.REDUCE_RESOURCES,
                    parameter_adjustments={"cv_folds": max(plan.cv_folds - 1, 2)},
                    rationale=f"Reduce cv_folds from {plan.cv_folds} to {max(plan.cv_folds - 1, 2)} to reduce data partition copies.",
                )
            )

    # 3. Timeout Recovery
    elif category == FailureCategory.TIMEOUT:
        curr_time = plan.max_total_time_seconds
        new_time = min(int(curr_time * 1.5) + 30, 900)
        actions.append(
            RecoveryAction(
                action_type=ActionType.INCREASE_TIMEOUT,
                parameter_adjustments={"max_total_time_seconds": new_time},
                rationale=f"Increase experiment time limit from {curr_time}s to {new_time}s.",
            )
        )
        # Also reduce ensemble complexity for candidates that took too long
        for cand in target_candidates:
            if "n_estimators" in cand.hyperparameters:
                curr_n = int(cand.hyperparameters["n_estimators"])
                if curr_n > 25:
                    actions.append(
                        RecoveryAction(
                            action_type=ActionType.REDUCE_RESOURCES,
                            target_candidate=cand.name,
                            target_model_family=cand.model_family,
                            parameter_adjustments={"n_estimators": 25},
                            rationale="Cap n_estimators to 25 to complete training within time budget.",
                        )
                    )

    # 4. Invalid Hyperparameters Recovery
    elif category == FailureCategory.INVALID_HYPERPARAMETERS:
        for cand in target_candidates:
            fixed_params: Dict[str, Any] = {}
            for k, v in cand.hyperparameters.items():
                if k == "C" and isinstance(v, (int, float)) and v <= 0:
                    fixed_params["C"] = 1.0
                elif k == "alpha" and isinstance(v, (int, float)) and v < 0:
                    fixed_params["alpha"] = 1.0
                elif k == "n_estimators" and isinstance(v, (int, float)) and v < 1:
                    fixed_params["n_estimators"] = 50
                elif k == "min_samples_split" and isinstance(v, (int, float)) and v < 2:
                    fixed_params["min_samples_split"] = 2

            if fixed_params:
                actions.append(
                    RecoveryAction(
                        action_type=ActionType.ADJUST_HYPERPARAMETERS,
                        target_candidate=cand.name,
                        target_model_family=cand.model_family,
                        parameter_adjustments=fixed_params,
                        rationale=f"Reset invalid hyperparameters {list(fixed_params.keys())} to safe allowlisted defaults.",
                    )
                )
            else:
                # If cannot fix parameters automatically, skip the candidate safely
                actions.append(
                    RecoveryAction(
                        action_type=ActionType.SKIP_CANDIDATE,
                        target_candidate=cand.name,
                        target_model_family=cand.model_family,
                        rationale="Skip candidate with unfixable invalid hyperparameters.",
                    )
                )

    # 5. Preprocessing Data Hygiene Recovery
    elif category == FailureCategory.PREPROCESSING_DATA:
        actions.append(
            RecoveryAction(
                action_type=ActionType.ADJUST_PREPROCESSING,
                parameter_adjustments={
                    "imputation_strategy": ImputationStrategy.MEDIAN.value,
                    "encode_categoricals": True,
                    "scaling_strategy": ScalingStrategy.STANDARD.value,
                },
                rationale="Apply median imputation, categorical encoding, and standard scaling to resolve data hygiene errors.",
            )
        )

    # 6. Unsupported Configuration Recovery
    elif category == FailureCategory.UNSUPPORTED_CONFIGURATION:
        for cand in target_candidates:
            actions.append(
                RecoveryAction(
                    action_type=ActionType.SKIP_CANDIDATE,
                    target_candidate=cand.name,
                    target_model_family=cand.model_family,
                    rationale="Remove unsupported model candidate from experiment plan.",
                )
            )

    return actions


def validate_recovery_action(
    action: RecoveryAction,
    plan: ExperimentConfig,
) -> bool:
    """Validate proposed recovery action against safety boundaries and catalogue allowlists.

    Args:
        action: RecoveryAction to validate.
        plan: Current ExperimentConfig.

    Returns:
        bool: True if safe and approved, False otherwise.
    """
    if action.action_type in (ActionType.TERMINATE_UNRECOVERABLE, ActionType.SKIP_CANDIDATE):
        return True

    if action.action_type == ActionType.INCREASE_TIMEOUT:
        time_adj = action.parameter_adjustments.get("max_total_time_seconds")
        if time_adj is not None:
            return isinstance(time_adj, int) and 30 <= time_adj <= 1800
        return True

    if action.action_type == ActionType.ADJUST_PREPROCESSING:
        # Preprocessing adjustments must be valid enums/booleans
        for k, v in action.parameter_adjustments.items():
            if k == "imputation_strategy" and v not in [e.value for e in ImputationStrategy]:
                return False
            if k == "scaling_strategy" and v not in [e.value for e in ScalingStrategy]:
                return False
            if k in ("encode_categoricals", "drop_high_cardinality") and not isinstance(v, bool):
                return False
        return True

    if action.action_type in (ActionType.ADJUST_HYPERPARAMETERS, ActionType.REDUCE_RESOURCES):
        # Global resource adjustment (e.g. cv_folds)
        if "cv_folds" in action.parameter_adjustments and not action.target_candidate:
            folds = action.parameter_adjustments["cv_folds"]
            return isinstance(folds, int) and 2 <= folds <= 10

        # Target candidate must exist
        target_name = action.target_candidate
        candidate = next((c for c in plan.candidates if c.name == target_name), None)
        if not candidate:
            return False

        allowed_spec = ALLOWED_HYPERPARAMETERS.get(candidate.model_family, {})

        for param_key, param_val in action.parameter_adjustments.items():
            if param_key not in allowed_spec:
                logger.warning(
                    f"Rejected recovery action: '{param_key}' is not allowed for {candidate.model_family.value}"
                )
                return False

            expected_type, min_val, max_val = allowed_spec[param_key]
            # Type check
            if not isinstance(param_val, expected_type) or isinstance(param_val, bool) and expected_type != bool:
                return False
            # Bounds check if applicable
            if min_val is not None and param_val < min_val:
                return False
            if max_val is not None and param_val > max_val:
                return False

        return True

    return False


def apply_recovery_actions_to_plan(
    plan: ExperimentConfig,
    actions: List[RecoveryAction],
) -> Tuple[ExperimentConfig, List[str]]:
    """Apply validated recovery actions to produce an adapted experiment plan.

    Args:
        plan: Base ExperimentConfig.
        actions: Validated recovery actions.

    Returns:
        Tuple[ExperimentConfig, List[str]]: Modified plan and list of applied modification summaries.
    """
    applied_summaries: List[str] = []
    # Work on a copy of candidates and plan
    new_candidates = list(plan.candidates)
    candidates_to_remove = set()
    new_max_time = plan.max_total_time_seconds
    new_cv_folds = plan.cv_folds
    new_preprocessing = plan.preprocessing.model_copy()

    for action in actions:
        if not action.is_safe or not validate_recovery_action(action, plan):
            logger.warning(f"Skipping unvalidated recovery action: {action.action_type.value}")
            continue

        if action.action_type == ActionType.SKIP_CANDIDATE:
            if action.target_candidate:
                candidates_to_remove.add(action.target_candidate)
                applied_summaries.append(f"Skipped candidate '{action.target_candidate}': {action.rationale}")

        elif action.action_type in (ActionType.ADJUST_HYPERPARAMETERS, ActionType.REDUCE_RESOURCES):
            target_name = action.target_candidate
            for idx, c in enumerate(new_candidates):
                if c.name == target_name:
                    updated_hp = dict(c.hyperparameters)
                    updated_hp.update(action.parameter_adjustments)
                    new_candidates[idx] = c.model_copy(update={"hyperparameters": updated_hp})
                    applied_summaries.append(
                        f"Adjusted '{target_name}' parameters {action.parameter_adjustments}: {action.rationale}"
                    )

            # Global adjustments like cv_folds
            if "cv_folds" in action.parameter_adjustments:
                new_cv_folds = int(action.parameter_adjustments["cv_folds"])
                applied_summaries.append(f"Updated cv_folds to {new_cv_folds}: {action.rationale}")

        elif action.action_type == ActionType.INCREASE_TIMEOUT:
            if "max_total_time_seconds" in action.parameter_adjustments:
                new_max_time = int(action.parameter_adjustments["max_total_time_seconds"])
                applied_summaries.append(f"Updated max_total_time_seconds to {new_max_time}: {action.rationale}")

        elif action.action_type == ActionType.ADJUST_PREPROCESSING:
            new_preprocessing = new_preprocessing.model_copy(update=action.parameter_adjustments)
            applied_summaries.append(f"Updated preprocessing with {action.parameter_adjustments}: {action.rationale}")

    # Remove skipped candidates
    final_candidates = [c for c in new_candidates if c.name not in candidates_to_remove]

    # Ensure at least one candidate remains; if all removed, keep baseline
    if not final_candidates:
        baseline = next((c for c in plan.candidates if c.is_baseline), None)
        if baseline:
            final_candidates = [baseline]
            applied_summaries.append(f"Retained baseline candidate '{baseline.name}'.")

    updated_plan = plan.model_copy(
        update={
            "candidates": final_candidates,
            "max_total_time_seconds": new_max_time,
            "cv_folds": new_cv_folds,
            "preprocessing": new_preprocessing,
        }
    )

    return updated_plan, applied_summaries


def consult_llm_for_diagnosis(
    error_msg: str,
    plan: ExperimentConfig,
    candidate_runs: Optional[List[CandidateRunResult]] = None,
    llm: Optional[Any] = None,
) -> Optional[LLMDiagnosisRecommendation]:
    """Consult LangChain provider for structured failure diagnosis and recommended adjustment.

    Args:
        error_msg: Failure error message.
        plan: Current ExperimentConfig.
        candidate_runs: Optional candidate run results.
        llm: Optional LangChain chat model.

    Returns:
        Optional[LLMDiagnosisRecommendation]: Validated structured diagnosis recommendation, or None.
    """
    chat_model = get_llm_provider(custom_llm=llm)
    if chat_model is None:
        return None

    failed_runs_desc = []
    if candidate_runs:
        for r in candidate_runs:
            if r.status == RunStatus.FAILED:
                failed_runs_desc.append(
                    f"Candidate '{r.candidate_name}' ({r.model_family.value}): error='{r.error_message}' params={r.parameters}"
                )

    failed_info = "\n".join(failed_runs_desc) if failed_runs_desc else "No candidate run details available."

    parser = PydanticOutputParser(pydantic_object=LLMDiagnosisRecommendation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", DIAGNOSIS_SYSTEM_PROMPT),
        (
            "human",
            "An experiment failed with the following error:\n\n"
            "ERROR MESSAGE:\n{error_msg}\n\n"
            "FAILED CANDIDATE RUNS:\n{failed_info}\n\n"
            "CURRENT PLAN:\n"
            "- Task Type: {task_type}\n"
            "- Primary Metric: {primary_metric}\n"
            "- Candidates: {candidates}\n\n"
            "{format_instructions}\n"
        ),
    ])

    formatted_prompt = prompt.format_prompt(
        error_msg=error_msg,
        failed_info=failed_info,
        task_type=plan.task_type.value,
        primary_metric=plan.primary_metric.value,
        candidates=[c.name for c in plan.candidates],
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response = chat_model.invoke(formatted_prompt.to_messages())
        raw_content = response.content if hasattr(response, "content") else str(response)
        # Strip markdown fences if present
        clean_content = raw_content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]

        recommendation = parser.parse(clean_content.strip())
        logger.info(f"LLM failure recommendation received: category={recommendation.category.value}")
        return recommendation
    except Exception as e:
        logger.warning(f"LLM failure diagnosis consultation failed: {str(e)}")
        return None


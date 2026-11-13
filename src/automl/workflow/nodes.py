"""Graph node implementations for validation, profiling, planning, execution, and error handling."""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from automl.core.exceptions import (
    AutoMLError,
    DatasetValidationError,
    ExperimentPlanError,
    ModelTrainingError,
)
from automl.core.logging import get_logger
from automl.profiler.loader import load_dataset
from automl.profiler.profiler import profile_dataset
from automl.profiler.schemas import TaskType
from automl.profiler.validator import validate_dataset
from automl.planner.explainer import explain_experiment_results
from automl.planner.proposer import (
    convert_proposal_to_experiment_config,
    generate_initial_proposal,
    propose_experiment_plan,
    repair_experiment_plan,
    validate_experiment_proposal,
)
from automl.planner.provider import get_llm_provider
from automl.planner.schemas import LLMExperimentProposal
from automl.trainer.engine import ExperimentEngine, build_default_experiment_config
from automl.trainer.schemas import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    MetricName,
)
from automl.workflow.registry import DatasetRegistry, LLMRegistry
from automl.workflow.state import (
    ErrorCategory,
    ExperimentWorkflowState,
    WorkflowStatus,
)

logger = get_logger("workflow.nodes")


def validate_input_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Validate dataset source, format, and target column presence."""
    trace = list(state.get("execution_trace", []))
    trace.append("validate_input")
    logger.info("Executing node: validate_input")

    try:
        ref = state.get("dataset_ref")
        target_col = state.get("target_column")

        if not ref:
            raise DatasetValidationError("No dataset_ref provided in workflow state.")

        source = DatasetRegistry.resolve(ref)
        df = load_dataset(source)
        validate_dataset(df, target_column=target_col)

        metadata = {
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": list(df.columns[:20]),
        }

        return {
            "dataset_metadata": metadata,
            "workflow_status": WorkflowStatus.PROFILING,
            "error_category": ErrorCategory.NONE,
            "error_message": None,
            "execution_trace": trace,
        }

    except Exception as e:
        logger.warning(f"validate_input failed: {str(e)}")
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_DATA,
            "error_message": str(e),
            "is_recoverable": False,
            "execution_trace": trace,
        }


def profile_data_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Extract comprehensive dataset profile and infer task type."""
    trace = list(state.get("execution_trace", []))
    trace.append("profile_data")
    logger.info("Executing node: profile_data")

    try:
        source = DatasetRegistry.resolve(state["dataset_ref"])
        profile = profile_dataset(
            source,
            target_column=state.get("target_column"),
            task_type_override=state.get("task_type_override"),
        )
        return {
            "profile": profile,
            "workflow_status": WorkflowStatus.PLANNING,
            "execution_trace": trace,
        }
    except Exception as e:
        logger.warning(f"profile_data failed: {str(e)}")
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_DATA,
            "error_message": f"Profiling failed: {str(e)}",
            "is_recoverable": False,
            "execution_trace": trace,
        }


def validate_user_constraints(
    constraints: Dict[str, Any],
    task_type: Optional[TaskType] = None,
) -> None:
    """Validate user constraints against boundaries and detected task type."""
    if not constraints:
        return

    # 1. cv_folds
    if "cv_folds" in constraints and constraints["cv_folds"] is not None:
        cv_folds = constraints["cv_folds"]
        if not isinstance(cv_folds, int) or isinstance(cv_folds, bool) or not (2 <= cv_folds <= 10):
            raise ExperimentPlanError(f"User constraint 'cv_folds' must be an integer between 2 and 10 (got {cv_folds}).")

    # 2. test_size
    if "test_size" in constraints and constraints["test_size"] is not None:
        test_size = constraints["test_size"]
        if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or not (0.05 <= test_size <= 0.50):
            raise ExperimentPlanError(f"User constraint 'test_size' must be between 0.05 and 0.50 (got {test_size}).")

    # 3. random_seed
    if "random_seed" in constraints and constraints["random_seed"] is not None:
        random_seed = constraints["random_seed"]
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise ExperimentPlanError(f"User constraint 'random_seed' must be an integer (got {random_seed}).")

    # 4. primary_metric
    if "primary_metric" in constraints and constraints["primary_metric"] is not None:
        metric = constraints["primary_metric"]
        if isinstance(metric, str):
            try:
                metric = MetricName(metric.lower())
            except ValueError:
                valid_metrics = [m.value for m in MetricName]
                raise ExperimentPlanError(
                    f"User constraint primary_metric '{metric}' is invalid. Allowed metrics: {valid_metrics}"
                )

        if task_type is not None:
            if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                if metric not in CLASSIFICATION_METRICS:
                    raise ExperimentPlanError(
                        f"User constraint primary_metric '{metric.value}' is incompatible with classification task. "
                        f"Allowed metrics: {[m.value for m in CLASSIFICATION_METRICS]}"
                    )
            elif task_type == TaskType.REGRESSION:
                if metric not in REGRESSION_METRICS:
                    raise ExperimentPlanError(
                        f"User constraint primary_metric '{metric.value}' is incompatible with regression task. "
                        f"Allowed metrics: {[m.value for m in REGRESSION_METRICS]}"
                    )


def create_plan_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Generate experiment plan (via LLM planner if requested, else deterministic baseline)."""
    trace = list(state.get("execution_trace", []))
    trace.append("create_plan")
    logger.info("Executing node: create_plan")

    try:
        # Use existing plan if already supplied by user/caller
        if state.get("plan") is not None:
            return {
                "plan": state["plan"],
                "plan_metadata": state.get("plan_metadata", {"source": "user_provided"}),
                "execution_trace": trace,
            }

        profile = state.get("profile")
        if profile is None or profile.target_profile is None:
            raise ExperimentPlanError("Cannot create plan: dataset profile or target profile is missing.")

        task_type = profile.target_profile.detected_task_type
        user_constraints = state.get("user_constraints") or {}

        # Validate user constraints against detected task and parameter bounds
        validate_user_constraints(user_constraints, task_type=task_type)

        primary_metric_val = user_constraints.get("primary_metric")
        if isinstance(primary_metric_val, str):
            primary_metric_val = MetricName(primary_metric_val.lower())

        cv_folds_val = user_constraints.get("cv_folds") if user_constraints.get("cv_folds") is not None else 5
        test_size_val = user_constraints.get("test_size") if user_constraints.get("test_size") is not None else 0.2
        random_seed_val = user_constraints.get("random_seed") if user_constraints.get("random_seed") is not None else 42

        # Effective constraints dict to pass downstream
        effective_constraints = dict(user_constraints)
        if primary_metric_val:
            effective_constraints["primary_metric"] = primary_metric_val
        effective_constraints["cv_folds"] = cv_folds_val
        effective_constraints["test_size"] = test_size_val
        effective_constraints["random_seed"] = random_seed_val

        if state.get("use_llm_planner", False):
            # Check if LLM is available
            custom_llm = LLMRegistry.resolve(state.get("llm_ref")) or state.get("llm_instance")
            chat_model = get_llm_provider(custom_llm=custom_llm)
            if chat_model is None:
                # LLM unavailable -> fallback immediately (independent of LLM availability)
                default_plan = build_default_experiment_config(
                    profile,
                    primary_metric=primary_metric_val,
                    cv_folds=cv_folds_val,
                    test_size=test_size_val,
                    random_seed=random_seed_val,
                )
                if "max_models" in user_constraints and user_constraints["max_models"] is not None:
                    max_m = user_constraints["max_models"]
                    if len(default_plan.candidates) > max_m:
                        default_plan.candidates = default_plan.candidates[:max_m]

                plan_meta = {
                    "source": "deterministic_fallback",
                    "reason": "No LLM provider or credentials available.",
                    "user_constraints_applied": bool(user_constraints),
                }
                trace.append("plan_attempt_1: source=fallback, outcome=valid, error_summary=none")
                return {
                    "plan": default_plan,
                    "plan_metadata": plan_meta,
                    "current_proposal": None,
                    "validation_errors": [],
                    "plan_retry_count": 0,
                    "execution_trace": trace,
                }

            # Generate initial proposal via LLM
            proposal, err_msg = generate_initial_proposal(
                profile=profile,
                llm=chat_model,
                user_constraints=effective_constraints,
            )

            if proposal is not None:
                current_proposal = proposal.model_dump()
                return {
                    "current_proposal": current_proposal,
                    "plan": None,
                    "plan_retry_count": 0,
                    "validation_errors": [],
                    "execution_trace": trace,
                }
            else:
                return {
                    "current_proposal": None,
                    "plan": None,
                    "plan_retry_count": 0,
                    "validation_errors": [f"[schema_error] Initial proposal generation failed: {err_msg}"],
                    "execution_trace": trace,
                }
        else:
            plan = build_default_experiment_config(
                profile,
                primary_metric=primary_metric_val,
                cv_folds=cv_folds_val,
                test_size=test_size_val,
                random_seed=random_seed_val,
            )
            plan_meta = {
                "source": "deterministic",
                "user_constraints_applied": bool(user_constraints),
            }

            if "max_models" in user_constraints and user_constraints["max_models"] is not None:
                max_m = user_constraints["max_models"]
                if len(plan.candidates) > max_m:
                    plan.candidates = plan.candidates[:max_m]

            return {
                "plan": plan,
                "plan_metadata": plan_meta,
                "current_proposal": None,
                "validation_errors": [],
                "plan_retry_count": 0,
                "execution_trace": trace,
            }
    except Exception as e:
        logger.warning(f"create_plan failed: {str(e)}")
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": f"Plan creation failed: {str(e)}",
            "is_recoverable": False,
            "execution_trace": trace,
        }


def validate_plan_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Verify plan compatibility against profile and resource bounds."""
    trace = list(state.get("execution_trace", []))
    trace.append("validate_plan")
    logger.info("Executing node: validate_plan")

    if state.get("workflow_status") == WorkflowStatus.FAILED:
        return {
            "execution_trace": trace,
        }

    profile = state.get("profile")
    if profile is None or profile.target_profile is None:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_DATA,
            "error_message": "Dataset profile or target profile is missing.",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    user_constraints = state.get("user_constraints") or {}
    cv_folds_val = user_constraints.get("cv_folds") if user_constraints.get("cv_folds") is not None else 5
    test_size_val = user_constraints.get("test_size") if user_constraints.get("test_size") is not None else 0.2
    random_seed_val = user_constraints.get("random_seed") if user_constraints.get("random_seed") is not None else 42

    # Check if we are validating an in-flight LLM proposal
    if state.get("use_llm_planner", False) and state.get("plan") is None:
        attempt_num = state.get("plan_retry_count", 0) + 1
        curr_prop_dict = state.get("current_proposal")

        if curr_prop_dict is not None:
            try:
                proposal_obj = LLMExperimentProposal.model_validate(curr_prop_dict)
                val_result = validate_experiment_proposal(proposal_obj, profile)

                if val_result.is_valid:
                    # Successfully validated! Convert to ExperimentConfig
                    plan_cfg = convert_proposal_to_experiment_config(
                        proposal_obj,
                        profile,
                        test_size=test_size_val,
                        cv_folds=cv_folds_val,
                        random_seed=random_seed_val,
                        user_constraints=user_constraints,
                    )
                    trace.append(f"plan_attempt_{attempt_num}: source=llm, outcome=valid, error_summary=none")
                    plan_meta = {
                        "source": "llm",
                        "rationale": proposal_obj.rationale,
                        "data_hygiene_observations": proposal_obj.data_hygiene_observations,
                        "warnings": val_result.warnings,
                        "user_constraints_applied": bool(user_constraints),
                        "plan_retry_count": state.get("plan_retry_count", 0),
                    }
                    return {
                        "plan": plan_cfg,
                        "plan_metadata": plan_meta,
                        "validation_errors": [],
                        "workflow_status": WorkflowStatus.RUNNING,
                        "execution_trace": trace,
                    }
                else:
                    # Invalid proposal
                    err_summary = "; ".join(val_result.errors)
                    trace.append(f"plan_attempt_{attempt_num}: source=llm, outcome=invalid, error_summary={err_summary}")
                    return {
                        "validation_errors": val_result.errors,
                        "workflow_status": WorkflowStatus.PLANNING,
                        "execution_trace": trace,
                    }
            except Exception as e:
                err_summary = f"[schema_error] Proposal parsing/validation error: {str(e)}"
                trace.append(f"plan_attempt_{attempt_num}: source=llm, outcome=invalid, error_summary={err_summary}")
                return {
                    "validation_errors": [err_summary],
                    "workflow_status": WorkflowStatus.PLANNING,
                    "execution_trace": trace,
                }
        else:
            # Proposal generation or parsing failed
            errors = state.get("validation_errors") or ["[schema_error] Proposal is missing or unparseable."]
            err_summary = "; ".join(errors)
            trace.append(f"plan_attempt_{attempt_num}: source=llm, outcome=invalid, error_summary={err_summary}")
            return {
                "validation_errors": errors,
                "workflow_status": WorkflowStatus.PLANNING,
                "execution_trace": trace,
            }

    plan = state.get("plan")
    if plan is None:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": state.get("error_message") or "Experiment plan is missing.",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    detected_task = profile.target_profile.detected_task_type

    # Verify task type alignment
    if plan.task_type != detected_task:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": (
                f"Plan task_type '{plan.task_type}' does not match detected task_type "
                f"'{detected_task}'."
            ),
            "is_recoverable": False,
            "execution_trace": trace,
        }

    # Verify primary metric compatibility with detected task
    if detected_task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        if plan.primary_metric not in CLASSIFICATION_METRICS:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "error_category": ErrorCategory.INVALID_PLAN,
                "error_message": (
                    f"Plan primary_metric '{plan.primary_metric.value}' is invalid for classification task. "
                    f"Allowed metrics: {[m.value for m in CLASSIFICATION_METRICS]}"
                ),
                "is_recoverable": False,
                "execution_trace": trace,
            }
    elif detected_task == TaskType.REGRESSION:
        if plan.primary_metric not in REGRESSION_METRICS:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "error_category": ErrorCategory.INVALID_PLAN,
                "error_message": (
                    f"Plan primary_metric '{plan.primary_metric.value}' is invalid for regression task. "
                    f"Allowed metrics: {[m.value for m in REGRESSION_METRICS]}"
                ),
                "is_recoverable": False,
                "execution_trace": trace,
            }

    # Verify bounds: cv_folds, test_size, random_seed
    if not (2 <= plan.cv_folds <= 10):
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": f"cv_folds must be between 2 and 10 (got {plan.cv_folds}).",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    if not (0.05 <= plan.test_size <= 0.50):
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": f"test_size must be between 0.05 and 0.50 (got {plan.test_size}).",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    if not isinstance(plan.random_seed, int) or isinstance(plan.random_seed, bool):
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": f"random_seed must be an integer (got {type(plan.random_seed).__name__}).",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    if not plan.candidates:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": "Plan has 0 model candidates.",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    return {
        "workflow_status": WorkflowStatus.RUNNING,
        "execution_trace": trace,
    }


def repair_plan_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Prompt LLM to reflect and repair invalid experiment proposal using error feedback."""
    trace = list(state.get("execution_trace", []))
    trace.append("repair_plan")
    logger.info("Executing node: repair_plan")

    profile = state.get("profile")
    user_constraints = state.get("user_constraints") or {}
    prev_errors = state.get("validation_errors", [])
    current_retry = state.get("plan_retry_count", 0) + 1

    custom_llm = LLMRegistry.resolve(state.get("llm_ref")) or state.get("llm_instance")
    chat_model = get_llm_provider(custom_llm=custom_llm)
    repaired_proposal, repair_err = repair_experiment_plan(
        profile=profile,
        previous_proposal=state.get("current_proposal"),
        validation_errors=prev_errors,
        llm=chat_model,
        user_constraints=user_constraints,
    )

    if repaired_proposal is not None:
        return {
            "current_proposal": repaired_proposal.model_dump(),
            "plan_retry_count": current_retry,
            "validation_errors": [],
            "execution_trace": trace,
        }
    else:
        return {
            "current_proposal": None,
            "validation_errors": [f"[schema_error] Plan repair failed: {repair_err}"],
            "plan_retry_count": current_retry,
            "execution_trace": trace,
        }


def fallback_plan_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Produce safe deterministic default plan after reflection retries are exhausted."""
    trace = list(state.get("execution_trace", []))
    trace.append("fallback_plan")
    logger.info("Executing node: fallback_plan")

    profile = state.get("profile")
    user_constraints = state.get("user_constraints") or {}

    primary_metric_val = user_constraints.get("primary_metric")
    if isinstance(primary_metric_val, str):
        primary_metric_val = MetricName(primary_metric_val.lower())
    cv_folds_val = user_constraints.get("cv_folds") if user_constraints.get("cv_folds") is not None else 5
    test_size_val = user_constraints.get("test_size") if user_constraints.get("test_size") is not None else 0.2
    random_seed_val = user_constraints.get("random_seed") if user_constraints.get("random_seed") is not None else 42

    default_plan = build_default_experiment_config(
        profile,
        primary_metric=primary_metric_val,
        cv_folds=cv_folds_val,
        test_size=test_size_val,
        random_seed=random_seed_val,
    )
    if "max_models" in user_constraints and user_constraints["max_models"] is not None:
        max_m = user_constraints["max_models"]
        if len(default_plan.candidates) > max_m:
            default_plan.candidates = default_plan.candidates[:max_m]

    attempt_num = state.get("plan_retry_count", 0) + 1
    trace.append(f"plan_attempt_{attempt_num}: source=fallback, outcome=valid, error_summary=none")

    plan_meta = {
        "source": "deterministic_fallback",
        "reason": f"Plan retries exhausted ({state.get('plan_retry_count', 0)} attempts failed validation).",
        "validation_errors": state.get("validation_errors", []),
        "user_constraints_applied": bool(user_constraints),
        "plan_retry_count": state.get("plan_retry_count", 0),
    }

    return {
        "plan": default_plan,
        "plan_metadata": plan_meta,
        "validation_errors": [],
        "workflow_status": WorkflowStatus.RUNNING,
        "execution_trace": trace,
    }


def execute_experiments_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Execute cross-validation and candidate training with engine."""
    trace = list(state.get("execution_trace", []))
    trace.append("execute_experiments")
    logger.info("Executing node: execute_experiments")

    try:
        source = DatasetRegistry.resolve(state["dataset_ref"])
        engine = ExperimentEngine()
        result = engine.run_experiment(
            data=source,
            profile=state["profile"],
            config=state["plan"],
            enable_tracking=True,
        )

        return {
            "experiment_result": result,
            "candidate_runs": result.leaderboard,
            "workflow_status": WorkflowStatus.EVALUATING,
            "execution_trace": trace,
        }

    except Exception as e:
        logger.warning(f"execute_experiments encountered error: {str(e)}")
        # Check if error looks like a transient recoverable issue
        is_recoverable = isinstance(e, (TimeoutError, ConnectionError)) or "timeout" in str(e).lower()
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.RECOVERABLE_RUNTIME if is_recoverable else ErrorCategory.FATAL_RUNTIME,
            "error_message": str(e),
            "is_recoverable": is_recoverable,
            "execution_trace": trace,
        }


def evaluate_results_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Inspect experiment outcomes and check candidate survival."""
    trace = list(state.get("execution_trace", []))
    trace.append("evaluate_results")
    logger.info("Executing node: evaluate_results")

    runs = state.get("candidate_runs", [])
    if not runs:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.FATAL_RUNTIME,
            "error_message": "No candidate runs recorded.",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    successful = [r for r in runs if r.status.value == "success"]
    failed = [r for r in runs if r.status.value == "failed"]

    if not successful:
        # All candidates failed
        logger.warning(f"All {len(runs)} candidates failed.")
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.RECOVERABLE_RUNTIME,
            "error_message": "All model candidates failed during training.",
            "is_recoverable": True,
            "execution_trace": trace,
        }

    if failed:
        logger.info(f"Partial success: {len(successful)} candidates succeeded, {len(failed)} failed.")
        status = WorkflowStatus.PARTIAL_SUCCESS
    else:
        logger.info(f"Full success: all {len(successful)} candidates succeeded.")
        status = WorkflowStatus.SUCCESS

    return {
        "workflow_status": status,
        "execution_trace": trace,
    }


def handle_failure_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Bounded retry handler for explicitly classified recoverable failures."""
    trace = list(state.get("execution_trace", []))
    trace.append("handle_failure")
    logger.info("Executing node: handle_failure")

    retries = state.get("retry_count", 0) + 1
    max_retries = state.get("max_retries", 2)
    is_recoverable = state.get("is_recoverable", False)

    logger.info(f"handle_failure: retry {retries}/{max_retries}, recoverable={is_recoverable}")

    if is_recoverable and retries <= max_retries:
        logger.info(f"Attempting recoverable retry #{retries}...")
        # Adjust plan slightly for retry (e.g., increase max total time by 50%)
        current_plan = state.get("plan")
        if current_plan:
            adjusted_plan = current_plan.model_copy(
                update={"max_total_time_seconds": int(current_plan.max_total_time_seconds * 1.5)}
            )
        else:
            adjusted_plan = None

        return {
            "retry_count": retries,
            "plan": adjusted_plan,
            "workflow_status": WorkflowStatus.RUNNING,
            "execution_trace": trace,
        }

    # Non-recoverable or retries exhausted
    return {
        "retry_count": retries,
        "workflow_status": WorkflowStatus.FAILED,
        "is_recoverable": False,
        "execution_trace": trace,
    }


def rank_leaderboard_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Consolidate final leaderboard and identify best model candidate."""
    trace = list(state.get("execution_trace", []))
    trace.append("rank_leaderboard")
    logger.info("Executing node: rank_leaderboard")

    exp_result = state.get("experiment_result")
    best_name = exp_result.best_candidate_name if exp_result else None

    return {
        "best_candidate_name": best_name,
        "execution_trace": trace,
    }


def finalize_workflow_node(state: ExperimentWorkflowState) -> Dict[str, Any]:
    """Node: Wrap up execution trace, generate grounded result explanation, and set final state."""
    trace = list(state.get("execution_trace", []))
    trace.append("finalize_workflow")
    logger.info(f"Finalizing workflow with status: {state.get('workflow_status')}")

    explanation = None
    exp_res = state.get("experiment_result")
    if exp_res is not None:
        try:
            explanation = explain_experiment_results(exp_res)
        except Exception as e:
            logger.warning(f"Could not generate result explanation: {str(e)}")

    return {
        "result_explanation": explanation,
        "execution_trace": trace,
    }

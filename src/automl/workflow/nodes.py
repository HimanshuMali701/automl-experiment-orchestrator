"""Graph node implementations for validation, profiling, planning, execution, and error handling."""

from typing import Any, Dict
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
from automl.profiler.validator import validate_dataset
from automl.planner.explainer import explain_experiment_results
from automl.planner.proposer import propose_experiment_plan
from automl.trainer.engine import ExperimentEngine, build_default_experiment_config
from automl.workflow.registry import DatasetRegistry
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
        if profile is None:
            raise ExperimentPlanError("Cannot create plan: dataset profile is missing.")

        if state.get("use_llm_planner", False):
            plan, plan_meta = propose_experiment_plan(profile, fallback_to_default=True)
        else:
            plan = build_default_experiment_config(profile)
            plan_meta = {"source": "deterministic"}

        return {
            "plan": plan,
            "plan_metadata": plan_meta,
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

    plan = state.get("plan")
    profile = state.get("profile")

    if plan is None:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": "Experiment plan is missing.",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    if profile is None or profile.target_profile is None:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_DATA,
            "error_message": "Dataset profile or target profile is missing.",
            "is_recoverable": False,
            "execution_trace": trace,
        }

    # Verify task type alignment
    if plan.task_type != profile.target_profile.detected_task_type:
        return {
            "workflow_status": WorkflowStatus.FAILED,
            "error_category": ErrorCategory.INVALID_PLAN,
            "error_message": (
                f"Plan task_type '{plan.task_type}' does not match detected task_type "
                f"'{profile.target_profile.detected_task_type}'."
            ),
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

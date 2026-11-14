"""StateGraph assembly, conditional routing, and workflow execution engine."""

from typing import Any, Dict, Optional, Union
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from automl.core.logging import get_logger
from automl.profiler.schemas import TaskType
from automl.trainer.schemas import ExperimentConfig
from automl.workflow.checkpointer import SqliteCheckpointSaver, get_default_checkpointer
from automl.workflow.nodes import (
    approval_gate_node,
    create_plan_node,
    diagnose_failure_node,
    evaluate_results_node,
    execute_experiments_node,
    fallback_plan_node,
    finalize_workflow_node,
    handle_failure_node,
    profile_data_node,
    rank_leaderboard_node,
    repair_plan_node,
    validate_input_node,
    validate_plan_node,
)
from automl.workflow.registry import DatasetRegistry, LLMRegistry
from automl.workflow.state import (
    ErrorCategory,
    ExperimentWorkflowState,
    WorkflowStatus,
)

logger = get_logger("workflow.graph")


# --- Conditional Routing Functions ---

def route_after_input_validation(state: ExperimentWorkflowState) -> str:
    """Route to data profiling if input is valid, otherwise to failure handling."""
    if state.get("error_category") == ErrorCategory.NONE:
        return "profile_data"
    return "handle_failure"


def route_after_profiling(state: ExperimentWorkflowState) -> str:
    """Route to plan creation if profiling succeeded, otherwise to failure handling."""
    if state.get("profile") is not None:
        return "create_plan"
    return "handle_failure"


def route_after_plan_validation(state: ExperimentWorkflowState) -> str:
    """Route after plan validation: approval_gate if valid, repair if retries remain, fallback if exhausted."""
    if state.get("use_llm_planner", False):
        val_errors = state.get("validation_errors", [])
        if not val_errors and state.get("plan") is not None:
            return "approval_gate"

        retry_count = state.get("plan_retry_count", 0)
        max_retries = state.get("max_plan_retries", 2)
        if retry_count < max_retries:
            return "repair_plan"
        else:
            return "fallback_plan"
    else:
        if state.get("workflow_status") in (WorkflowStatus.RUNNING, WorkflowStatus.PLANNING) and state.get("plan") is not None:
            return "approval_gate"
        return "handle_failure"


def route_after_approval_gate(state: ExperimentWorkflowState) -> str:
    """Route after approval gate: finalize if rejected, else execute_experiments."""
    if (
        state.get("approval_status") == "rejected"
        or state.get("workflow_status") == WorkflowStatus.REJECTED
    ):
        return "finalize_workflow"
    return "execute_experiments"


def route_after_execution(state: ExperimentWorkflowState) -> str:
    """Route to evaluation if execution produced results, otherwise to failure diagnosis."""
    if state.get("workflow_status") == WorkflowStatus.EVALUATING:
        return "evaluate_results"
    return "diagnose_failure"


def route_after_evaluation(state: ExperimentWorkflowState) -> str:
    """Route to ranking if models succeeded (full or partial), otherwise to failure diagnosis."""
    status = state.get("workflow_status")
    if status in (WorkflowStatus.SUCCESS, WorkflowStatus.PARTIAL_SUCCESS):
        return "rank_leaderboard"
    return "diagnose_failure"


def route_after_diagnosis(state: ExperimentWorkflowState) -> str:
    """Route after failure diagnosis based on recoverability, retry budget, and candidate survival."""
    if state.get("is_recoverable") and state.get("retry_count", 0) < state.get("max_retries", 2):
        return "handle_failure"

    # Candidate-specific failure where other candidates succeeded
    runs = state.get("candidate_runs") or []
    if any(r.status.value == "success" for r in runs):
        return "rank_leaderboard"

    return "finalize_workflow"


def route_after_failure(state: ExperimentWorkflowState) -> str:
    """Route after failure handling: rerun experiments if retry prepared, else finalize."""
    if (
        state.get("is_recoverable")
        and state.get("retry_count", 0) <= state.get("max_retries", 2)
        and state.get("workflow_status") == WorkflowStatus.RUNNING
    ):
        return "execute_experiments"
    return "finalize_workflow"


# --- Graph Construction ---

def create_workflow_graph(checkpointer: Optional[Any] = None) -> Any:
    """Build and compile the LangGraph StateGraph for AutoML experiment orchestration.

    Checkpointer Design Decision:
    Uses `MemorySaver` by default for fast, in-memory execution and step-level state inspection.
    In distributed environments, a persistent checkpointer (e.g., SQLite or Postgres) can be passed.

    Returns:
        CompiledStateGraph: The compiled executable state graph.
    """
    workflow = StateGraph(ExperimentWorkflowState)

    # 1. Register nodes
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("profile_data", profile_data_node)
    workflow.add_node("create_plan", create_plan_node)
    workflow.add_node("validate_plan", validate_plan_node)
    workflow.add_node("repair_plan", repair_plan_node)
    workflow.add_node("fallback_plan", fallback_plan_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("execute_experiments", execute_experiments_node)
    workflow.add_node("evaluate_results", evaluate_results_node)
    workflow.add_node("diagnose_failure", diagnose_failure_node)
    workflow.add_node("handle_failure", handle_failure_node)
    workflow.add_node("rank_leaderboard", rank_leaderboard_node)
    workflow.add_node("finalize_workflow", finalize_workflow_node)

    # 2. Add edges & conditional transitions
    workflow.add_edge(START, "validate_input")

    workflow.add_conditional_edges(
        "validate_input",
        route_after_input_validation,
        {
            "profile_data": "profile_data",
            "handle_failure": "handle_failure",
        },
    )

    workflow.add_conditional_edges(
        "profile_data",
        route_after_profiling,
        {
            "create_plan": "create_plan",
            "handle_failure": "handle_failure",
        },
    )

    workflow.add_edge("create_plan", "validate_plan")

    workflow.add_conditional_edges(
        "validate_plan",
        route_after_plan_validation,
        {
            "approval_gate": "approval_gate",
            "repair_plan": "repair_plan",
            "fallback_plan": "fallback_plan",
            "handle_failure": "handle_failure",
        },
    )

    workflow.add_edge("repair_plan", "validate_plan")
    workflow.add_edge("fallback_plan", "approval_gate")

    workflow.add_conditional_edges(
        "approval_gate",
        route_after_approval_gate,
        {
            "execute_experiments": "execute_experiments",
            "finalize_workflow": "finalize_workflow",
        },
    )

    workflow.add_conditional_edges(
        "execute_experiments",
        route_after_execution,
        {
            "evaluate_results": "evaluate_results",
            "diagnose_failure": "diagnose_failure",
        },
    )

    workflow.add_conditional_edges(
        "evaluate_results",
        route_after_evaluation,
        {
            "rank_leaderboard": "rank_leaderboard",
            "diagnose_failure": "diagnose_failure",
        },
    )

    workflow.add_conditional_edges(
        "diagnose_failure",
        route_after_diagnosis,
        {
            "handle_failure": "handle_failure",
            "rank_leaderboard": "rank_leaderboard",
            "finalize_workflow": "finalize_workflow",
        },
    )

    workflow.add_conditional_edges(
        "handle_failure",
        route_after_failure,
        {
            "execute_experiments": "execute_experiments",
            "finalize_workflow": "finalize_workflow",
        },
    )

    workflow.add_edge("rank_leaderboard", "finalize_workflow")
    workflow.add_edge("finalize_workflow", END)

    saver = checkpointer or get_default_checkpointer()
    return workflow.compile(checkpointer=saver)


def run_orchestration_workflow(
    dataset_source: Any,
    target_column: Optional[str] = None,
    task_type_override: Optional[TaskType] = None,
    plan: Optional[ExperimentConfig] = None,
    use_llm_planner: bool = False,
    max_retries: int = 2,
    max_plan_retries: int = 2,
    thread_id: Optional[str] = None,
    user_constraints: Optional[Dict[str, Any]] = None,
    llm: Optional[Any] = None,
    requires_approval: bool = False,
    approval_status: Optional[str] = None,
    approval_feedback: Optional[str] = None,
    approved_plan: Optional[ExperimentConfig] = None,
    checkpointer: Optional[Any] = None,
) -> ExperimentWorkflowState:
    """Execute end-to-end AutoML workflow with LangGraph orchestration.

    Args:
        dataset_source: Filepath, bytes, or DataFrame.
        target_column: Target column for supervised prediction.
        task_type_override: Optional task type override.
        plan: Optional custom experiment plan.
        use_llm_planner: Whether to use LLM planner for plan creation.
        max_retries: Maximum bounded retry count for recoverable failures.
        max_plan_retries: Maximum bounded retry count for LLM plan reflection and repair.
        thread_id: Optional checkpointer thread ID.
        user_constraints: Optional user-specified constraints (metrics, folds, seed, test_size).
        llm: Optional LangChain chat model instance.
        requires_approval: Whether to pause workflow at approval gate for human review.
        approval_status: Optional initial approval status ('pending', 'approved', 'rejected').
        approval_feedback: Optional initial review notes.
        approved_plan: Optional pre-approved ExperimentConfig.
        checkpointer: Optional custom checkpointer (defaults to persistent SqliteCheckpointSaver).

    Returns:
        ExperimentWorkflowState: Final state containing profile, results, and trace.
    """
    graph = create_workflow_graph(checkpointer=checkpointer)
    dataset_ref = DatasetRegistry.register(dataset_source)
    llm_ref = LLMRegistry.register(llm)

    initial_state: ExperimentWorkflowState = {
        "dataset_ref": dataset_ref,
        "target_column": target_column,
        "task_type_override": task_type_override,
        "dataset_metadata": {},
        "user_constraints": user_constraints or {},
        "profile": None,
        "plan": plan,
        "plan_metadata": {},
        "use_llm_planner": use_llm_planner,
        "validation_errors": [],
        "plan_retry_count": 0,
        "max_plan_retries": max_plan_retries,
        "current_proposal": None,
        "llm_ref": llm_ref,
        "llm_instance": None,
        "requires_approval": requires_approval,
        "approval_status": approval_status,
        "approval_feedback": approval_feedback,
        "approved_plan": approved_plan,
        "current_node": None,
        "candidate_runs": [],
        "experiment_result": None,
        "best_candidate_name": None,
        "result_explanation": None,
        "workflow_status": WorkflowStatus.INITIALIZED,
        "error_category": ErrorCategory.NONE,
        "error_message": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "is_recoverable": False,
        "failure_category": None,
        "failure_diagnosis": None,
        "recovery_actions": [],
        "recovery_attempt_count": 0,
        "execution_trace": [],
    }

    t_id = thread_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": t_id}}

    logger.info(f"Invoking workflow graph (thread_id={t_id})")
    final_state = graph.invoke(initial_state, config=config)

    # Check if workflow was paused by human approval interrupt
    snapshot = graph.get_state(config)
    if snapshot.next and "approval_gate" in snapshot.next:
        if requires_approval and final_state.get("approval_status") != "approved":
            final_state["workflow_status"] = WorkflowStatus.WAITING_FOR_APPROVAL
            final_state["approval_status"] = "pending"
            final_state["current_node"] = "approval_gate"
            trace = list(final_state.get("execution_trace", []))
            if "approval_gate: waiting_for_approval" not in trace:
                trace.append("approval_gate: waiting_for_approval")
            final_state["execution_trace"] = trace

    return final_state


def resume_orchestration_workflow(
    thread_id: str,
    approved_plan: Optional[ExperimentConfig] = None,
    approval_status: str = "approved",
    approval_feedback: Optional[str] = None,
    checkpointer: Optional[Any] = None,
) -> ExperimentWorkflowState:
    """Resume a paused workflow thread following human approval or rejection.

    Args:
        thread_id: Checkpoint thread identifier matching the pending job.
        approved_plan: Optional validated ExperimentConfig to execute.
        approval_status: 'approved' or 'rejected'.
        approval_feedback: Optional reviewer notes or rationale.
        checkpointer: Optional checkpointer instance.

    Returns:
        ExperimentWorkflowState: Final state after resumed execution.
    """
    graph = create_workflow_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    current_snapshot = graph.get_state(config)
    trace = list(current_snapshot.values.get("execution_trace", [])) if current_snapshot else []
    trace.append(f"approval_gate: {approval_status}")

    updates: Dict[str, Any] = {
        "approval_status": approval_status,
        "approval_feedback": approval_feedback,
        "current_node": "approval_gate",
        "execution_trace": trace,
    }
    if approval_status == "approved":
        updates["workflow_status"] = WorkflowStatus.RUNNING
        if approved_plan is not None:
            updates["approved_plan"] = approved_plan
            updates["plan"] = approved_plan
    elif approval_status == "rejected":
        updates["workflow_status"] = WorkflowStatus.REJECTED
        updates["error_message"] = approval_feedback or "Experiment plan was rejected by user."

    graph.update_state(config, updates, as_node="approval_gate")
    logger.info(f"Resumed workflow thread {thread_id} with approval_status={approval_status}")
    graph.invoke(None, config=config)
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot else {}

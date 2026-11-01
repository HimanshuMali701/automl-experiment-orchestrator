"""StateGraph assembly, conditional routing, and workflow execution engine."""

from typing import Any, Dict, Optional, Union
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from automl.core.logging import get_logger
from automl.profiler.schemas import TaskType
from automl.trainer.schemas import ExperimentConfig
from automl.workflow.nodes import (
    create_plan_node,
    evaluate_results_node,
    execute_experiments_node,
    finalize_workflow_node,
    handle_failure_node,
    profile_data_node,
    rank_leaderboard_node,
    validate_input_node,
    validate_plan_node,
)
from automl.workflow.registry import DatasetRegistry
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
    """Route to experiment execution if plan is valid, otherwise to failure handling."""
    if state.get("workflow_status") == WorkflowStatus.RUNNING:
        return "execute_experiments"
    return "handle_failure"


def route_after_execution(state: ExperimentWorkflowState) -> str:
    """Route to evaluation if execution produced results, otherwise to failure handling."""
    if state.get("workflow_status") == WorkflowStatus.EVALUATING:
        return "evaluate_results"
    return "handle_failure"


def route_after_evaluation(state: ExperimentWorkflowState) -> str:
    """Route to ranking if models succeeded (full or partial), otherwise to failure handling."""
    status = state.get("workflow_status")
    if status in (WorkflowStatus.SUCCESS, WorkflowStatus.PARTIAL_SUCCESS):
        return "rank_leaderboard"
    return "handle_failure"


def route_after_failure(state: ExperimentWorkflowState) -> str:
    """Route to retry execution if failure is recoverable and retries remain, else finalize."""
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
    workflow.add_node("execute_experiments", execute_experiments_node)
    workflow.add_node("evaluate_results", evaluate_results_node)
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
            "execute_experiments": "execute_experiments",
            "handle_failure": "handle_failure",
        },
    )

    workflow.add_conditional_edges(
        "execute_experiments",
        route_after_execution,
        {
            "evaluate_results": "evaluate_results",
            "handle_failure": "handle_failure",
        },
    )

    workflow.add_conditional_edges(
        "evaluate_results",
        route_after_evaluation,
        {
            "rank_leaderboard": "rank_leaderboard",
            "handle_failure": "handle_failure",
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

    saver = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=saver)


def run_orchestration_workflow(
    dataset_source: Any,
    target_column: Optional[str] = None,
    task_type_override: Optional[TaskType] = None,
    plan: Optional[ExperimentConfig] = None,
    use_llm_planner: bool = False,
    max_retries: int = 2,
    thread_id: Optional[str] = None,
) -> ExperimentWorkflowState:
    """Execute end-to-end AutoML workflow with LangGraph orchestration.

    Args:
        dataset_source: Filepath, bytes, or DataFrame.
        target_column: Target column for supervised prediction.
        task_type_override: Optional task type override.
        plan: Optional custom experiment plan.
        use_llm_planner: Whether to use LLM planner for plan creation.
        max_retries: Maximum bounded retry count for recoverable failures.
        thread_id: Optional checkpointer thread ID.

    Returns:
        ExperimentWorkflowState: Final state containing profile, results, and trace.
    """
    graph = create_workflow_graph()
    dataset_ref = DatasetRegistry.register(dataset_source)

    initial_state: ExperimentWorkflowState = {
        "dataset_ref": dataset_ref,
        "target_column": target_column,
        "task_type_override": task_type_override,
        "dataset_metadata": {},
        "profile": None,
        "plan": plan,
        "plan_metadata": {},
        "use_llm_planner": use_llm_planner,
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
        "execution_trace": [],
    }

    t_id = thread_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": t_id}}

    logger.info(f"Invoking workflow graph (thread_id={t_id})")
    final_state = graph.invoke(initial_state, config=config)
    return final_state

"""StateGraph assembly, conditional routing, and workflow execution engine."""

import time
from typing import Any, Callable, Dict, Optional, Union
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, StateGraph

from automl.core.logging import get_logger
from automl.profiler.schemas import TaskType
from automl.trainer.schemas import ExperimentConfig
from automl.workflow.checkpointer import SqliteCheckpointSaver, get_default_checkpointer
from automl.workflow.events import (
    EventType,
    NODE_TO_STAGE,
    current_experiment_id,
    get_event_manager,
)
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


# --- Conditional Routing Functions & Instrumentation ---

def _record_routing_decision(state: ExperimentWorkflowState, from_node: str, to_node: str, reason: str) -> None:
    """Record an evaluated conditional routing decision in the active experiment event store."""
    t_id = state.get("thread_id") or current_experiment_id.get()
    if t_id:
        get_event_manager().record_event(
            experiment_id=t_id,
            event_type=EventType.ROUTING_DECISION,
            node=from_node,
            summary=f"Routing from '{from_node}' to '{to_node}': {reason}",
            details={"from_node": from_node, "to_node": to_node, "reason": reason},
        )


def route_after_input_validation(state: ExperimentWorkflowState) -> str:
    """Route to data profiling if input is valid, otherwise to failure handling."""
    if state.get("error_category") == ErrorCategory.NONE:
        target = "profile_data"
        reason = "Input dataset validated successfully without errors."
    else:
        target = "handle_failure"
        reason = f"Input validation failed: {state.get('error_message') or 'Invalid dataset'}"
    _record_routing_decision(state, "validate_input", target, reason)
    return target


def route_after_profiling(state: ExperimentWorkflowState) -> str:
    """Route to plan creation if profiling succeeded, otherwise to failure handling."""
    profile = state.get("profile")
    if profile is not None:
        target = "create_plan"
        tp = getattr(profile, "target_profile", None)
        task_str = (
            tp.detected_task_type.value
            if (tp and hasattr(tp.detected_task_type, "value"))
            else "detected"
        )
        reason = f"Dataset profiled successfully (task_type={task_str})."
    else:
        target = "handle_failure"
        reason = "Profiling step failed to generate valid dataset profile."
    _record_routing_decision(state, "profile_data", target, reason)
    return target


def route_after_plan_validation(state: ExperimentWorkflowState) -> str:
    """Route after plan validation: approval_gate if valid, repair if retries remain, fallback if exhausted."""
    if state.get("use_llm_planner", False):
        val_errors = state.get("validation_errors", [])
        if not val_errors and state.get("plan") is not None:
            target = "approval_gate"
            reason = "LLM proposed plan passed all schema and parameter validations."
        else:
            retry_count = state.get("plan_retry_count", 0)
            max_retries = state.get("max_plan_retries", 2)
            if retry_count < max_retries:
                target = "repair_plan"
                reason = f"LLM plan validation failed with {len(val_errors)} error(s); triggering reflection repair (attempt {retry_count + 1} of {max_retries})."
            else:
                target = "fallback_plan"
                reason = f"LLM plan reflection retries exhausted ({retry_count}/{max_retries}); falling back to deterministic template."
    else:
        if state.get("workflow_status") in (WorkflowStatus.RUNNING, WorkflowStatus.PLANNING) and state.get("plan") is not None:
            target = "approval_gate"
            reason = "Deterministic experiment plan validated successfully."
        else:
            target = "handle_failure"
            reason = "Deterministic plan validation failed."

    _record_routing_decision(state, "validate_plan", target, reason)
    return target


def route_after_approval_gate(state: ExperimentWorkflowState) -> str:
    """Route after approval gate: finalize if rejected, else execute_experiments."""
    if (
        state.get("approval_status") == "rejected"
        or state.get("workflow_status") == WorkflowStatus.REJECTED
    ):
        target = "finalize_workflow"
        reason = f"Plan rejected by user: {state.get('approval_feedback') or 'No feedback provided'}."
    else:
        target = "execute_experiments"
        reason = "Plan approved (or approval disabled); proceeding to model execution."

    _record_routing_decision(state, "approval_gate", target, reason)
    return target


def route_after_execution(state: ExperimentWorkflowState) -> str:
    """Route to evaluation if execution produced results, otherwise to failure diagnosis."""
    if state.get("workflow_status") == WorkflowStatus.EVALUATING:
        target = "evaluate_results"
        reason = "Candidate model training completed; proceeding to evaluation."
    else:
        target = "diagnose_failure"
        reason = f"Execution step failed: {state.get('error_message') or 'Training failure'}."
    _record_routing_decision(state, "execute_experiments", target, reason)
    return target


def route_after_evaluation(state: ExperimentWorkflowState) -> str:
    """Route to ranking if models succeeded (full or partial), otherwise to failure diagnosis."""
    status = state.get("workflow_status")
    if status in (WorkflowStatus.SUCCESS, WorkflowStatus.PARTIAL_SUCCESS):
        target = "rank_leaderboard"
        reason = f"Evaluation finished with status '{status.value if hasattr(status, 'value') else status}'; ranking models."
    else:
        target = "diagnose_failure"
        reason = "Evaluation failed to find any successful candidate models."
    _record_routing_decision(state, "evaluate_results", target, reason)
    return target


def route_after_diagnosis(state: ExperimentWorkflowState) -> str:
    """Route after failure diagnosis based on recoverability, retry budget, and candidate survival."""
    if state.get("is_recoverable") and state.get("retry_count", 0) < state.get("max_retries", 2):
        target = "handle_failure"
        cat = state.get("failure_category")
        reason = f"Failure diagnosed as '{cat.value if hasattr(cat, 'value') else cat}'; recoverable with retry budget ({state.get('retry_count', 0)}/{state.get('max_retries', 2)})."
        _record_routing_decision(state, "diagnose_failure", target, reason)
        return target

    # Candidate-specific failure where other candidates succeeded
    runs = state.get("candidate_runs") or []
    if any(r.status.value == "success" for r in runs):
        target = "rank_leaderboard"
        reason = "One or more candidate models failed, but successful candidates exist; ranking leaderboard."
        _record_routing_decision(state, "diagnose_failure", target, reason)
        return target

    target = "finalize_workflow"
    reason = f"Unrecoverable failure or retry budget exhausted: {state.get('error_message')}."
    _record_routing_decision(state, "diagnose_failure", target, reason)
    return target


def route_after_failure(state: ExperimentWorkflowState) -> str:
    """Route after failure handling: rerun experiments if retry prepared, else finalize."""
    if (
        state.get("is_recoverable")
        and state.get("retry_count", 0) <= state.get("max_retries", 2)
        and state.get("workflow_status") == WorkflowStatus.RUNNING
    ):
        target = "execute_experiments"
        reason = f"Adaptive recovery actions applied; re-running training (attempt {state.get('retry_count')})."
    else:
        target = "finalize_workflow"
        reason = "Recovery could not prepare viable execution; terminating workflow."
    _record_routing_decision(state, "handle_failure", target, reason)
    return target


# --- Observable Node Instrumentation ---

def _summarize_node_output(node_name: str, output: Any) -> Dict[str, Any]:
    """Extract a privacy-safe, structured technical summary of a node's output."""
    if not isinstance(output, dict):
        return {}
    summary: Dict[str, Any] = {}
    if "workflow_status" in output:
        status = output["workflow_status"]
        summary["workflow_status"] = status.value if hasattr(status, "value") else str(status)
    if "error_message" in output and output["error_message"]:
        summary["error"] = str(output["error_message"])

    if node_name == "validate_input" and "dataset_metadata" in output:
        meta = output["dataset_metadata"]
        summary["row_count"] = meta.get("row_count")
        summary["col_count"] = meta.get("col_count")
    elif node_name == "profile_data" and "profile" in output and output["profile"]:
        p = output["profile"]
        tp = getattr(p, "target_profile", None)
        if tp:
            summary["task_type"] = (
                tp.detected_task_type.value
                if hasattr(tp.detected_task_type, "value")
                else str(tp.detected_task_type)
            )
        summary["row_count"] = getattr(p, "row_count", None)
        summary["column_count"] = getattr(p, "column_count", None)
    elif node_name in ("create_plan", "fallback_plan", "repair_plan") and "plan" in output and output["plan"]:
        pl = output["plan"]
        summary["candidates"] = [c.name for c in pl.candidates]
        summary["primary_metric"] = pl.primary_metric.value if hasattr(pl.primary_metric, "value") else str(pl.primary_metric)
    elif node_name == "validate_plan":
        val_errors = output.get("validation_errors", [])
        summary["validation_errors"] = val_errors
        summary["is_valid"] = len(val_errors) == 0
    elif node_name == "execute_experiments":
        runs = output.get("candidate_runs", [])
        summary["completed_candidates"] = len(runs)
        summary["successful_candidates"] = sum(
            1 for r in runs if hasattr(r, "status") and getattr(r.status, "value", str(r.status)) == "success"
        )
    elif node_name == "rank_leaderboard":
        summary["best_candidate"] = output.get("best_candidate_name")
    elif node_name == "diagnose_failure":
        cat = output.get("failure_category")
        summary["failure_category"] = cat.value if hasattr(cat, "value") else str(cat)
        summary["is_recoverable"] = output.get("is_recoverable", False)
    elif node_name == "handle_failure":
        summary["recovery_actions"] = [str(a) for a in output.get("recovery_actions", [])]
        summary["retry_count"] = output.get("retry_count")

    return summary


def make_observable_node(node_name: str, node_func: Any) -> Any:
    """Wrap a graph node function with real-time event recording, durations, and state transitions."""
    def wrapped_node(state: ExperimentWorkflowState, *args, **kwargs) -> Dict[str, Any]:
        t_id = state.get("thread_id") or current_experiment_id.get() or "default"
        mgr = get_event_manager()
        stage_name = NODE_TO_STAGE.get(node_name, node_name)
        start_t = time.perf_counter()

        mgr.record_event(
            experiment_id=t_id,
            event_type=EventType.NODE_START,
            node=node_name,
            stage=stage_name,
            status="running",
            summary=f"Started executing node '{node_name}' ({stage_name})",
        )

        try:
            output = node_func(state, *args, **kwargs)
        except NodeInterrupt:
            # Captures human approval gate interrupt cleanly
            mgr.record_event(
                experiment_id=t_id,
                event_type=EventType.APPROVAL_CHECKPOINT,
                node=node_name,
                stage=stage_name,
                status="waiting_for_approval",
                summary="Workflow paused at approval_gate waiting for human plan review",
                details={
                    "pending_plan": state.get("plan").model_dump() if state.get("plan") else None,
                    "validation_errors": state.get("validation_errors", []),
                },
            )
            raise
        except Exception as exc:
            duration = time.perf_counter() - start_t
            mgr.record_event(
                experiment_id=t_id,
                event_type=EventType.NODE_FINISH,
                node=node_name,
                stage=stage_name,
                status="failed",
                duration_seconds=duration,
                summary=f"Node '{node_name}' failed with error: {str(exc)}",
                details={"error": str(exc)},
            )
            raise

        duration = time.perf_counter() - start_t
        out_status = "completed"
        if isinstance(output, dict):
            wf_status = output.get("workflow_status")
            if wf_status in (WorkflowStatus.FAILED, WorkflowStatus.REJECTED):
                out_status = "failed" if wf_status == WorkflowStatus.FAILED else "rejected"
            elif wf_status == WorkflowStatus.WAITING_FOR_APPROVAL:
                out_status = "waiting_for_approval"

        # Emit specialized sub-events based on node outputs
        if node_name in ("create_plan", "fallback_plan") and output.get("plan"):
            candidates = [c.name for c in output["plan"].candidates]
            mgr.record_event(
                experiment_id=t_id,
                event_type=EventType.LLM_PROPOSAL,
                node=node_name,
                stage=stage_name,
                status="completed",
                summary=f"{'LLM proposed' if state.get('use_llm_planner') else 'Generated default'} {len(candidates)} candidate model(s): {', '.join(candidates)}",
                details={"candidates": candidates, "use_llm": state.get("use_llm_planner", False)},
            )
        elif node_name == "validate_plan":
            errs = output.get("validation_errors", [])
            mgr.record_event(
                experiment_id=t_id,
                event_type=EventType.PLAN_VALIDATION,
                node=node_name,
                stage=stage_name,
                status="completed",
                summary=f"Plan validation {'passed' if not errs else f'failed with {len(errs)} error(s)'}",
                details={"is_valid": len(errs) == 0, "errors": errs},
            )
        elif node_name == "repair_plan":
            retry = output.get("plan_retry_count", 0)
            mgr.record_event(
                experiment_id=t_id,
                event_type=EventType.LLM_REFLECTION,
                node=node_name,
                stage=stage_name,
                status="completed",
                summary=f"LLM plan reflection & repair completed (attempt {retry})",
                details={"retry_count": retry, "candidates": [c.name for c in output['plan'].candidates] if output.get('plan') else []},
            )
        elif node_name == "diagnose_failure":
            cat = output.get("failure_category")
            diag = output.get("failure_diagnosis")
            mgr.record_event(
                experiment_id=t_id,
                event_type=EventType.FAILURE_DIAGNOSIS,
                node=node_name,
                stage=stage_name,
                status="completed",
                summary=f"Failure diagnosed as '{getattr(cat, 'value', str(cat))}': {getattr(diag, 'root_cause', 'Unknown') if diag else 'Unknown'}",
                details={"category": str(cat), "is_recoverable": output.get("is_recoverable", False)},
            )
        elif node_name == "handle_failure":
            acts = output.get("recovery_actions", [])
            mgr.record_event(
                experiment_id=t_id,
                event_type=EventType.RECOVERY_ACTION,
                node=node_name,
                stage=stage_name,
                status="completed",
                summary=f"Applied {len(acts)} adaptive recovery actions (retry {output.get('retry_count', 0)})",
                details={"actions": [str(a) for a in acts], "retry_count": output.get("retry_count")},
            )

        mgr.record_event(
            experiment_id=t_id,
            event_type=EventType.NODE_FINISH,
            node=node_name,
            stage=stage_name,
            status=out_status,
            duration_seconds=duration,
            summary=f"Completed node '{node_name}' in {duration:.2f}s ({out_status})",
            details=_summarize_node_output(node_name, output),
        )
        return output

    return wrapped_node


# --- Graph Construction ---

def create_workflow_graph(checkpointer: Optional[Any] = None) -> Any:
    """Build and compile the LangGraph StateGraph for AutoML experiment orchestration.

    Returns:
        CompiledStateGraph: The compiled executable state graph with real-time observability.
    """
    workflow = StateGraph(ExperimentWorkflowState)

    # 1. Register instrumented nodes
    workflow.add_node("validate_input", make_observable_node("validate_input", validate_input_node))
    workflow.add_node("profile_data", make_observable_node("profile_data", profile_data_node))
    workflow.add_node("create_plan", make_observable_node("create_plan", create_plan_node))
    workflow.add_node("validate_plan", make_observable_node("validate_plan", validate_plan_node))
    workflow.add_node("repair_plan", make_observable_node("repair_plan", repair_plan_node))
    workflow.add_node("fallback_plan", make_observable_node("fallback_plan", fallback_plan_node))
    workflow.add_node("approval_gate", make_observable_node("approval_gate", approval_gate_node))
    workflow.add_node("execute_experiments", make_observable_node("execute_experiments", execute_experiments_node))
    workflow.add_node("evaluate_results", make_observable_node("evaluate_results", evaluate_results_node))
    workflow.add_node("diagnose_failure", make_observable_node("diagnose_failure", diagnose_failure_node))
    workflow.add_node("handle_failure", make_observable_node("handle_failure", handle_failure_node))
    workflow.add_node("rank_leaderboard", make_observable_node("rank_leaderboard", rank_leaderboard_node))
    workflow.add_node("finalize_workflow", make_observable_node("finalize_workflow", finalize_workflow_node))

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
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> ExperimentWorkflowState:
    """Execute end-to-end AutoML workflow with LangGraph streaming orchestration."""
    graph = create_workflow_graph(checkpointer=checkpointer)
    dataset_ref = DatasetRegistry.register(dataset_source)
    llm_ref = LLMRegistry.register(llm)

    t_id = thread_id or str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": t_id}}

    initial_state: ExperimentWorkflowState = {
        "thread_id": t_id,
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

    token = current_experiment_id.set(t_id)
    logger.info(f"Invoking streaming workflow graph (thread_id={t_id})")

    get_event_manager().record_event(
        experiment_id=t_id,
        event_type=EventType.WORKFLOW_START,
        summary=f"Started AutoML experiment orchestration workflow (thread_id={t_id})",
        details={
            "requires_approval": requires_approval,
            "use_llm_planner": use_llm_planner,
            "target_column": target_column,
        },
    )

    try:
        # LangGraph streaming execution yielding per-node updates
        for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                if node_name != "__interrupt__" and on_step is not None:
                    try:
                        on_step(node_name, node_update)
                    except Exception as cb_err:
                        logger.warning(f"Error in on_step callback for {node_name}: {cb_err}")

        snapshot = graph.get_state(config)
        final_state = dict(snapshot.values) if snapshot and snapshot.values else initial_state

        # Check if workflow was paused by human approval interrupt
        if snapshot and snapshot.next and "approval_gate" in snapshot.next:
            if requires_approval and final_state.get("approval_status") != "approved":
                final_state["workflow_status"] = WorkflowStatus.WAITING_FOR_APPROVAL
                final_state["approval_status"] = "pending"
                final_state["current_node"] = "approval_gate"
                trace = list(final_state.get("execution_trace", []))
                if "approval_gate: waiting_for_approval" not in trace:
                    trace.append("approval_gate: waiting_for_approval")
                final_state["execution_trace"] = trace
        else:
            wf_status = final_state.get("workflow_status", WorkflowStatus.SUCCESS)
            get_event_manager().record_event(
                experiment_id=t_id,
                event_type=EventType.WORKFLOW_FINISH,
                status=getattr(wf_status, "value", str(wf_status)),
                summary=f"Workflow finished with status: {getattr(wf_status, 'value', str(wf_status))}",
                details={"best_candidate": final_state.get("best_candidate_name")},
            )

        return final_state
    finally:
        current_experiment_id.reset(token)


def resume_orchestration_workflow(
    thread_id: str,
    approved_plan: Optional[ExperimentConfig] = None,
    approval_status: str = "approved",
    approval_feedback: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> ExperimentWorkflowState:
    """Resume a paused workflow thread following human approval or rejection."""
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

    token = current_experiment_id.set(thread_id)
    get_event_manager().record_event(
        experiment_id=thread_id,
        event_type=EventType.APPROVAL_RESUMED,
        node="approval_gate",
        stage=NODE_TO_STAGE.get("approval_gate"),
        status=approval_status,
        summary=f"Workflow resumed at approval_gate with status '{approval_status}'",
        details={"approval_status": approval_status, "feedback": approval_feedback},
    )

    try:
        graph.update_state(config, updates, as_node="approval_gate")
        logger.info(f"Resumed workflow thread {thread_id} with approval_status={approval_status}")

        for chunk in graph.stream(None, config=config, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                if node_name != "__interrupt__" and on_step is not None:
                    try:
                        on_step(node_name, node_update)
                    except Exception as cb_err:
                        logger.warning(f"Error in on_step callback for {node_name}: {cb_err}")

        snapshot = graph.get_state(config)
        final_state = dict(snapshot.values) if snapshot and snapshot.values else {}
        wf_status = final_state.get("workflow_status", WorkflowStatus.SUCCESS)
        get_event_manager().record_event(
            experiment_id=thread_id,
            event_type=EventType.WORKFLOW_FINISH,
            status=getattr(wf_status, "value", str(wf_status)),
            summary=f"Workflow finished after resumption with status: {getattr(wf_status, 'value', str(wf_status))}",
            details={"best_candidate": final_state.get("best_candidate_name")},
        )
        return final_state
    finally:
        current_experiment_id.reset(token)

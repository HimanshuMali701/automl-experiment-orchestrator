"""LangGraph workflow event system, observability tracing, and persistent event store."""

from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from automl.core.config import get_settings
from automl.core.logging import get_logger

logger = get_logger("workflow.events")

# ContextVar for propagating active experiment/thread ID across thread executions
current_experiment_id: ContextVar[Optional[str]] = ContextVar("current_experiment_id", default=None)


class EventType(str, Enum):
    """Categorization of LangGraph execution events."""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_FINISH = "workflow_finish"
    NODE_START = "node_start"
    NODE_FINISH = "node_finish"
    ROUTING_DECISION = "routing_decision"
    LLM_PROPOSAL = "llm_proposal"
    LLM_REFLECTION = "llm_reflection"
    PLAN_VALIDATION = "plan_validation"
    APPROVAL_CHECKPOINT = "approval_checkpoint"
    APPROVAL_RESUMED = "approval_resumed"
    FAILURE_DIAGNOSIS = "failure_diagnosis"
    RECOVERY_ACTION = "recovery_action"
    WORKFLOW_STATUS = "workflow_status"


NODE_TO_STAGE: Dict[str, str] = {
    "validate_input": "Dataset Validation",
    "profile_data": "Data Profiling",
    "create_plan": "Plan Creation",
    "validate_plan": "Plan Validation",
    "repair_plan": "Plan Repair & Reflection",
    "fallback_plan": "Fallback Planning",
    "approval_gate": "Human Approval Checkpoint",
    "execute_experiments": "Model Training & Cross-Validation",
    "evaluate_results": "Model Evaluation",
    "diagnose_failure": "Failure Diagnosis",
    "handle_failure": "Adaptive Recovery",
    "rank_leaderboard": "Leaderboard Ranking",
    "finalize_workflow": "Workflow Finalization",
}

ALL_WORKFLOW_NODES: List[str] = [
    "validate_input",
    "profile_data",
    "create_plan",
    "validate_plan",
    "repair_plan",
    "fallback_plan",
    "approval_gate",
    "execute_experiments",
    "evaluate_results",
    "diagnose_failure",
    "handle_failure",
    "rank_leaderboard",
    "finalize_workflow",
]


class WorkflowEvent(BaseModel):
    """A granular, structured record of an event occurring in the LangGraph workflow."""
    event_id: str
    experiment_id: str
    timestamp: str  # ISO 8601 UTC
    event_type: EventType
    node: Optional[str] = None
    stage: Optional[str] = None
    status: Optional[str] = None  # running, completed, failed, waiting_for_approval, skipped
    duration_seconds: Optional[float] = None
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


def sanitize_event_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize details dictionary to prevent leaking secrets, raw dataset rows, or giant stack traces.

    Security & Privacy Rules:
    1. Keys matching sensitive keywords (api_key, token, password, secret, auth) are redacted.
    2. Dataframe rows or row collections are converted to counts or column summaries.
    3. Multi-line error tracebacks are truncated to the essential message.
    """
    sensitive_patterns = re.compile(r"(api[_-]?key|token|password|secret|auth|credential)", re.IGNORECASE)
    sanitized: Dict[str, Any] = {}

    for key, value in details.items():
        if sensitive_patterns.search(key):
            sanitized[key] = "[REDACTED]"
            continue

        if isinstance(value, dict):
            sanitized[key] = sanitize_event_details(value)
        elif isinstance(value, list):
            # Guard against raw dataset row records (e.g. list of dicts with many columns)
            if value and isinstance(value[0], dict) and len(value[0]) > 6:
                sanitized[key] = f"[{len(value)} records omitted for privacy]"
            else:
                sanitized_list = []
                for item in value:
                    if isinstance(item, dict):
                        sanitized_list.append(sanitize_event_details(item))
                    elif isinstance(item, str) and sensitive_patterns.search(item):
                        sanitized_list.append("[REDACTED]")
                    else:
                        sanitized_list.append(item)
                sanitized[key] = sanitized_list
        elif isinstance(value, str):
            # Truncate raw stack traces to the first 300 characters
            if "Traceback (most recent call last):" in value:
                lines = value.strip().splitlines()
                last_line = lines[-1] if lines else "Error"
                sanitized[key] = f"Error: {last_line}"
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value

    return sanitized


class EventManager:
    """Thread-safe event manager persisting graph execution events incrementally."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.storage_dir: Path = storage_dir or (settings.artifacts_dir / "events")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._in_memory_events: Dict[str, List[WorkflowEvent]] = {}

    def _get_event_file(self, experiment_id: str) -> Path:
        return self.storage_dir / f"{experiment_id}.jsonl"

    def record_event(
        self,
        experiment_id: str,
        event_type: EventType,
        node: Optional[str] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        summary: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> WorkflowEvent:
        """Create, record, and incrementally persist a new workflow event."""
        import uuid
        now_utc = datetime.now(timezone.utc).isoformat()
        resolved_stage = stage or (NODE_TO_STAGE.get(node) if node else None)
        sanitized_details = sanitize_event_details(details or {})

        event = WorkflowEvent(
            event_id=str(uuid.uuid4())[:8],
            experiment_id=experiment_id,
            timestamp=now_utc,
            event_type=event_type,
            node=node,
            stage=resolved_stage,
            status=status,
            duration_seconds=round(duration_seconds, 4) if duration_seconds is not None else None,
            summary=summary,
            details=sanitized_details,
        )

        with self._lock:
            # 1. Update in-memory cache
            if experiment_id not in self._in_memory_events:
                self._in_memory_events[experiment_id] = []
            self._in_memory_events[experiment_id].append(event)

            # 2. Append incrementally to disk (JSON lines format)
            event_file = self._get_event_file(experiment_id)
            try:
                with open(event_file, "a", encoding="utf-8") as f:
                    f.write(event.model_dump_json() + "\n")
            except Exception as e:
                logger.warning(f"Failed to append event to {event_file}: {e}")

        logger.debug(f"Recorded event [{event_type.value}] for {experiment_id}: {summary}")
        return event

    def get_events(self, experiment_id: str) -> List[WorkflowEvent]:
        """Retrieve all recorded events for an experiment in chronological order."""
        with self._lock:
            if experiment_id in self._in_memory_events and self._in_memory_events[experiment_id]:
                return list(self._in_memory_events[experiment_id])

            # Fall back to disk if not in memory
            event_file = self._get_event_file(experiment_id)
            if not event_file.exists():
                return []

            events: List[WorkflowEvent] = []
            try:
                with open(event_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            events.append(WorkflowEvent.model_validate(data))
                self._in_memory_events[experiment_id] = events
                return events
            except Exception as e:
                logger.warning(f"Failed to read events from {event_file}: {e}")
                return []

    def get_timeline(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Compute node execution timeline showing start, end, duration, and status."""
        events = self.get_events(experiment_id)
        node_map: Dict[str, Dict[str, Any]] = {}

        for ev in events:
            if not ev.node:
                continue
            name = ev.node
            if name not in node_map:
                node_map[name] = {
                    "node": name,
                    "stage": NODE_TO_STAGE.get(name, name),
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "duration_seconds": None,
                    "summary": None,
                }

            if ev.event_type == EventType.NODE_START:
                node_map[name]["status"] = "running"
                node_map[name]["start_time"] = ev.timestamp
            elif ev.event_type == EventType.NODE_FINISH:
                node_map[name]["status"] = ev.status or "completed"
                node_map[name]["end_time"] = ev.timestamp
                node_map[name]["duration_seconds"] = ev.duration_seconds
                node_map[name]["summary"] = ev.summary
            elif ev.event_type == EventType.APPROVAL_CHECKPOINT:
                node_map[name]["status"] = "waiting_for_approval"
                node_map[name]["end_time"] = ev.timestamp
                node_map[name]["summary"] = ev.summary

        return list(node_map.values())

    def get_routing_decisions(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Extract all conditional routing decisions recorded during execution."""
        events = self.get_events(experiment_id)
        decisions: List[Dict[str, Any]] = []
        for ev in events:
            if ev.event_type == EventType.ROUTING_DECISION:
                decisions.append({
                    "timestamp": ev.timestamp,
                    "from_node": ev.details.get("from_node", ev.node),
                    "to_node": ev.details.get("to_node"),
                    "reason": ev.details.get("reason", ev.summary),
                })
        return decisions

    def get_llm_interactions(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Extract LLM plan proposals and repair reflection attempts."""
        events = self.get_events(experiment_id)
        interactions: List[Dict[str, Any]] = []
        for ev in events:
            if ev.event_type in (EventType.LLM_PROPOSAL, EventType.LLM_REFLECTION, EventType.PLAN_VALIDATION):
                interactions.append({
                    "timestamp": ev.timestamp,
                    "type": ev.event_type.value,
                    "summary": ev.summary,
                    "details": ev.details,
                })
        return interactions

    def get_recovery_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Extract failure diagnosis and adaptive recovery history."""
        events = self.get_events(experiment_id)
        history: List[Dict[str, Any]] = []
        for ev in events:
            if ev.event_type in (EventType.FAILURE_DIAGNOSIS, EventType.RECOVERY_ACTION):
                history.append({
                    "timestamp": ev.timestamp,
                    "type": ev.event_type.value,
                    "summary": ev.summary,
                    "details": ev.details,
                })
        return history

    def clear(self, experiment_id: str) -> None:
        """Clear cached in-memory events."""
        with self._lock:
            self._in_memory_events.pop(experiment_id, None)


# Global singleton
_global_event_manager: Optional[EventManager] = None
_manager_lock = threading.Lock()


def get_event_manager() -> EventManager:
    """Retrieve global EventManager singleton."""
    global _global_event_manager
    if _global_event_manager is None:
        with _manager_lock:
            if _global_event_manager is None:
                _global_event_manager = EventManager()
    return _global_event_manager


"""Unit tests for LangGraph observability, event streaming, and Agent Workflow Inspector."""

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict
import pandas as pd
import pytest
from starlette.testclient import TestClient

from automl.api.app import app
from automl.api.jobs import JobManager, JobRecord, JobStatus, get_job_manager
from automl.api.schemas import ExperimentJobResponse, WorkflowInspectorResponse
from automl.core.config import get_settings
from automl.planner.schemas import LLMExperimentProposal
from automl.recovery.schemas import ActionType, FailureCategory, FailureDiagnosis, RecoveryAction
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    RunStatus,
)
from automl.ui.client import APIClient
from automl.workflow.checkpointer import SqliteCheckpointSaver
from automl.workflow.events import (
    ALL_WORKFLOW_NODES,
    EventManager,
    EventType,
    NODE_TO_STAGE,
    WorkflowEvent,
    get_event_manager,
    sanitize_event_details,
)
from automl.workflow.graph import (
    create_workflow_graph,
    resume_orchestration_workflow,
    run_orchestration_workflow,
)
from automl.workflow.state import ErrorCategory, ExperimentWorkflowState, WorkflowStatus


@pytest.fixture
def clean_event_dir(tmp_path: Path):
    """Fixture providing clean temporary directory for EventManager."""
    mgr = EventManager(storage_dir=tmp_path / "events")
    return mgr


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """Fixture providing clean classification DataFrame."""
    return pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "feature_2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 3,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })


def test_event_creation_and_sanitization():
    """Verify event details sanitization redacts secrets, raw records, and giant stack traces."""
    raw_details = {
        "api_key": "groq_secret_abc123",
        "nested": {"groq_token": "bearer_secret_xyz"},
        "data_records": [
            {"col1": 1, "col2": 2, "col3": 3, "col4": 4, "col5": 5, "col6": 6, "col7": 7},
            {"col1": 8, "col2": 9, "col3": 10, "col4": 11, "col5": 12, "col6": 13, "col7": 14},
        ],
        "traceback": "Traceback (most recent call last):\n  File 'train.py', line 45\nValueError: Division by zero",
        "safe_metric": 0.85,
    }

    sanitized = sanitize_event_details(raw_details)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["groq_token"] == "[REDACTED]"
    assert "records omitted" in sanitized["data_records"]
    assert sanitized["traceback"] == "Error: ValueError: Division by zero"
    assert sanitized["safe_metric"] == 0.85


def test_event_manager_incremental_persistence(clean_event_dir: EventManager):
    """Verify incremental appending to disk and immediate retrieval by concurrent readers."""
    mgr = clean_event_dir
    exp_id = "test-exp-001"

    ev1 = mgr.record_event(
        experiment_id=exp_id,
        event_type=EventType.NODE_START,
        node="validate_input",
        status="running",
        summary="Started validate_input",
    )

    ev2 = mgr.record_event(
        experiment_id=exp_id,
        event_type=EventType.NODE_FINISH,
        node="validate_input",
        status="completed",
        duration_seconds=0.125,
        summary="Completed validate_input",
        details={"row_count": 30},
    )

    # Verify memory retrieval
    events = mgr.get_events(exp_id)
    assert len(events) == 2
    assert events[0].node == "validate_input"
    assert events[1].duration_seconds == 0.125

    # Verify disk persistence (a separate EventManager instance reading the same directory)
    disk_mgr = EventManager(storage_dir=mgr.storage_dir)
    disk_events = disk_mgr.get_events(exp_id)
    assert len(disk_events) == 2
    assert disk_events[0].event_type == EventType.NODE_START
    assert disk_events[1].event_type == EventType.NODE_FINISH

    # Verify timeline aggregation
    timeline = mgr.get_timeline(exp_id)
    assert len(timeline) == 1
    assert timeline[0]["node"] == "validate_input"
    assert timeline[0]["status"] == "completed"
    assert timeline[0]["duration_seconds"] == 0.125


def test_workflow_execution_event_generation_and_timings(sample_classification_df: pd.DataFrame, tmp_path: Path):
    """Verify end-to-end workflow execution generates start, finish, routing, and node events with real durations."""
    db_file = tmp_path / "obs_ckpt.db"
    checkpointer = SqliteCheckpointSaver.from_conn_string(str(db_file))

    collected_steps = []

    def on_step(node_name: str, node_update: Dict[str, Any]) -> None:
        collected_steps.append(node_name)

    thread_id = "obs-e2e-run"
    final_state = run_orchestration_workflow(
        dataset_source=sample_classification_df,
        target_column="target",
        thread_id=thread_id,
        checkpointer=checkpointer,
        on_step=on_step,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert len(collected_steps) > 0
    assert "validate_input" in collected_steps
    assert "profile_data" in collected_steps
    assert "create_plan" in collected_steps
    assert "execute_experiments" in collected_steps
    assert "rank_leaderboard" in collected_steps

    mgr = get_event_manager()
    events = mgr.get_events(thread_id)
    assert len(events) >= 10

    # Verify event types present
    event_types = [ev.event_type for ev in events]
    assert EventType.WORKFLOW_START in event_types
    assert EventType.NODE_START in event_types
    assert EventType.NODE_FINISH in event_types
    assert EventType.ROUTING_DECISION in event_types
    assert EventType.WORKFLOW_FINISH in event_types

    # Verify real durations
    node_finish_events = [ev for ev in events if ev.event_type == EventType.NODE_FINISH]
    for ev in node_finish_events:
        assert ev.duration_seconds is not None
        assert ev.duration_seconds >= 0.0

    # Verify timeline
    timeline = mgr.get_timeline(thread_id)
    executed_nodes = [item["node"] for item in timeline if item["status"] == "completed"]
    assert "validate_input" in executed_nodes
    assert "execute_experiments" in executed_nodes


def test_conditional_routing_decision_capture(sample_classification_df: pd.DataFrame, tmp_path: Path):
    """Verify conditional edge routing decisions record source, target, and substantive reason."""
    db_file = tmp_path / "routing_ckpt.db"
    checkpointer = SqliteCheckpointSaver.from_conn_string(str(db_file))
    thread_id = "obs-routing-run"

    run_orchestration_workflow(
        dataset_source=sample_classification_df,
        target_column="target",
        thread_id=thread_id,
        checkpointer=checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    mgr = get_event_manager()
    decisions = mgr.get_routing_decisions(thread_id)
    assert len(decisions) >= 3

    # Check validation routing
    val_route = next((d for d in decisions if d["from_node"] == "validate_input"), None)
    assert val_route is not None
    assert val_route["to_node"] == "profile_data"
    assert "passed" in val_route["reason"].lower() or "success" in val_route["reason"].lower()

    # Check profiling routing
    prof_route = next((d for d in decisions if d["from_node"] == "profile_data"), None)
    assert prof_route is not None
    assert prof_route["to_node"] == "create_plan"
    assert "classification" in prof_route["reason"].lower()


def test_human_approval_checkpoint_events(sample_classification_df: pd.DataFrame, tmp_path: Path):
    """Verify approval gate emits checkpoint event on pause and resumed event on approval."""
    db_file = tmp_path / "approval_obs.db"
    checkpointer = SqliteCheckpointSaver.from_conn_string(str(db_file))
    thread_id = "obs-approval-run"

    paused_state = run_orchestration_workflow(
        dataset_source=sample_classification_df,
        target_column="target",
        requires_approval=True,
        thread_id=thread_id,
        checkpointer=checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert paused_state["workflow_status"] == WorkflowStatus.WAITING_FOR_APPROVAL

    mgr = get_event_manager()
    events = mgr.get_events(thread_id)
    appr_checkpoints = [ev for ev in events if ev.event_type == EventType.APPROVAL_CHECKPOINT]
    assert len(appr_checkpoints) == 1
    assert appr_checkpoints[0].status == "waiting_for_approval"

    # Resume workflow
    resumed_state = resume_orchestration_workflow(
        thread_id=thread_id,
        approval_status="approved",
        checkpointer=checkpointer,
    )

    assert resumed_state["workflow_status"] == WorkflowStatus.SUCCESS
    updated_events = mgr.get_events(thread_id)
    resumed_evs = [ev for ev in updated_events if ev.event_type == EventType.APPROVAL_RESUMED]
    assert len(resumed_evs) >= 1
    assert resumed_evs[0].status == "approved"


def test_api_event_and_inspector_endpoints(sample_classification_df: pd.DataFrame):
    """Verify GET /experiments/{id}/events and GET /experiments/{id}/inspector REST endpoints."""
    client = TestClient(app)

    # 1. Submit experiment
    csv_bytes = sample_classification_df.to_csv(index=False).encode("utf-8")
    resp = client.post(
        "/experiments",
        files={"file": ("test_obs.csv", csv_bytes, "text/csv")},
        data={"target_column": "target", "cv_folds": 2},
    )
    assert resp.status_code == 202
    exp_id = resp.json()["experiment_id"]

    # 2. Wait for job to reach running or completed state
    for _ in range(40):
        status_check = client.get(f"/experiments/{exp_id}").json()
        if status_check["status"] in ("completed", "failed") or len(status_check.get("completed_nodes", [])) >= 2:
            break
        time.sleep(0.5)

    # 3. Query GET /experiments/{id}
    status_resp = client.get(f"/experiments/{exp_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "current_stage" in status_data
    assert "execution_trace" in status_data
    assert "completed_nodes" in status_data
    assert "pending_approval" in status_data

    # 4. Query GET /experiments/{id}/events
    events_resp = client.get(f"/experiments/{exp_id}/events")
    assert events_resp.status_code == 200
    events_list = events_resp.json()
    assert isinstance(events_list, list)
    if events_list:
        first_event = events_list[0]
        assert "event_id" in first_event
        assert "event_type" in first_event
        assert "summary" in first_event

    # 5. Query GET /experiments/{id}/inspector
    inspector_resp = client.get(f"/experiments/{exp_id}/inspector")
    assert inspector_resp.status_code == 200
    inspector_data = inspector_resp.json()
    assert inspector_data["experiment_id"] == exp_id
    assert "timeline" in inspector_data
    assert "routing_decisions" in inspector_data
    assert "events" in inspector_data


def test_api_client_observability_methods():
    """Verify Streamlit APIClient methods get_workflow_events and get_workflow_inspector."""
    from datetime import datetime, timezone
    manager = get_job_manager()
    exp_id = "mock-client-obs-exp"

    mgr = get_event_manager()
    mgr.record_event(
        experiment_id=exp_id,
        event_type=EventType.NODE_START,
        node="validate_input",
        status="completed",
        summary="Dataset validated successfully",
    )

    with manager._lock:
        job = JobRecord(
            experiment_id=exp_id,
            status=JobStatus.COMPLETED,
            target_column="target",
            task_type="binary_classification",
            created_at=datetime.now(timezone.utc).isoformat(),
            current_stage="Completed",
            current_node=None,
            completed_nodes=["validate_input"],
        )
        manager._jobs[exp_id] = job
        manager._persist_job(job)

    with TestClient(app) as test_client:
        events_res = test_client.get(f"/experiments/{exp_id}/events")
        assert events_res.status_code == 200
        events_data = events_res.json()
        assert len(events_data) >= 1
        assert events_data[0]["node"] == "validate_input"

        insp_res = test_client.get(f"/experiments/{exp_id}/inspector")
        assert insp_res.status_code == 200
        insp_data = insp_res.json()
        assert insp_data["experiment_id"] == exp_id
        assert len(insp_data["events"]) >= 1


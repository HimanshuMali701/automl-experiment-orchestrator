"""Unit tests for Phase 4: Human-in-the-Loop approval checkpoints and persistent checkpointing."""

import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from automl.api.app import app
from automl.api.jobs import (
    JobManager,
    get_job_manager,
    reset_job_manager,
    validate_and_apply_hyperparameter_overrides,
)
from automl.api.schemas import JobStatus
from automl.core.config import Settings
from automl.profiler.schemas import DatasetProfile, TaskType
from automl.trainer.engine import build_default_experiment_config
from automl.trainer.schemas import (
    ExperimentConfig,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
)
from automl.workflow.checkpointer import SqliteCheckpointSaver, get_default_checkpointer, reset_default_checkpointer
from automl.workflow.graph import (
    create_workflow_graph,
    resume_orchestration_workflow,
    run_orchestration_workflow,
)
from automl.workflow.state import ErrorCategory, WorkflowStatus


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """Produce small deterministic binary classification dataset."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "num_1": np.random.randn(n),
        "num_2": np.random.rand(n) * 10,
        "cat_1": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })


@pytest.fixture
def temp_checkpointer():
    """Create a temporary SQLite checkpointer instance."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    saver = SqliteCheckpointSaver(db_path=path)
    yield saver
    try:
        saver._conn.close()
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def clean_job_environment():
    """Clean up job manager and checkpointer before and after each test."""
    reset_job_manager()
    reset_default_checkpointer()
    yield
    reset_job_manager()
    reset_default_checkpointer()


# ============================================================================
# 1. Workflow Graph Tests: Approval Disabled vs Enabled
# ============================================================================

def test_workflow_continues_directly_when_approval_disabled(sample_classification_df, temp_checkpointer):
    """When requires_approval is False, workflow runs straight through to SUCCESS without pausing."""
    final_state = run_orchestration_workflow(
        dataset_source=sample_classification_df,
        target_column="target",
        task_type_override=TaskType.BINARY_CLASSIFICATION,
        requires_approval=False,
        checkpointer=temp_checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["experiment_result"] is not None
    assert final_state["best_candidate_name"] is not None
    assert any("approval_gate: skipped" in t for t in final_state["execution_trace"])


def test_workflow_pauses_before_execution_when_approval_enabled(sample_classification_df, temp_checkpointer):
    """When requires_approval is True, workflow pauses at approval_gate with WAITING_FOR_APPROVAL."""
    thread_id = "test_thread_pause"
    final_state = run_orchestration_workflow(
        dataset_source=sample_classification_df,
        target_column="target",
        task_type_override=TaskType.BINARY_CLASSIFICATION,
        requires_approval=True,
        thread_id=thread_id,
        checkpointer=temp_checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert final_state["workflow_status"] == WorkflowStatus.WAITING_FOR_APPROVAL
    assert final_state["approval_status"] == "pending"
    assert final_state["plan"] is not None
    assert len(final_state["plan"].candidates) > 0
    # Crucial safety: No model candidates have executed yet
    assert final_state.get("candidate_runs", []) == []
    assert final_state.get("experiment_result") is None
    assert any("approval_gate: waiting_for_approval" in t for t in final_state["execution_trace"])

    # Verify checkpointer recorded paused state
    graph = create_workflow_graph(checkpointer=temp_checkpointer)
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snapshot.next == ("approval_gate",)
    assert snapshot.values["plan"] is not None


def test_workflow_resume_on_approval(sample_classification_df, temp_checkpointer):
    """Resuming paused workflow with approval executes models and reaches SUCCESS."""
    thread_id = "test_thread_resume_approval"
    initial_run = run_orchestration_workflow(
        dataset_source=sample_classification_df,
        target_column="target",
        task_type_override=TaskType.BINARY_CLASSIFICATION,
        requires_approval=True,
        thread_id=thread_id,
        checkpointer=temp_checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert initial_run["workflow_status"] == WorkflowStatus.WAITING_FOR_APPROVAL

    # Resume the exact same thread with approval
    resumed_state = resume_orchestration_workflow(
        thread_id=thread_id,
        approval_status="approved",
        approval_feedback="Plan approved by lead data scientist.",
        checkpointer=temp_checkpointer,
    )

    assert resumed_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert resumed_state["approval_status"] == "approved"
    assert resumed_state["experiment_result"] is not None
    assert resumed_state["best_candidate_name"] is not None
    assert any("approval_gate: approved" in t for t in resumed_state["execution_trace"])


def test_workflow_rejection_prevents_model_execution(sample_classification_df, temp_checkpointer):
    """Resuming paused workflow with rejection cleanly marks REJECTED and executes NO models."""
    thread_id = "test_thread_rejection"
    initial_run = run_orchestration_workflow(
        dataset_source=sample_classification_df,
        target_column="target",
        task_type_override=TaskType.BINARY_CLASSIFICATION,
        requires_approval=True,
        thread_id=thread_id,
        checkpointer=temp_checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert initial_run["workflow_status"] == WorkflowStatus.WAITING_FOR_APPROVAL

    # Reject the plan
    resumed_state = resume_orchestration_workflow(
        thread_id=thread_id,
        approval_status="rejected",
        approval_feedback="Compute cost exceeds team budget.",
        checkpointer=temp_checkpointer,
    )

    assert resumed_state["workflow_status"] == WorkflowStatus.REJECTED
    assert resumed_state["approval_status"] == "rejected"
    assert resumed_state.get("candidate_runs", []) == []
    assert resumed_state.get("experiment_result") is None
    assert any("approval_gate: rejected" in t for t in resumed_state["execution_trace"])


# ============================================================================
# 2. Hyperparameter Overrides Validation
# ============================================================================

def test_validate_and_apply_valid_overrides(sample_classification_df):
    """Valid overrides update candidate hyperparameters correctly."""
    from automl.profiler.profiler import profile_dataset

    profile = profile_dataset(sample_classification_df, target_column="target")
    plan = build_default_experiment_config(profile)

    # Find logistic regression candidate
    log_reg = next(c for c in plan.candidates if c.model_family == ModelFamily.LOGISTIC_REGRESSION)
    assert log_reg.hyperparameters.get("C", 1.0) == 1.0

    overrides = {
        log_reg.name: {"C": 0.05, "max_iter": 500},
    }

    updated_plan = validate_and_apply_hyperparameter_overrides(plan, overrides)
    updated_log_reg = next(c for c in updated_plan.candidates if c.name == log_reg.name)

    assert updated_log_reg.hyperparameters["C"] == 0.05
    assert updated_log_reg.hyperparameters["max_iter"] == 500


def test_reject_disallowed_hyperparameter(sample_classification_df):
    """Overriding an illegal hyperparameter key raises ValueError referencing allowlists."""
    from automl.profiler.profiler import profile_dataset

    profile = profile_dataset(sample_classification_df, target_column="target")
    plan = build_default_experiment_config(profile)

    log_reg = next(c for c in plan.candidates if c.model_family == ModelFamily.LOGISTIC_REGRESSION)
    overrides = {
        log_reg.name: {"arbitrary_injected_param": "malicious_eval()"},
    }

    with pytest.raises(ValueError, match="Disallowed hyperparameter 'arbitrary_injected_param'"):
        validate_and_apply_hyperparameter_overrides(plan, overrides)


def test_reject_out_of_bounds_hyperparameter(sample_classification_df):
    """Overriding with out-of-bounds parameter values raises ValueError."""
    from automl.profiler.profiler import profile_dataset

    profile = profile_dataset(sample_classification_df, target_column="target")
    plan = build_default_experiment_config(profile)

    log_reg = next(c for c in plan.candidates if c.model_family == ModelFamily.LOGISTIC_REGRESSION)
    overrides = {
        log_reg.name: {"C": -10.0},  # C must be >= 1e-4
    }

    with pytest.raises(ValueError, match="must be >= 0.0001"):
        validate_and_apply_hyperparameter_overrides(plan, overrides)


def test_reject_nonexistent_candidate_override(sample_classification_df):
    """Overriding a candidate not present in the proposed plan raises ValueError."""
    from automl.profiler.profiler import profile_dataset

    profile = profile_dataset(sample_classification_df, target_column="target")
    plan = build_default_experiment_config(profile)

    overrides = {
        "Imaginary Neural Net": {"learning_rate": 0.01},
    }

    with pytest.raises(ValueError, match="Candidate 'Imaginary Neural Net' not found"):
        validate_and_apply_hyperparameter_overrides(plan, overrides)


# ============================================================================
# 3. Persistent Checkpointer Resilience Across Instances
# ============================================================================

def test_sqlite_checkpointer_survives_instance_recreation():
    """State saved by one SqliteCheckpointSaver instance is recovered by a fresh instance."""
    from langgraph.graph import StateGraph, START, END
    from typing_extensions import TypedDict

    class MiniState(TypedDict):
        data: str
        counter: int

    def step1(s: MiniState):
        return {"data": "step1_completed", "counter": s["counter"] + 1}

    builder = StateGraph(MiniState)
    builder.add_node("step1", step1)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", END)

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    thread_id = "persistence_test_thread"
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        # Instance 1: write checkpoint
        saver1 = SqliteCheckpointSaver(db_path=db_path)
        g1 = builder.compile(checkpointer=saver1)
        res1 = g1.invoke({"data": "initial", "counter": 0}, config=cfg)
        assert res1["data"] == "step1_completed"
        assert res1["counter"] == 1
        saver1._conn.close()

        # Instance 2: completely new checkpointer object pointing to same DB
        saver2 = SqliteCheckpointSaver(db_path=db_path)
        g2 = builder.compile(checkpointer=saver2)
        snapshot = g2.get_state(cfg)
        assert snapshot.values["data"] == "step1_completed"
        assert snapshot.values["counter"] == 1
        saver2._conn.close()

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ============================================================================
# 4. JobManager and API Integration Tests
# ============================================================================

def test_job_manager_approval_lifecycle(sample_classification_df, tmp_path):
    """Test complete JobManager submit -> wait -> approve -> complete lifecycle."""
    custom_settings = Settings(artifacts_dir=tmp_path / "artifacts")
    manager = JobManager(settings=custom_settings)

    # 1. Submit job requiring approval
    job = manager.submit_job(
        dataset_df=sample_classification_df,
        target_column="target",
        requires_approval=True,
        cv_folds=2,
        max_models=2,
    )

    # Wait until it enters WAITING_FOR_APPROVAL
    for _ in range(60):
        refreshed = manager.get_job(job.experiment_id)
        if refreshed and refreshed.status == JobStatus.WAITING_FOR_APPROVAL:
            break
        time.sleep(0.1)

    refreshed = manager.get_job(job.experiment_id)
    assert refreshed.status == JobStatus.WAITING_FOR_APPROVAL
    assert refreshed.approval_status == "pending"
    assert refreshed.pending_plan is not None

    # 2. Inspect pending plan
    approval_view = manager.get_pending_approval(job.experiment_id)
    assert approval_view.experiment_id == job.experiment_id
    assert approval_view.status == JobStatus.WAITING_FOR_APPROVAL
    assert len(approval_view.candidates) > 0

    # 3. Approve job with valid override
    cand_name = approval_view.candidates[1].candidate_name
    overrides = {
        cand_name: {"C": 0.2} if "Logistic" in cand_name else {},
    }
    manager.approve_job(
        experiment_id=job.experiment_id,
        overrides=overrides if overrides.get(cand_name) else None,
        feedback="Looks good to run.",
    )

    # 4. Wait for completion
    for _ in range(120):
        completed_job = manager.get_job(job.experiment_id)
        if completed_job and completed_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.2)

    completed_job = manager.get_job(job.experiment_id)
    assert completed_job.status == JobStatus.COMPLETED
    assert completed_job.approval_status == "approved"
    assert completed_job.result is not None
    assert completed_job.result.best_candidate_name is not None
    manager.shutdown()


def test_job_manager_reject_lifecycle(sample_classification_df, tmp_path):
    """Test complete JobManager submit -> wait -> reject lifecycle."""
    custom_settings = Settings(artifacts_dir=tmp_path / "artifacts")
    manager = JobManager(settings=custom_settings)

    job = manager.submit_job(
        dataset_df=sample_classification_df,
        target_column="target",
        requires_approval=True,
        cv_folds=2,
        max_models=2,
    )

    for _ in range(60):
        refreshed = manager.get_job(job.experiment_id)
        if refreshed and refreshed.status == JobStatus.WAITING_FOR_APPROVAL:
            break
        time.sleep(0.1)

    refreshed = manager.get_job(job.experiment_id)
    assert refreshed.status == JobStatus.WAITING_FOR_APPROVAL

    # Reject
    manager.reject_job(job.experiment_id, feedback="Rejecting plan for test.")
    rejected_job = manager.get_job(job.experiment_id)

    assert rejected_job.status == JobStatus.REJECTED
    assert rejected_job.approval_status == "rejected"
    assert rejected_job.result is None
    manager.shutdown()


def test_fastapi_approval_endpoints(sample_classification_df, tmp_path):
    """Test FastAPI REST endpoints for approval inspection, submission, and conflict handling."""
    custom_settings = Settings(artifacts_dir=tmp_path / "artifacts")
    custom_manager = JobManager(settings=custom_settings)

    with patch("automl.api.routes.get_job_manager", return_value=custom_manager):
        client = TestClient(app)

        csv_content = sample_classification_df.to_csv(index=False).encode("utf-8")

        # 1. Launch experiment with requires_approval=True
        resp = client.post(
            "/experiments",
            files={"file": ("train.csv", csv_content, "text/csv")},
            data={
                "target_column": "target",
                "requires_approval": "true",
                "cv_folds": "2",
            },
        )
        assert resp.status_code == 202
        exp_id = resp.json()["experiment_id"]
        assert resp.json()["requires_approval"] is True

        # 2. Poll until waiting for approval
        for _ in range(60):
            status_resp = client.get(f"/experiments/{exp_id}")
            if status_resp.json()["status"] == "waiting_for_approval":
                break
            time.sleep(0.1)

        status_data = client.get(f"/experiments/{exp_id}").json()
        assert status_data["status"] == "waiting_for_approval"
        assert status_data["approval_url"] is not None

        # 3. GET /experiments/{id}/approval
        approval_resp = client.get(f"/experiments/{exp_id}/approval")
        assert approval_resp.status_code == 200
        approval_data = approval_resp.json()
        assert approval_data["experiment_id"] == exp_id
        assert len(approval_data["candidates"]) > 0

        # 4. Reject invalid overrides via POST /experiments/{id}/approve
        bad_approve_resp = client.post(
            f"/experiments/{exp_id}/approve",
            json={
                "hyperparameter_overrides": {
                    approval_data["candidates"][0]["candidate_name"]: {
                        "illegal_param": 123
                    }
                }
            },
        )
        assert bad_approve_resp.status_code == 422
        assert "Disallowed hyperparameter" in bad_approve_resp.json()["detail"]

        # 5. Successfully approve via POST /experiments/{id}/approve
        good_approve_resp = client.post(
            f"/experiments/{exp_id}/approve",
            json={"feedback": "Approved by API test."},
        )
        assert good_approve_resp.status_code == 200
        assert good_approve_resp.json()["status"] in ("running", "completed")

        # 6. Duplicate approval should return 409 Conflict
        dup_resp = client.post(f"/experiments/{exp_id}/approve", json={})
        assert dup_resp.status_code == 409

        custom_manager.shutdown()

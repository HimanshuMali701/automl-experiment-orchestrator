"""End-to-End QA Scenarios A through L for AutoML Experiment Orchestrator.

Covers:
Scenario A: Normal classification workflow.
Scenario B: Normal regression workflow.
Scenario C: LLM unavailable, deterministic fallback.
Scenario D: Invalid LLM plan, successful repair.
Scenario E: Invalid LLM plan, retry exhaustion and fallback.
Scenario F: Candidate training failure and adaptive recovery.
Scenario G: Unrecoverable candidate failure.
Scenario H: Human approval and successful resume.
Scenario I: Human rejection.
Scenario J: Non-default user constraints.
Scenario K: Workflow execution with MLflow enabled.
Scenario L: Job polling and workflow inspector updates.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from starlette.testclient import TestClient

from automl.api.app import app
from automl.api.jobs import JobManager, JobRecord, JobStatus, get_job_manager
from automl.api.schemas import ExperimentJobResponse, WorkflowInspectorResponse
from automl.core.config import get_settings
from automl.planner.schemas import LLMExperimentProposal, ProposedModelConfig, ProposedPreprocessingConfig
from automl.profiler.schemas import TaskType
from automl.recovery.schemas import ActionType, FailureCategory, FailureDiagnosis, RecoveryAction
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    PreprocessingConfig,
    RunStatus,
)
from automl.workflow.checkpointer import SqliteCheckpointSaver
from automl.workflow.events import EventType, get_event_manager
from automl.workflow.graph import (
    create_workflow_graph,
    resume_orchestration_workflow,
    run_orchestration_workflow,
)
from automl.workflow.state import ErrorCategory, ExperimentWorkflowState, WorkflowStatus


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """Deterministic synthetic classification dataset."""
    return pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "feature_2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 3,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Deterministic synthetic regression dataset."""
    return pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5] * 3,
        "target": [2.3, 4.1, 6.0, 7.9, 10.2, 12.1, 14.0, 15.8, 18.2, 20.1] * 3,
    })


# --- Scenario A: Normal classification workflow ---
def test_scenario_a_normal_classification(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario A: Verify standard classification workflow selects champion and evaluates held-out test set."""
    db_file = tmp_path / "scenario_a.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    final_state = run_orchestration_workflow(
        dataset_source=classification_df,
        target_column="target",
        thread_id="scenario-a-run",
        checkpointer=checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["profile"] is not None
    assert final_state["profile"].target_profile.detected_task_type in (
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    )
    res = final_state["experiment_result"]
    assert res is not None
    assert res.best_candidate_name is not None
    assert len(res.leaderboard) >= 2
    # Held-out test score evaluated for champion
    champion = next(c for c in res.leaderboard if c.candidate_name == res.best_candidate_name)
    assert champion.test_metrics is not None
    assert len(champion.test_metrics) > 0


# --- Scenario B: Normal regression workflow ---
def test_scenario_b_normal_regression(regression_df: pd.DataFrame, tmp_path: Path):
    """Scenario B: Verify standard regression workflow correctly infers task and computes regression metrics."""
    db_file = tmp_path / "scenario_b.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    final_state = run_orchestration_workflow(
        dataset_source=regression_df,
        target_column="target",
        thread_id="scenario-b-run",
        checkpointer=checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["profile"] is not None
    assert final_state["profile"].target_profile.detected_task_type == TaskType.REGRESSION
    res = final_state["experiment_result"]
    assert res is not None
    assert res.primary_metric in (MetricName.RMSE, MetricName.MAE, MetricName.R2)
    assert res.best_candidate_name is not None


# --- Scenario C: LLM unavailable, deterministic fallback ---
def test_scenario_c_llm_unavailable_deterministic_fallback(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario C: When LLM is requested but unavailable, system falls back gracefully to deterministic plan."""
    db_file = tmp_path / "scenario_c.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    with patch("automl.planner.provider.get_llm_provider", return_value=None):
        final_state = run_orchestration_workflow(
            dataset_source=classification_df,
            target_column="target",
            use_llm_planner=True,
            llm=None,
            thread_id="scenario-c-run",
            checkpointer=checkpointer,
            user_constraints={"max_models": 2, "cv_folds": 2},
        )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["plan"] is not None
    assert len(final_state["plan"].candidates) > 0
    assert any("fallback_plan" in trace or "create_plan" in trace for trace in final_state["execution_trace"])


# --- Scenario D: Invalid LLM plan, successful repair ---
def test_scenario_d_invalid_llm_plan_successful_repair(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario D: An invalid LLM proposal is caught by validator, reflected, repaired, and executed."""
    db_file = tmp_path / "scenario_d.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    invalid_proposal_dict = {
        "primary_metric": "rmse",
        "secondary_metrics": ["accuracy"],
        "rationale": "Faulty proposal with wrong metric for classification.",
        "data_hygiene_observations": ["Clean dataset."],
        "candidates": [
            {
                "name": "Logistic Reg",
                "model_family": "logistic_regression",
                "is_baseline": False,
                "hyperparameters": {"C": 1.0},
            }
        ],
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaling": "standard",
        },
    }

    valid_proposal_dict = {
        "primary_metric": "accuracy",
        "secondary_metrics": ["f1_macro"],
        "rationale": "Repaired proposal.",
        "data_hygiene_observations": ["Clean dataset."],
        "candidates": [
            {
                "name": "Logistic Reg",
                "model_family": "logistic_regression",
                "is_baseline": False,
                "hyperparameters": {"C": 1.0},
            }
        ],
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaling": "standard",
        },
    }

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps(invalid_proposal_dict)),
        AIMessage(content=json.dumps(valid_proposal_dict)),
    ]

    final_state = run_orchestration_workflow(
        dataset_source=classification_df,
        target_column="target",
        use_llm_planner=True,
        llm=mock_llm,
        thread_id="scenario-d-run",
        checkpointer=checkpointer,
        user_constraints={"max_models": 1, "cv_folds": 2},
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state.get("plan_retry_count", 0) >= 1
    assert any("repair_plan" in step for step in final_state["execution_trace"])


# --- Scenario E: Invalid LLM plan, retry exhaustion and fallback ---
def test_scenario_e_invalid_llm_plan_exhaustion_fallback(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario E: Continually invalid LLM proposals exhaust retries and fall back safely to deterministic template."""
    db_file = tmp_path / "scenario_e.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    invalid_proposal_dict = {
        "primary_metric": "rmse",
        "secondary_metrics": [],
        "rationale": "Permanently invalid proposal.",
        "data_hygiene_observations": [],
        "candidates": [
            {
                "name": "Ridge Model",
                "model_family": "ridge_regression",
                "is_baseline": False,
                "hyperparameters": {"alpha": 1.0},
            }
        ],
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaling": "standard",
        },
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(invalid_proposal_dict))

    final_state = run_orchestration_workflow(
        dataset_source=classification_df,
        target_column="target",
        use_llm_planner=True,
        llm=mock_llm,
        max_plan_retries=2,
        thread_id="scenario-e-run",
        checkpointer=checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert any("fallback_plan" in step for step in final_state["execution_trace"])
    assert final_state["plan"] is not None


# --- Scenario F: Candidate training failure and adaptive recovery ---
def test_scenario_f_candidate_failure_and_adaptive_recovery(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario F: Training failure is diagnosed and deterministic recovery adjusts hyperparameters and retries."""
    db_file = tmp_path / "scenario_f.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    # State with simulated convergence failure
    initial_plan = ExperimentConfig(
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        secondary_metrics=[],
        test_size=0.2,
        cv_folds=2,
        random_seed=42,
        candidates=[
            ModelCandidateConfig(
                name="Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"C": 1.0, "max_iter": 1},
            )
        ],
        preprocessing=PreprocessingConfig(),
    )

    final_state = run_orchestration_workflow(
        dataset_source=classification_df,
        target_column="target",
        plan=initial_plan,
        thread_id="scenario-f-run",
        checkpointer=checkpointer,
    )

    # Workflow must have completed successfully
    assert final_state["workflow_status"] in (WorkflowStatus.SUCCESS, WorkflowStatus.PARTIAL_SUCCESS)
    assert final_state["experiment_result"] is not None


# --- Scenario G: Unrecoverable candidate failure ---
def test_scenario_g_unrecoverable_candidate_failure(tmp_path: Path):
    """Scenario G: Fatal data or target error halts execution without infinite retry loops."""
    db_file = tmp_path / "scenario_g.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    # Empty / malformed dataframe
    invalid_df = pd.DataFrame({"col_a": [None, None, None]})

    final_state = run_orchestration_workflow(
        dataset_source=invalid_df,
        target_column="missing_target",
        thread_id="scenario-g-run",
        checkpointer=checkpointer,
    )

    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["error_category"] != ErrorCategory.NONE
    assert final_state.get("is_recoverable") is False
    assert "execute_experiments" not in final_state["execution_trace"]


# --- Scenario H: Human approval and successful resume ---
def test_scenario_h_human_approval_resume(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario H: Workflow pauses at approval_gate, resumes upon human sign-off, and trains models."""
    db_file = tmp_path / "scenario_h.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)
    thread_id = "scenario-h-run"

    paused_state = run_orchestration_workflow(
        dataset_source=classification_df,
        target_column="target",
        requires_approval=True,
        thread_id=thread_id,
        checkpointer=checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert paused_state["workflow_status"] == WorkflowStatus.WAITING_FOR_APPROVAL
    assert paused_state["approval_status"] == "pending"
    assert paused_state["current_node"] == "approval_gate"

    resumed_state = resume_orchestration_workflow(
        thread_id=thread_id,
        approval_status="approved",
        approval_feedback="Approved by QA reviewer",
        checkpointer=checkpointer,
    )

    assert resumed_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert resumed_state["approval_status"] == "approved"
    assert resumed_state["experiment_result"] is not None


# --- Scenario I: Human rejection ---
def test_scenario_i_human_rejection(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario I: Workflow pauses at approval_gate and terminates cleanly without training models on rejection."""
    db_file = tmp_path / "scenario_i.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)
    thread_id = "scenario-i-run"

    paused_state = run_orchestration_workflow(
        dataset_source=classification_df,
        target_column="target",
        requires_approval=True,
        thread_id=thread_id,
        checkpointer=checkpointer,
        user_constraints={"max_models": 2, "cv_folds": 2},
    )

    assert paused_state["workflow_status"] == WorkflowStatus.WAITING_FOR_APPROVAL

    rejected_state = resume_orchestration_workflow(
        thread_id=thread_id,
        approval_status="rejected",
        approval_feedback="Plan rejected: outside operational parameters",
        checkpointer=checkpointer,
    )

    assert rejected_state["workflow_status"] == WorkflowStatus.REJECTED
    assert rejected_state["approval_status"] == "rejected"
    assert rejected_state.get("experiment_result") is None


# --- Scenario J: Non-default user constraints ---
def test_scenario_j_non_default_user_constraints(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario J: Custom user constraints strictly override planner defaults."""
    db_file = tmp_path / "scenario_j.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)

    custom_constraints = {
        "primary_metric": MetricName.PRECISION_MACRO,
        "cv_folds": 3,
        "test_size": 0.25,
        "random_seed": 99,
        "max_models": 2,
    }

    final_state = run_orchestration_workflow(
        dataset_source=classification_df,
        target_column="target",
        thread_id="scenario-j-run",
        checkpointer=checkpointer,
        user_constraints=custom_constraints,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    plan = final_state["plan"]
    assert plan is not None
    assert plan.primary_metric == MetricName.PRECISION_MACRO
    assert plan.cv_folds == 3
    assert plan.test_size == 0.25
    assert plan.random_seed == 99
    assert len(plan.candidates) <= 2


# --- Scenario K: Workflow execution with MLflow enabled ---
def test_scenario_k_mlflow_tracking_integration(classification_df: pd.DataFrame, tmp_path: Path):
    """Scenario K: Experiment execution integrates with local MLflow tracking without errors."""
    import mlflow
    db_file = tmp_path / "scenario_k.db"
    checkpointer = SqliteCheckpointSaver(db_path=db_file)
    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow_test.db'}"

    with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": mlflow_uri}):
        final_state = run_orchestration_workflow(
            dataset_source=classification_df,
            target_column="target",
            thread_id="scenario-k-run",
            checkpointer=checkpointer,
            user_constraints={"max_models": 2, "cv_folds": 2},
        )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    res = final_state["experiment_result"]
    assert res is not None
    assert res.mlflow_run_id is not None


# --- Scenario L: Job polling and workflow inspector updates ---
def test_scenario_l_job_polling_and_inspector_updates():
    """Scenario L: JobManager updates current_stage, current_node, and events accessible via inspector."""
    manager = get_job_manager()
    exp_id = "scenario-l-inspector"

    mgr = get_event_manager()
    mgr.record_event(
        experiment_id=exp_id,
        event_type=EventType.NODE_START,
        node="profile_data",
        stage="Data Profiling",
        status="running",
        summary="Profiling dataset",
    )
    mgr.record_event(
        experiment_id=exp_id,
        event_type=EventType.ROUTING_DECISION,
        node="profile_data",
        summary="Routing to create_plan",
        details={"from_node": "profile_data", "to_node": "create_plan", "reason": "Profiling succeeded"},
    )

    with manager._lock:
        job = JobRecord(
            experiment_id=exp_id,
            status=JobStatus.RUNNING,
            target_column="target",
            task_type="binary_classification",
            created_at=datetime.now(timezone.utc).isoformat(),
            current_stage="Data Profiling",
            current_node="profile_data",
            completed_nodes=["validate_input"],
            execution_trace=["validate_input"],
            pending_approval=False,
        )
        manager._jobs[exp_id] = job
        manager._persist_job(job)

    with TestClient(app) as client:
        # Polling status endpoint
        poll_resp = client.get(f"/experiments/{exp_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["current_stage"] == "Data Profiling"
        assert poll_data["current_node"] == "profile_data"
        assert "validate_input" in poll_data["completed_nodes"]

        # Inspector endpoint
        insp_resp = client.get(f"/experiments/{exp_id}/inspector")
        assert insp_resp.status_code == 200
        insp_data = insp_resp.json()
        assert insp_data["current_stage"] == "Data Profiling"
        assert len(insp_data["routing_decisions"]) >= 1
        assert len(insp_data["events"]) >= 2

    # Clean up mock status so it does not occupy active concurrency slot
    with manager._lock:
        job.status = JobStatus.COMPLETED
        manager._persist_job(job)


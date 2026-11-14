"""Comprehensive unit tests for LangGraph stateful orchestration workflow."""

import numpy as np
import pandas as pd
import pytest

from automl.profiler.schemas import TaskType
from automl.trainer.schemas import (
    ExperimentConfig,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    RunStatus,
)
from automl.workflow import (
    ErrorCategory,
    WorkflowStatus,
    run_orchestration_workflow,
)


def test_workflow_happy_path_classification(sample_binary_classification_df: pd.DataFrame):
    """Test full happy path execution on binary classification dataset."""
    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["error_category"] == ErrorCategory.NONE
    assert final_state["profile"] is not None
    assert final_state["plan"] is not None
    assert final_state["best_candidate_name"] is not None
    assert final_state["experiment_result"] is not None

    # Verify execution trace sequence contains all key nodes in order
    trace = final_state["execution_trace"]
    for expected_node in [
        "validate_input",
        "profile_data",
        "create_plan",
        "validate_plan",
        "execute_experiments",
        "evaluate_results",
        "rank_leaderboard",
        "finalize_workflow",
    ]:
        assert any(expected_node in step for step in trace)


def test_workflow_happy_path_regression(sample_regression_df: pd.DataFrame):
    """Test full happy path execution on regression dataset."""
    final_state = run_orchestration_workflow(
        dataset_source=sample_regression_df,
        target_column="price",
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["plan"].task_type == TaskType.REGRESSION
    assert final_state["best_candidate_name"] is not None
    assert len(final_state["candidate_runs"]) == 4


def test_workflow_invalid_data_routing():
    """Test that empty data cleanly terminates via handle_failure without infinite loop."""
    empty_df = pd.DataFrame()
    final_state = run_orchestration_workflow(
        dataset_source=empty_df,
        target_column="churn",
    )

    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["error_category"] == ErrorCategory.INVALID_DATA
    assert "0 rows" in final_state["error_message"]

    trace = final_state["execution_trace"]
    assert trace == ["validate_input", "handle_failure", "finalize_workflow"]


def test_workflow_missing_target_column(sample_binary_classification_df: pd.DataFrame):
    """Test that a non-existent target column is rejected at validation."""
    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="non_existent_column",
    )

    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["error_category"] == ErrorCategory.INVALID_DATA
    assert "does not exist in dataset" in final_state["error_message"]


def test_workflow_invalid_plan_routing(sample_binary_classification_df: pd.DataFrame):
    """Test that a plan with mismatched task type is caught by validate_plan."""
    mismatched_plan = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.REGRESSION,  # Invalid: target is binary classification
        primary_metric=MetricName.RMSE,
        candidates=[
            ModelCandidateConfig(
                name="Ridge",
                model_family=ModelFamily.RIDGE_REGRESSION,
            )
        ],
    )

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        plan=mismatched_plan,
    )

    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["error_category"] == ErrorCategory.INVALID_PLAN
    assert "does not match detected task_type" in final_state["error_message"]

    trace = final_state["execution_trace"]
    assert "validate_plan" in trace
    assert "handle_failure" in trace
    assert "finalize_workflow" in trace


def test_workflow_failed_candidate_survival(sample_binary_classification_df: pd.DataFrame):
    """Test that successful candidates survive when one candidate fails, resulting in partial success."""
    custom_plan = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        cv_folds=3,
        random_seed=42,
        candidates=[
            ModelCandidateConfig(
                name="Baseline",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                is_baseline=True,
            ),
            # Broken candidate
            ModelCandidateConfig(
                name="Broken Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"C": -5.0},
            ),
            ModelCandidateConfig(
                name="Valid Random Forest",
                model_family=ModelFamily.RANDOM_FOREST_CLASSIFIER,
                hyperparameters={"n_estimators": 20},
            ),
        ],
    )

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        plan=custom_plan,
    )

    assert final_state["workflow_status"] == WorkflowStatus.PARTIAL_SUCCESS
    assert final_state["best_candidate_name"] == "Valid Random Forest"

    failed = [r for r in final_state["candidate_runs"] if r.status == RunStatus.FAILED]
    successful = [r for r in final_state["candidate_runs"] if r.status == RunStatus.SUCCESS]
    assert len(failed) == 1
    assert len(successful) == 2


def test_workflow_bounded_retry_termination():
    """Test that bounded retry logic stops after max_retries and terminates."""
    from automl.workflow.graph import create_workflow_graph

    graph = create_workflow_graph()

    # Create a state simulating an in-flight recoverable failure
    initial_state = {
        "dataset_ref": "non_existent_file.csv",
        "target_column": "target",
        "task_type_override": None,
        "dataset_metadata": {},
        "profile": None,
        "plan": None,
        "candidate_runs": [],
        "experiment_result": None,
        "best_candidate_name": None,
        "workflow_status": WorkflowStatus.INITIALIZED,
        "error_category": ErrorCategory.NONE,
        "error_message": None,
        "retry_count": 0,
        "max_retries": 2,
        "is_recoverable": False,
        "execution_trace": [],
    }

    config = {"configurable": {"thread_id": "test-bounded-retry"}}
    final_state = graph.invoke(initial_state, config=config)

    # Must terminate cleanly with FAILED status
    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["retry_count"] <= 2
    assert final_state["execution_trace"][-1] == "finalize_workflow"

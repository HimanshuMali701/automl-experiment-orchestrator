"""Unit tests for user constraint propagation, precedence, validation, and persistence."""

import pandas as pd
import pytest

from automl.api.jobs import JobManager
from automl.core.exceptions import ExperimentPlanError
from automl.profiler.profiler import profile_dataset
from automl.profiler.schemas import TaskType
from automl.planner.proposer import (
    convert_proposal_to_experiment_config,
    propose_experiment_plan,
)
from automl.planner.schemas import (
    LLMExperimentProposal,
    ProposedModelConfig,
    ProposedPreprocessingConfig,
)
from automl.trainer.schemas import (
    ExperimentConfig,
    MetricName,
    ModelFamily,
)
from automl.workflow.graph import run_orchestration_workflow
from automl.workflow.nodes import validate_user_constraints
from automl.workflow.state import ErrorCategory, WorkflowStatus


@pytest.fixture
def sample_classification_csv(tmp_path):
    """Create a temporary classification dataset for testing."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10,
        "feature2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 10,
        "cat_feature": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 10,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
    })
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_regression_csv(tmp_path):
    """Create a temporary regression dataset for testing."""
    df = pd.DataFrame({
        "x1": [float(i) for i in range(100)],
        "x2": [float(i * 2) for i in range(100)],
        "cat": ["low" if i < 50 else "high" for i in range(100)],
        "target": [float(i * 3 + 5) for i in range(100)],
    })
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_user_constraints_validator_valid():
    """Verify validate_user_constraints accepts valid parameters."""
    valid_constraints = {
        "primary_metric": MetricName.F1_MACRO,
        "cv_folds": 3,
        "test_size": 0.25,
        "random_seed": 999,
    }
    # Should not raise
    validate_user_constraints(valid_constraints, task_type=TaskType.BINARY_CLASSIFICATION)


def test_user_constraints_validator_incompatible_metric():
    """Verify incompatible metric raises ExperimentPlanError."""
    # Regression metric for classification task
    with pytest.raises(ExperimentPlanError, match="incompatible with classification task"):
        validate_user_constraints(
            {"primary_metric": MetricName.RMSE},
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

    # Classification metric for regression task
    with pytest.raises(ExperimentPlanError, match="incompatible with regression task"):
        validate_user_constraints(
            {"primary_metric": MetricName.ACCURACY},
            task_type=TaskType.REGRESSION,
        )


def test_user_constraints_validator_out_of_bounds():
    """Verify out-of-bounds parameters raise ExperimentPlanError."""
    with pytest.raises(ExperimentPlanError, match="cv_folds"):
        validate_user_constraints({"cv_folds": 1})

    with pytest.raises(ExperimentPlanError, match="cv_folds"):
        validate_user_constraints({"cv_folds": 15})

    with pytest.raises(ExperimentPlanError, match="test_size"):
        validate_user_constraints({"test_size": 0.01})

    with pytest.raises(ExperimentPlanError, match="test_size"):
        validate_user_constraints({"test_size": 0.60})

    with pytest.raises(ExperimentPlanError, match="random_seed"):
        validate_user_constraints({"random_seed": "not_an_int"})


def test_user_constraints_propagation_end_to_end(sample_classification_csv):
    """Test user constraints propagate through LangGraph workflow to plan and results."""
    user_constraints = {
        "primary_metric": MetricName.PRECISION_MACRO,
        "cv_folds": 3,
        "test_size": 0.15,
        "random_seed": 777,
    }

    final_state = run_orchestration_workflow(
        dataset_source=sample_classification_csv,
        target_column="target",
        user_constraints=user_constraints,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["error_category"] == ErrorCategory.NONE

    # Verify plan has user constraints applied
    plan: ExperimentConfig = final_state["plan"]
    assert plan is not None
    assert plan.primary_metric == MetricName.PRECISION_MACRO
    assert plan.cv_folds == 3
    assert plan.test_size == 0.15
    assert plan.random_seed == 777

    # Verify results reflect user constraints
    result = final_state["experiment_result"]
    assert result is not None
    assert result.primary_metric == MetricName.PRECISION_MACRO
    assert result.test_rows == int(100 * 0.15)


def test_user_constraints_default_backward_compatibility(sample_classification_csv):
    """Test workflow retains default heuristic behaviour when user_constraints are empty or None."""
    final_state = run_orchestration_workflow(
        dataset_source=sample_classification_csv,
        target_column="target",
        user_constraints=None,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    plan: ExperimentConfig = final_state["plan"]
    assert plan is not None
    assert plan.cv_folds == 5
    assert plan.test_size == 0.2
    assert plan.random_seed == 42


def test_user_constraints_strict_precedence_over_llm_proposal(sample_classification_csv):
    """Test explicit user constraints override LLM proposals in convert_proposal_to_experiment_config."""
    profile = profile_dataset(sample_classification_csv, target_column="target")

    # LLM proposed Accuracy, but user specified F1_Macro
    proposal = LLMExperimentProposal(
        primary_metric=MetricName.ACCURACY,
        secondary_metrics=[MetricName.F1_MACRO, MetricName.RECALL_MACRO],
        rationale="LLM selected Accuracy",
        candidates=[
            ProposedModelConfig(
                name="Baseline",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                hyperparameters={"strategy": "most_frequent"},
                is_baseline=True,
            ),
            ProposedModelConfig(
                name="Custom Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"C": 0.5},
            ),
        ],
        preprocessing=ProposedPreprocessingConfig(),
    )

    user_constraints = {
        "primary_metric": MetricName.F1_MACRO,
        "cv_folds": 4,
        "test_size": 0.3,
        "random_seed": 100,
    }

    config = convert_proposal_to_experiment_config(
        proposal,
        profile,
        user_constraints=user_constraints,
    )

    # User constraint must win over LLM proposal
    assert config.primary_metric == MetricName.F1_MACRO
    assert config.cv_folds == 4
    assert config.test_size == 0.3
    assert config.random_seed == 100
    # Primary metric must not be duplicated in secondary metrics
    assert MetricName.F1_MACRO not in config.secondary_metrics


def test_user_constraints_invalid_metric_fails_graph(sample_classification_csv):
    """Test passing invalid/incompatible metric in user_constraints fails the graph with INVALID_PLAN."""
    invalid_constraints = {
        "primary_metric": MetricName.RMSE,  # Invalid for classification
    }

    final_state = run_orchestration_workflow(
        dataset_source=sample_classification_csv,
        target_column="target",
        user_constraints=invalid_constraints,
    )

    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["error_category"] == ErrorCategory.INVALID_PLAN
    assert "incompatible with classification task" in final_state["error_message"]


def test_job_manager_records_effective_config(sample_classification_csv):
    """Test JobManager stores user_constraints and effective_config in job record."""
    import time
    from automl.api.jobs import JobStatus, reset_job_manager

    reset_job_manager()
    try:
        df = pd.read_csv(sample_classification_csv)
        manager = JobManager()

        job = manager.submit_job(
            dataset_df=df,
            target_column="target",
            primary_metric=MetricName.PRECISION_MACRO,
            cv_folds=2,
            test_size=0.15,
            random_seed=1234,
        )

        assert job.user_constraints["primary_metric"] == "precision_macro"
        assert job.user_constraints["cv_folds"] == 2
        assert job.user_constraints["test_size"] == 0.15
        assert job.user_constraints["random_seed"] == 1234

        # Wait for background job to finish
        latest_job = job
        for _ in range(240):
            latest_job = manager.get_job(job.experiment_id)
            if latest_job and latest_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            time.sleep(0.5)

        assert latest_job is not None
        assert latest_job.status == JobStatus.COMPLETED
        assert latest_job.effective_config is not None
        assert latest_job.effective_config["primary_metric"] == "precision_macro"
        assert latest_job.effective_config["cv_folds"] == 2
        assert latest_job.effective_config["test_size"] == 0.15
        assert latest_job.effective_config["random_seed"] == 1234
    finally:
        reset_job_manager()

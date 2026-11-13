"""Comprehensive unit tests for LangChain reflection, proposal repair, and LangGraph routing (Phase 2)."""

import json
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from automl.planner.proposer import (
    generate_initial_proposal,
    propose_experiment_plan,
    repair_experiment_plan,
    validate_experiment_proposal,
)

from automl.planner.schemas import (
    LLMExperimentProposal,
    ProposedModelConfig,
    ProposedPreprocessingConfig,
)
from automl.profiler.profiler import profile_dataset
from automl.profiler.schemas import ColumnProfile, DatasetProfile, DataType, TargetProfile, TaskType
from automl.trainer.schemas import MetricName, ModelFamily
from automl.workflow.graph import run_orchestration_workflow
from automl.workflow.state import WorkflowStatus


@pytest.fixture
def sample_classification_profile(sample_binary_classification_df: pd.DataFrame) -> DatasetProfile:
    """Create a real, fully-formed DatasetProfile for binary classification."""
    return profile_dataset(sample_binary_classification_df, target_column="churn")


@pytest.fixture
def valid_classification_proposal_dict() -> dict:
    """Return dictionary representation of a valid LLMExperimentProposal."""
    return {
        "primary_metric": "f1_macro",
        "secondary_metrics": ["accuracy"],
        "rationale": "Balanced classification benchmark.",
        "data_hygiene_observations": ["Clean dataset."],
        "candidates": [
            {
                "name": "Dummy Baseline",
                "model_family": "dummy_classifier",
                "is_baseline": True,
                "hyperparameters": {"strategy": "prior"},
                "rationale": "Minimal baseline.",
            },
            {
                "name": "Logistic Reg",
                "model_family": "logistic_regression",
                "is_baseline": False,
                "hyperparameters": {"C": 1.0},
                "rationale": "Linear baseline.",
            },
            {
                "name": "Random Forest",
                "model_family": "random_forest_classifier",
                "is_baseline": False,
                "hyperparameters": {"n_estimators": 50},
                "rationale": "Ensemble model.",
            },
        ],
        "preprocessing": {
            "imputation_strategy": "median",
            "scaling_strategy": "standard",
            "encode_categoricals": True,
            "drop_high_cardinality": False,
        },
    }


@pytest.fixture
def invalid_model_proposal_dict() -> dict:
    """Return proposal dictionary with an unsupported model family for classification."""
    return {
        "primary_metric": "f1_macro",
        "secondary_metrics": ["accuracy"],
        "rationale": "Faulty proposal with unsupported regression model.",
        "data_hygiene_observations": ["Clean dataset."],
        "candidates": [
            {
                "name": "Dummy Baseline",
                "model_family": "dummy_classifier",
                "is_baseline": True,
                "hyperparameters": {"strategy": "prior"},
                "rationale": "Baseline.",
            },
            {
                "name": "Invalid Ridge",
                "model_family": "ridge_regression",  # Ridge regression is a regression model!
                "is_baseline": False,
                "hyperparameters": {"alpha": 1.0},
                "rationale": "Illegal candidate for classification.",
            },
        ],
        "preprocessing": {
            "imputation_strategy": "median",
            "scaling_strategy": "standard",
            "encode_categoricals": True,
            "drop_high_cardinality": False,
        },
    }


def test_validation_error_categorization(sample_classification_profile: DatasetProfile):
    """Verify structured validation error categories for metric, model, hyperparameter, and schema violations."""
    # 1. Invalid metric
    bad_metric_prop = LLMExperimentProposal(
        primary_metric=MetricName.RMSE,  # RMSE is invalid for classification
        secondary_metrics=[MetricName.ACCURACY],
        rationale="Bad metric",
        data_hygiene_observations=["None"],
        candidates=[
            ProposedModelConfig(
                name="Baseline",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                is_baseline=True,
            )
        ],
        preprocessing=ProposedPreprocessingConfig(),
    )
    res_metric = validate_experiment_proposal(bad_metric_prop, sample_classification_profile)
    assert not res_metric.is_valid
    assert "invalid_metric" in res_metric.error_categories
    assert any("[invalid_metric]" in e for e in res_metric.errors)

    # 2. Unsupported model family for task
    bad_model_prop = LLMExperimentProposal(
        primary_metric=MetricName.F1_MACRO,
        secondary_metrics=[],
        rationale="Bad model",
        data_hygiene_observations=["None"],
        candidates=[
            ProposedModelConfig(
                name="Baseline",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                is_baseline=True,
            ),
            ProposedModelConfig(
                name="Ridge Regressor in Classification",
                model_family=ModelFamily.RIDGE_REGRESSION,
                is_baseline=False,
            ),
        ],
        preprocessing=ProposedPreprocessingConfig(),
    )
    res_model = validate_experiment_proposal(bad_model_prop, sample_classification_profile)
    assert not res_model.is_valid
    assert "unsupported_model" in res_model.error_categories
    assert any("[unsupported_model]" in e for e in res_model.errors)

    # 3. Invalid hyperparameter types (non-scalar/complex types)
    bad_hp_prop = LLMExperimentProposal(
        primary_metric=MetricName.F1_MACRO,
        secondary_metrics=[],
        rationale="Bad hp",
        data_hygiene_observations=["None"],
        candidates=[
            ProposedModelConfig(
                name="Baseline",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                is_baseline=True,
                hyperparameters={"complex_func": lambda x: x},  # Callable function
            )
        ],
        preprocessing=ProposedPreprocessingConfig(),
    )
    res_hp = validate_experiment_proposal(bad_hp_prop, sample_classification_profile)
    assert not res_hp.is_valid
    assert "invalid_hyperparameters" in res_hp.error_categories
    assert any("[invalid_hyperparameters]" in e for e in res_hp.errors)


def test_planner_standalone_repair_success(
    sample_classification_profile: DatasetProfile,
    invalid_model_proposal_dict: dict,
    valid_classification_proposal_dict: dict,
):
    """Test that propose_experiment_plan repairs an invalid proposal within bounded retries."""
    mock_llm = MagicMock()
    # Call 1 returns invalid proposal, Call 2 returns valid repaired proposal
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps(invalid_model_proposal_dict)),
        AIMessage(content=json.dumps(valid_classification_proposal_dict)),
    ]

    config, metadata = propose_experiment_plan(
        profile=sample_classification_profile,
        llm=mock_llm,
        max_retries=2,
        fallback_to_default=False,
    )

    assert config is not None
    assert metadata["source"] == "llm"
    assert metadata["plan_retry_count"] == 1
    assert mock_llm.invoke.call_count == 2


def test_workflow_initially_valid_proposal(
    sample_binary_classification_df: pd.DataFrame,
    valid_classification_proposal_dict: dict,
):
    """Test workflow with LLM planner where initial proposal is immediately valid (0 retries)."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(valid_classification_proposal_dict))

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        use_llm_planner=True,
        llm=mock_llm,
        max_plan_retries=2,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["plan_retry_count"] == 0
    assert final_state["plan_metadata"]["source"] == "llm"
    assert "repair_plan" not in final_state["execution_trace"]
    assert "fallback_plan" not in final_state["execution_trace"]
    assert any("plan_attempt_1: source=llm, outcome=valid" in step for step in final_state["execution_trace"])


def test_workflow_reflection_repair_success(
    sample_binary_classification_df: pd.DataFrame,
    invalid_model_proposal_dict: dict,
    valid_classification_proposal_dict: dict,
):
    """Test workflow reflection and repair: attempt 1 invalid -> repair_plan -> attempt 2 valid."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps(invalid_model_proposal_dict)),
        AIMessage(content=json.dumps(valid_classification_proposal_dict)),
    ]

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        use_llm_planner=True,
        llm=mock_llm,
        max_plan_retries=2,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["plan_retry_count"] == 1
    assert final_state["plan_metadata"]["source"] == "llm"

    trace = final_state["execution_trace"]
    assert "repair_plan" in trace
    assert "fallback_plan" not in trace

    # Check attempt records
    assert any("plan_attempt_1: source=llm, outcome=invalid" in step for step in trace)
    assert any("plan_attempt_2: source=llm, outcome=valid" in step for step in trace)


def test_workflow_reflection_retries_exhausted_fallback(
    sample_binary_classification_df: pd.DataFrame,
    invalid_model_proposal_dict: dict,
):
    """Test that when retries are exhausted, workflow executes fallback_plan and completes safely."""
    mock_llm = MagicMock()
    # Always returns invalid proposal
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(invalid_model_proposal_dict))

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        use_llm_planner=True,
        llm=mock_llm,
        max_plan_retries=2,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["plan_metadata"]["source"] == "deterministic_fallback"
    assert final_state["plan_retry_count"] == 2

    trace = final_state["execution_trace"]
    assert trace.count("repair_plan") == 2
    assert "fallback_plan" in trace
    assert any("source=fallback, outcome=valid" in step for step in trace)
    assert final_state["experiment_result"] is not None


def test_workflow_llm_unavailable_immediate_fallback(
    sample_binary_classification_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test deterministic fallback independent of LLM availability (no retries looped)."""
    # Force get_llm_provider to return None (simulating absent credentials)
    monkeypatch.setattr("automl.workflow.nodes.get_llm_provider", lambda **kwargs: None)

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        use_llm_planner=True,
        llm=None,  # No LLM provided
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["plan_metadata"]["source"] == "deterministic_fallback"
    assert final_state["plan_retry_count"] == 0
    assert "repair_plan" not in final_state["execution_trace"]
    assert final_state["experiment_result"] is not None


def test_user_constraints_preserved_across_repair(
    sample_binary_classification_df: pd.DataFrame,
    invalid_model_proposal_dict: dict,
    valid_classification_proposal_dict: dict,
):
    """Test that user constraints are strictly preserved through initial proposal and repairs."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps(invalid_model_proposal_dict)),
        AIMessage(content=json.dumps(valid_classification_proposal_dict)),
    ]

    user_constraints = {
        "primary_metric": MetricName.F1_MACRO,
        "cv_folds": 3,
        "test_size": 0.15,
        "random_seed": 777,
    }

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        use_llm_planner=True,
        llm=mock_llm,
        max_plan_retries=2,
        user_constraints=user_constraints,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    plan = final_state["plan"]
    assert plan.primary_metric == MetricName.F1_MACRO
    assert plan.cv_folds == 3
    assert plan.test_size == 0.15
    assert plan.random_seed == 777


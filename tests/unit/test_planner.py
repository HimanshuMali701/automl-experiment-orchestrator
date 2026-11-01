"""Comprehensive unit tests for LLM planning, allowlist validation, and result explanation."""

import json
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from automl.profiler import profile_dataset
from automl.profiler.schemas import TaskType
from automl.planner import (
    LLMExperimentProposal,
    explain_experiment_results,
    propose_experiment_plan,
    validate_experiment_proposal,
)
from automl.planner.schemas import ProposedModelConfig, ProposedPreprocessingConfig
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    ExperimentResult,
    MetricName,
    ModelFamily,
    RunStatus,
)
from automl.workflow import WorkflowStatus, run_orchestration_workflow


class MockChatModel:
    """Mock LangChain chat model returning predetermined responses without network access."""

    def __init__(self, response_text: str, should_raise: bool = False):
        self.response_text = response_text
        self.should_raise = should_raise

    def invoke(self, *args, **kwargs) -> AIMessage:
        if self.should_raise:
            raise RuntimeError("Simulated LLM provider service outage / 503 error")
        return AIMessage(content=self.response_text)


@pytest.fixture
def valid_classification_proposal_json() -> str:
    """Valid JSON matching LLMExperimentProposal for binary classification."""
    return json.dumps({
        "primary_metric": "f1_macro",
        "secondary_metrics": ["accuracy", "roc_auc_ovr"],
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaling": "standard",
        },
        "candidates": [
            {
                "name": "Benchmark Baseline",
                "model_family": "dummy_classifier",
                "hyperparameters": {"strategy": "most_frequent"},
                "is_baseline": True,
            },
            {
                "name": "Tuned Logistic Regression",
                "model_family": "logistic_regression",
                "hyperparameters": {"C": 0.5, "max_iter": 500},
                "is_baseline": False,
            },
            {
                "name": "Random Forest Specialist",
                "model_family": "random_forest_classifier",
                "hyperparameters": {"n_estimators": 40, "max_depth": 8},
                "is_baseline": False,
            },
        ],
        "rationale": "Moderate class balance allows robust tree ensembles alongside regularized linear baseline.",
        "data_hygiene_observations": ["Low missingness across numerical columns."],
    })


def test_valid_plan_proposal(
    sample_binary_classification_df: pd.DataFrame,
    valid_classification_proposal_json: str,
):
    """Verify that a schema-conforming LLM plan is accepted and converted to ExperimentConfig."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")
    mock_llm = MockChatModel(response_text=valid_classification_proposal_json)

    config, metadata = propose_experiment_plan(profile, llm=mock_llm)

    assert isinstance(config, ExperimentConfig)
    assert metadata["source"] == "llm"
    assert config.primary_metric == MetricName.F1_MACRO
    assert len(config.candidates) == 3
    assert config.candidates[1].name == "Tuned Logistic Regression"
    assert config.candidates[1].hyperparameters["C"] == 0.5


def test_malformed_llm_output_triggers_deterministic_fallback(
    sample_binary_classification_df: pd.DataFrame,
):
    """Verify that unparseable non-JSON output triggers deterministic fallback."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")
    mock_llm = MockChatModel(response_text="Sure! Here is a plan: import sklearn... invalid json")

    config, metadata = propose_experiment_plan(profile, llm=mock_llm, fallback_to_default=True)

    assert isinstance(config, ExperimentConfig)
    assert metadata["source"] == "deterministic_fallback"
    assert "LLM invocation or schema parsing error" in metadata["reason"]
    # Fallback plan contains standard candidates
    assert any(c.is_baseline for c in config.candidates)


def test_unsupported_model_proposal_rejection(
    sample_binary_classification_df: pd.DataFrame,
):
    """Verify that proposing an unapproved model family is rejected."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")

    # Proposal with a regression model for a classification task
    invalid_proposal = LLMExperimentProposal(
        primary_metric=MetricName.F1_MACRO,
        candidates=[
            ProposedModelConfig(
                name="Illegal Ridge",
                model_family=ModelFamily.RIDGE_REGRESSION,  # Invalid for classification
                is_baseline=False,
            )
        ],
        rationale="Invalid model family test",
    )

    validation_res = validate_experiment_proposal(invalid_proposal, profile)
    assert not validation_res.is_valid
    assert any("not in allowlist" in err for err in validation_res.errors)


def test_unsupported_model_triggers_fallback_in_proposer(
    sample_binary_classification_df: pd.DataFrame,
):
    """Verify that proposer automatically falls back to default plan when LLM violates allowlist."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")

    invalid_json = json.dumps({
        "primary_metric": "f1_macro",
        "secondary_metrics": [],
        "preprocessing": {},
        "candidates": [
            {
                "name": "Disallowed Model",
                "model_family": "ridge_regression",  # Disallowed for classification
                "hyperparameters": {},
                "is_baseline": False,
            }
        ],
        "rationale": "Attempting unauthorized model family",
    })
    mock_llm = MockChatModel(response_text=invalid_json)

    config, metadata = propose_experiment_plan(profile, llm=mock_llm, fallback_to_default=True)

    assert metadata["source"] == "deterministic_fallback"
    assert "rejected" in metadata["reason"].lower()
    assert config.candidates[0].is_baseline is True


def test_missing_credentials_fallback(sample_binary_classification_df: pd.DataFrame):
    """Verify clean deterministic fallback when no LLM API key is configured."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")

    # Pass llm=None and empty settings
    config, metadata = propose_experiment_plan(profile, llm=None, fallback_to_default=True)

    assert isinstance(config, ExperimentConfig)
    assert metadata["source"] == "deterministic_fallback"
    assert "No LLM provider or credentials" in metadata["reason"]


def test_provider_api_error_fallback(sample_binary_classification_df: pd.DataFrame):
    """Verify that runtime API failures (e.g. 503, timeout) safely trigger fallback."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")
    failing_llm = MockChatModel(response_text="", should_raise=True)

    config, metadata = propose_experiment_plan(profile, llm=failing_llm, fallback_to_default=True)

    assert metadata["source"] == "deterministic_fallback"
    assert "Simulated LLM provider service outage" in metadata["reason"]


def test_result_explainer_grounded_in_data():
    """Verify that result explanation adheres strictly to recorded metrics without hallucination."""
    res = ExperimentResult(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        primary_metric=MetricName.F1_MACRO,
        best_candidate_name="Champion Forest",
        train_rows=80,
        test_rows=20,
        total_execution_time_seconds=2.5,
        leaderboard=[
            CandidateRunResult(
                candidate_name="Champion Forest",
                model_family=ModelFamily.RANDOM_FOREST_CLASSIFIER,
                status=RunStatus.SUCCESS,
                cv_mean_metrics={"f1_macro": 0.9250},
                cv_std_metrics={"f1_macro": 0.0150},
                test_metrics={"f1_macro": 0.9400},
                fit_time_seconds=1.2,
            ),
            CandidateRunResult(
                candidate_name="Baseline Dummy",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                is_baseline=True,
                status=RunStatus.SUCCESS,
                cv_mean_metrics={"f1_macro": 0.5000},
                cv_std_metrics={"f1_macro": 0.0000},
                test_metrics={"f1_macro": 0.5000},
                fit_time_seconds=0.01,
            ),
        ],
    )

    explanation = explain_experiment_results(res, llm=None)

    assert explanation.champion_model == "Champion Forest"
    assert "0.9250" in explanation.summary or "0.9400" in explanation.summary
    assert "+0.4250" in explanation.baseline_comparison


def test_workflow_with_llm_planner_and_explanation(
    sample_binary_classification_df: pd.DataFrame,
    valid_classification_proposal_json: str,
    monkeypatch,
):
    """Verify LangGraph workflow integration with LLM planner and result explanation."""
    mock_llm = MockChatModel(response_text=valid_classification_proposal_json)
    monkeypatch.setattr("automl.planner.proposer.get_llm_provider", lambda **kw: mock_llm)
    monkeypatch.setattr("automl.planner.explainer.get_llm_provider", lambda **kw: None)

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        use_llm_planner=True,
    )

    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["plan_metadata"]["source"] == "llm"
    assert final_state["best_candidate_name"] is not None
    assert final_state["result_explanation"] is not None
    assert final_state["result_explanation"].champion_model == final_state["best_candidate_name"]


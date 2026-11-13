"""Comprehensive unit tests for failure diagnosis, adaptive recovery, and graph routing (Phase 3)."""

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from automl.profiler.schemas import TaskType
from automl.recovery.diagnoser import (
    ALLOWED_HYPERPARAMETERS,
    apply_recovery_actions_to_plan,
    categorize_failure,
    consult_llm_for_diagnosis,
    generate_deterministic_recovery,
    validate_recovery_action,
)
from automl.recovery.schemas import (
    ActionType,
    FailureCategory,
    FailureDiagnosis,
    LLMDiagnosisRecommendation,
    RecoveryAction,
)
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    RunStatus,
)
from automl.workflow.graph import run_orchestration_workflow
from automl.workflow.state import WorkflowStatus


@pytest.fixture
def sample_plan_classification() -> ExperimentConfig:
    """Create a sample classification experiment plan."""
    return ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        cv_folds=3,
        random_seed=42,
        max_total_time_seconds=60,
        candidates=[
            ModelCandidateConfig(
                name="Baseline Dummy",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                hyperparameters={"strategy": "most_frequent"},
                is_baseline=True,
            ),
            ModelCandidateConfig(
                name="Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"C": 1.0, "max_iter": 50},
            ),
            ModelCandidateConfig(
                name="Random Forest",
                model_family=ModelFamily.RANDOM_FOREST_CLASSIFIER,
                hyperparameters={"n_estimators": 50, "max_depth": 10},
            ),
        ],
    )


# --- 1. Failure Categorization Tests ---

def test_failure_categorization_all_categories():
    """Verify that categorize_failure classifies all required failure types accurately."""
    # Convergence
    cat, root, is_rec, _ = categorize_failure("ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.")
    assert cat == FailureCategory.CONVERGENCE
    assert is_rec is True

    # Resource / Memory
    cat, root, is_rec, _ = categorize_failure("MemoryError: Unable to allocate 8.00 GiB for an array with shape (100000, 10000)")
    assert cat == FailureCategory.RESOURCE
    assert is_rec is True

    # Timeout
    cat, root, is_rec, _ = categorize_failure("TimeoutError: Candidate training reached time limit (60s). Total time limit exceeded.")
    assert cat == FailureCategory.TIMEOUT
    assert is_rec is True

    # Invalid Hyperparameters
    cat, root, is_rec, _ = categorize_failure("ValueError: Invalid parameter 'C' for estimator LogisticRegression. C must be positive.")
    assert cat == FailureCategory.INVALID_HYPERPARAMETERS
    assert is_rec is True

    # Preprocessing / Data hygiene
    cat, root, is_rec, _ = categorize_failure("ValueError: Input contains NaN, infinity or a value too large for dtype('float64').")
    assert cat == FailureCategory.PREPROCESSING_DATA
    assert is_rec is True

    # Unsupported Configuration
    cat, root, is_rec, _ = categorize_failure("ExperimentPlanError: Candidate uses unsupported model_family 'forbidden_tree' not in allowlist.")
    assert cat == FailureCategory.UNSUPPORTED_CONFIGURATION
    assert is_rec is True

    # Unknown
    cat, root, is_rec, _ = categorize_failure("OSError: [Errno 12345] Unrecognized low-level OS fault.")
    assert cat == FailureCategory.UNKNOWN
    assert is_rec is False


# --- 2. Recovery Action Generation & Safety Validation ---

def test_deterministic_recovery_generation_and_allowlist_validation(sample_plan_classification: ExperimentConfig):
    """Test deterministic recovery generation and validation against allowlist."""
    # Convergence recovery
    diag_conv = FailureDiagnosis(
        category=FailureCategory.CONVERGENCE,
        is_recoverable=True,
        root_cause="Optimization reached max iterations.",
        affected_candidates=["Logistic Regression"],
    )
    actions_conv = generate_deterministic_recovery(diag_conv, sample_plan_classification)
    assert len(actions_conv) >= 1
    assert actions_conv[0].action_type == ActionType.ADJUST_HYPERPARAMETERS
    assert "max_iter" in actions_conv[0].parameter_adjustments
    assert actions_conv[0].parameter_adjustments["max_iter"] == 100  # Doubled from 50
    assert validate_recovery_action(actions_conv[0], sample_plan_classification) is True

    # Resource recovery
    diag_res = FailureDiagnosis(
        category=FailureCategory.RESOURCE,
        is_recoverable=True,
        root_cause="Out of memory.",
        affected_candidates=["Random Forest"],
    )
    actions_res = generate_deterministic_recovery(diag_res, sample_plan_classification)
    assert any(a.action_type == ActionType.REDUCE_RESOURCES for a in actions_res)
    for a in actions_res:
        assert validate_recovery_action(a, sample_plan_classification) is True

    # Illegal action validation rejection (e.g. attempting to inject arbitrary code or disallowed parameters)
    illegal_action = RecoveryAction(
        action_type=ActionType.ADJUST_HYPERPARAMETERS,
        target_candidate="Logistic Regression",
        parameter_adjustments={"arbitrary_code_exec": "os.system('id')"},
        rationale="Malicious parameter injection attempt.",
    )
    assert validate_recovery_action(illegal_action, sample_plan_classification) is False


def test_apply_recovery_actions_to_plan(sample_plan_classification: ExperimentConfig):
    """Test applying recovery actions modifies the plan without altering evaluation metrics or test splits."""
    actions = [
        RecoveryAction(
            action_type=ActionType.ADJUST_HYPERPARAMETERS,
            target_candidate="Logistic Regression",
            parameter_adjustments={"max_iter": 500},
            rationale="Increase max_iter.",
        ),
        RecoveryAction(
            action_type=ActionType.INCREASE_TIMEOUT,
            parameter_adjustments={"max_total_time_seconds": 120},
            rationale="Extend timeout.",
        ),
    ]

    updated_plan, summaries = apply_recovery_actions_to_plan(sample_plan_classification, actions)

    # Verify modifications
    log_reg_cand = next(c for c in updated_plan.candidates if c.name == "Logistic Regression")
    assert log_reg_cand.hyperparameters["max_iter"] == 500
    assert updated_plan.max_total_time_seconds == 120
    assert len(summaries) == 2

    # Verify safety: core evaluation settings unchanged
    assert updated_plan.primary_metric == sample_plan_classification.primary_metric
    assert updated_plan.task_type == sample_plan_classification.task_type
    assert updated_plan.test_size == sample_plan_classification.test_size


# --- 3. End-to-End Graph Adaptive Recovery Tests ---

def test_workflow_adaptive_recovery_success(sample_binary_classification_df: pd.DataFrame):
    """Test end-to-end adaptive recovery: simulated failure -> diagnose_failure -> handle_failure -> retry succeeds."""
    # Plan where candidate has an invalid hyperparameter (C = -1.0)
    plan_with_invalid_hp = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        cv_folds=2,
        random_seed=42,
        candidates=[
            ModelCandidateConfig(
                name="Broken Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"C": -1.0},  # Negative C causes ValueError in scikit-learn
            ),
        ],
    )

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        plan=plan_with_invalid_hp,
        max_retries=2,
    )

    # The workflow should diagnose the negative C, reset C=1.0, and succeed on retry!
    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["best_candidate_name"] == "Broken Logistic Regression"
    assert final_state["recovery_attempt_count"] == 1

    trace = final_state["execution_trace"]
    assert "diagnose_failure" in trace
    assert "handle_failure" in trace
    assert any("failure_diagnosis:" in s for s in trace)
    assert any("recovery_attempt_1:" in s for s in trace)


def test_workflow_adaptive_recovery_skipped_candidate(sample_binary_classification_df: pd.DataFrame):
    """Test that an unfixable candidate is skipped during recovery while a fixable candidate is corrected."""
    plan_with_unfixable_cand = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        cv_folds=2,
        random_seed=42,
        candidates=[
            ModelCandidateConfig(
                name="Fixable Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"C": -1.0},
            ),
            ModelCandidateConfig(
                name="Unfixable Mystery Candidate",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"solver": "completely_invalid_solver"},
            ),
        ],
    )

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        plan=plan_with_unfixable_cand,
        max_retries=2,
    )

    # Succeeded via surviving candidate
    assert final_state["workflow_status"] == WorkflowStatus.SUCCESS
    assert final_state["best_candidate_name"] == "Fixable Logistic Regression"

    trace = final_state["execution_trace"]
    assert "diagnose_failure" in trace
    assert any("Skipped candidate 'Unfixable Mystery Candidate'" in s for s in trace)


def test_workflow_retry_budget_exhausted_terminates_safely(sample_binary_classification_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch):
    """Test that when retries reach max_retries, workflow terminates safely with FAILED status and no infinite loop."""
    from automl.trainer.engine import ExperimentEngine

    # Make run_experiment repeatedly raise a timeout error
    def always_timeout(*args, **kwargs):
        raise TimeoutError("Simulated persistent hardware timeout.")

    monkeypatch.setattr(ExperimentEngine, "run_experiment", always_timeout)

    plan = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        candidates=[
            ModelCandidateConfig(
                name="Candidate",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
            ),
        ],
    )

    final_state = run_orchestration_workflow(
        dataset_source=sample_binary_classification_df,
        target_column="churn",
        plan=plan,
        max_retries=1,  # Strict retry budget of 1
    )

    # Must terminate cleanly
    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["retry_count"] >= 1
    assert "diagnose_failure" in final_state["execution_trace"]
    assert final_state["execution_trace"][-1] == "finalize_workflow"


def test_workflow_unrecoverable_failure_immediate_termination(sample_binary_classification_df: pd.DataFrame):
    """Test that an unrecoverable failure terminates immediately without wasteful retries."""
    from automl.workflow.nodes import diagnose_failure_node

    # State with unknown unrecoverable error
    state = {
        "error_message": "Fatal system library corruption: SIGSEGV",
        "candidate_runs": [],
        "plan": None,
        "retry_count": 0,
        "max_retries": 2,
        "execution_trace": [],
    }

    result = diagnose_failure_node(state)
    assert result["is_recoverable"] is False
    assert result["failure_category"] == FailureCategory.UNKNOWN
    assert any("recoverable=False" in s for s in result["execution_trace"])


def test_llm_assisted_diagnosis_recommendation():
    """Test optional LLM diagnosis consultation with structured Pydantic schema."""
    plan = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        candidates=[
            ModelCandidateConfig(
                name="Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"max_iter": 50},
            )
        ],
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='''{
            "category": "convergence",
            "is_recoverable": true,
            "root_cause": "The solver reached max_iter before converging.",
            "recommended_action": "adjust_hyperparameters",
            "target_candidate": "Logistic Regression",
            "parameter_adjustments": {"max_iter": 500},
            "rationale": "Increasing max_iter allows the lbfgs solver more iterations to reach minimum loss."
        }'''
    )

    rec = consult_llm_for_diagnosis(
        error_msg="ConvergenceWarning: lbfgs failed to converge",
        plan=plan,
        llm=mock_llm,
    )

    assert rec is not None
    assert rec.category == FailureCategory.CONVERGENCE
    assert rec.is_recoverable is True
    assert rec.parameter_adjustments["max_iter"] == 500


def test_llm_unavailable_deterministic_fallback():
    """Test that when LLM provider is unavailable, diagnosis proceeds deterministically without error."""
    plan = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        candidates=[
            ModelCandidateConfig(
                name="Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"max_iter": 50},
            )
        ],
    )

    # When llm is None, consult_llm_for_diagnosis returns None safely
    rec = consult_llm_for_diagnosis(
        error_msg="ConvergenceWarning: did not converge",
        plan=plan,
        llm=None,
    )
    assert rec is None

    # Deterministic recovery handles it smoothly
    cat, root, is_rec, affected = categorize_failure("ConvergenceWarning: did not converge", plan=plan)
    diag = FailureDiagnosis(category=cat, is_recoverable=is_rec, root_cause=root, affected_candidates=affected)
    actions = generate_deterministic_recovery(diag, plan)
    assert len(actions) > 0
    assert actions[0].action_type == ActionType.ADJUST_HYPERPARAMETERS


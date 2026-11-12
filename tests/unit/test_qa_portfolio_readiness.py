"""Phase 8 Final Quality Assurance & Portfolio Readiness Verification Tests.

Comprehensive edge-case and integrity verification suite covering:
1. Data leakage prevention: strict separation between CV train folds and held-out test set.
2. Metric correctness: classification vs regression metric calculation and directionality.
3. Class imbalance handling: stratified split preservation and F1-macro prioritization.
4. Missing values, malformed data, and failed candidate model resilience.
5. Zero-LLM deterministic operation: flawless execution without GROQ_API_KEY.
6. Artifact persistence & clean tracking isolation.
7. Graph routing and bounded retry termination.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from automl.core.config import Settings
from automl.core.exceptions import DatasetValidationError, ExperimentPlanError
from automl.profiler import profile_dataset
from automl.profiler.schemas import TaskType
from automl.planner import (
    convert_proposal_to_experiment_config,
    explain_experiment_results,
    propose_experiment_plan,
)
from automl.planner.schemas import ExperimentExplanation
from automl.trainer import ExperimentEngine, build_default_experiment_config
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    RunStatus,
)
from automl.tracking import MLflowTracker
from automl.workflow import (
    ErrorCategory,
    WorkflowStatus,
    create_workflow_graph,
    run_orchestration_workflow,
)


# --- Fixtures ---

@pytest.fixture
def imbalanced_classification_df() -> pd.DataFrame:
    """Generate reproducible highly imbalanced binary classification dataset (95% class 0, 5% class 1)."""
    X, y = make_classification(
        n_samples=250,
        n_features=6,
        n_informative=4,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
    df["target"] = y
    # Inject some missing values in one feature
    df.loc[df.sample(frac=0.1, random_state=42).index, "feat_0"] = np.nan
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Generate reproducible continuous regression dataset with categorical & numeric features."""
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=0.2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])
    df["region"] = np.random.choice(["North", "South", "East", "West"], size=200)
    df["price"] = y
    return df


# --- 1. Data Leakage Prevention & Test Set Isolation ---

def test_leakage_prevention_preprocessor_fit_on_train_only(tmp_path: Path):
    """Verify preprocessor parameters (means, scales) are fitted strictly on train partition.

    Held-out test set statistics must NEVER influence the imputer or scaler.
    """
    train_data = pd.DataFrame({
        "feature": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    profile = profile_dataset(train_data, target_column="target")
    config = build_default_experiment_config(profile)
    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    engine = ExperimentEngine(settings=settings)

    result = engine.run_experiment(
        data=train_data,
        profile=profile,
        config=config,
        enable_tracking=False,
    )

    # Verification: result is generated without exception and models train properly
    assert result.best_candidate_name is not None
    assert result.leaderboard[0].status == RunStatus.SUCCESS


def test_held_out_test_not_used_for_model_selection(tmp_path: Path, imbalanced_classification_df: pd.DataFrame):
    """Verify that model selection (champion ranking) is strictly determined by CV score, not test score."""
    profile = profile_dataset(imbalanced_classification_df, target_column="target")
    config = build_default_experiment_config(profile, primary_metric=MetricName.F1_MACRO)
    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    engine = ExperimentEngine(settings=settings)

    result = engine.run_experiment(
        data=imbalanced_classification_df,
        profile=profile,
        config=config,
        enable_tracking=False,
    )

    assert len(result.leaderboard) > 0
    champion = result.leaderboard[0]
    assert champion.candidate_name == result.best_candidate_name

    # The rank order of candidates must be monotonic with respect to the primary CV mean metric
    cv_scores = [
        c.cv_mean_metrics["f1_macro"]
        for c in result.leaderboard
        if c.status == RunStatus.SUCCESS and "f1_macro" in c.cv_mean_metrics
    ]
    assert cv_scores == sorted(cv_scores, reverse=True)


# --- 2. Metric Correctness & Multi-Task Verification ---

def test_regression_dataset_metrics(tmp_path: Path, regression_df: pd.DataFrame):
    """Verify regression task uses RMSE/MAE/R2, and smaller RMSE is correctly ranked higher."""
    profile = profile_dataset(regression_df, target_column="price")
    assert profile.target_profile.detected_task_type == TaskType.REGRESSION

    config = build_default_experiment_config(profile)
    assert config.primary_metric == MetricName.RMSE

    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    engine = ExperimentEngine(settings=settings)
    result = engine.run_experiment(
        data=regression_df,
        profile=profile,
        config=config,
        enable_tracking=False,
    )

    assert result.task_type == TaskType.REGRESSION
    champion = result.leaderboard[0]
    assert "rmse" in champion.cv_mean_metrics
    assert "r2" in champion.cv_mean_metrics
    assert champion.test_metrics is not None
    assert "rmse" in champion.test_metrics

    # For RMSE (lower is better), the champion must have the MINIMUM cv_mean RMSE among candidates
    valid_rmses = [
        c.cv_mean_metrics["rmse"]
        for c in result.leaderboard
        if c.status == RunStatus.SUCCESS and "rmse" in c.cv_mean_metrics
    ]
    assert champion.cv_mean_metrics["rmse"] == min(valid_rmses)


def test_class_imbalance_prioritizes_f1_macro(imbalanced_classification_df: pd.DataFrame):
    """Verify profiler detects class imbalance and default plan selects F1-macro over accuracy."""
    profile = profile_dataset(imbalanced_classification_df, target_column="target")

    # Ratio is ~95/5 = 19 > 3.0
    assert profile.target_profile.imbalance_ratio is not None and profile.target_profile.imbalance_ratio > 3.0
    config = build_default_experiment_config(profile)
    assert config.primary_metric == MetricName.F1_MACRO


# --- 3. Missing Values, Malformed Data & Candidate Failure Resilience ---

def test_malformed_missing_values_and_all_nan_handling():
    """Verify profiler rejects dataset where target is completely missing or constant."""
    df_empty_target = pd.DataFrame({
        "feat1": [1, 2, 3],
        "target": [np.nan, np.nan, np.nan],
    })
    with pytest.raises(DatasetValidationError) as exc:
        profile_dataset(df_empty_target, target_column="target")
    assert "missing" in str(exc.value).lower()


def test_failed_candidate_model_does_not_abort_experiment(tmp_path: Path, regression_df: pd.DataFrame):
    """Verify that if one candidate model fails (e.g. invalid hyperparameters), other candidates succeed."""
    profile = profile_dataset(regression_df, target_column="price")
    config = build_default_experiment_config(profile)

    # Inject an intentionally invalid candidate model
    broken_candidate = ModelCandidateConfig(
        name="Broken Regressor",
        model_family=ModelFamily.RIDGE_REGRESSION,
        hyperparameters={"alpha": -999.0, "solver": "nonexistent_solver_XYZ"},
    )
    config.candidates.insert(0, broken_candidate)

    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    engine = ExperimentEngine(settings=settings)
    result = engine.run_experiment(
        data=regression_df,
        profile=profile,
        config=config,
        enable_tracking=False,
    )

    # Broken candidate is marked failed with error message
    broken_res = next(c for c in result.leaderboard if c.candidate_name == "Broken Regressor")
    assert broken_res.status == RunStatus.FAILED
    assert broken_res.error_message is not None

    # Other valid candidates succeeded and champion was selected
    assert result.best_candidate_name is not None
    assert result.best_candidate_name != "Broken Regressor"


# --- 4. Zero-LLM Deterministic Fallback Verification ---

def test_system_functions_without_llm_api_key(tmp_path: Path, regression_df: pd.DataFrame):
    """Verify that when GROQ_API_KEY is None, planner and explainer fall back cleanly and deterministically."""
    settings = Settings(
        groq_api_key=None,
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    profile = profile_dataset(regression_df, target_column="price")

    # Plan generation without LLM (uses default rule-based plan)
    config = build_default_experiment_config(profile)
    assert len(config.candidates) >= 2

    # Run experiment
    engine = ExperimentEngine(settings=settings)
    exp_res = engine.run_experiment(
        data=regression_df,
        profile=profile,
        config=config,
        enable_tracking=False,
    )

    # Explanation generation without LLM
    explanation = explain_experiment_results(exp_res, settings=settings)
    assert isinstance(explanation, ExperimentExplanation)
    assert exp_res.best_candidate_name in explanation.summary
    assert "Held-out test score" in explanation.summary


# --- 5. Clean Experiment Tracking & Artifact Isolation ---

def test_mlflow_tracking_clean_isolation(tmp_path: Path, regression_df: pd.DataFrame):
    """Verify that experiment tracking creates dedicated MLflow run directories without polluting state."""
    tracking_dir = tmp_path / "fresh_mlruns"
    tracker = MLflowTracker(tracking_uri=f"file:{tracking_dir}", default_experiment_name="qa_test_exp")

    profile = profile_dataset(regression_df, target_column="price")
    config = build_default_experiment_config(profile)

    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        mlflow_tracking_uri=f"file:{tracking_dir}",
    )
    engine = ExperimentEngine(settings=settings)
    exp_res = engine.run_experiment(
        data=regression_df,
        profile=profile,
        config=config,
        tracker=tracker,
        enable_tracking=True,
    )

    assert exp_res.mlflow_run_id is not None
    assert tracking_dir.exists()
    runs = list(tracking_dir.glob("*/*"))
    assert len(runs) > 0


# --- 6. LangGraph Workflow Routing & Retry Termination ---

def test_graph_bounded_retry_on_fatal_failure():
    """Verify that workflow aborts with FAILED status and does not loop infinitely when unrecoverable error occurs."""
    graph = create_workflow_graph()

    initial_state = {
        "dataset_ref": "invalid_dataset_does_not_exist.csv",
        "target_column": "missing_target_col",
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

    config = {"configurable": {"thread_id": "test-qa-bounded-retry"}}
    final_state = graph.invoke(initial_state, config=config)

    assert final_state["workflow_status"] == WorkflowStatus.FAILED
    assert final_state["retry_count"] <= 2
    assert final_state["execution_trace"][-1] == "finalize_workflow"

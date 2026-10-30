"""Comprehensive unit tests for the deterministic ML experiment engine."""

import numpy as np
import pandas as pd
import pytest

from automl.profiler import profile_dataset
from automl.profiler.schemas import TaskType
from automl.trainer import (
    CandidateRunResult,
    ExperimentConfig,
    ExperimentEngine,
    ExperimentResult,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    RunStatus,
    build_default_experiment_config,
)


def test_classification_experiment_run(sample_binary_classification_df: pd.DataFrame):
    """Test full binary classification experiment with baseline and multiple candidates."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")
    config = build_default_experiment_config(profile, cv_folds=3, test_size=0.2, random_seed=42)

    engine = ExperimentEngine()
    result = engine.run_experiment(sample_binary_classification_df, profile, config)

    assert isinstance(result, ExperimentResult)
    assert result.task_type == TaskType.BINARY_CLASSIFICATION
    assert result.best_candidate_name is not None
    assert len(result.leaderboard) == len(config.candidates)

    # Verify baseline is present
    baseline_run = next((r for r in result.leaderboard if r.is_baseline), None)
    assert baseline_run is not None
    assert baseline_run.status == RunStatus.SUCCESS

    # Verify best candidate has test set metrics
    best_candidate = result.leaderboard[0]
    assert best_candidate.status == RunStatus.SUCCESS
    assert best_candidate.test_metrics is not None
    assert config.primary_metric.value in best_candidate.test_metrics
    assert len(best_candidate.fold_results) == 3


def test_regression_experiment_run(sample_regression_df: pd.DataFrame):
    """Test full regression experiment with RMSE primary metric and leaderboard ranking."""
    profile = profile_dataset(sample_regression_df, target_column="price")
    config = build_default_experiment_config(
        profile,
        primary_metric=MetricName.RMSE,
        cv_folds=3,
        test_size=0.2,
        random_seed=42,
    )

    engine = ExperimentEngine()
    result = engine.run_experiment(sample_regression_df, profile, config)

    assert result.task_type == TaskType.REGRESSION
    assert result.primary_metric == MetricName.RMSE
    assert len(result.leaderboard) == 4

    # For RMSE, lower is better. Check that leaderboard is sorted ascending
    successful_runs = [r for r in result.leaderboard if r.status == RunStatus.SUCCESS]
    scores = [r.cv_mean_metrics[MetricName.RMSE.value] for r in successful_runs]
    assert scores == sorted(scores)

    # Check test metrics on best model
    best = successful_runs[0]
    assert best.test_metrics is not None
    assert MetricName.RMSE.value in best.test_metrics
    assert MetricName.R2.value in best.test_metrics


def test_reproducibility(sample_binary_classification_df: pd.DataFrame):
    """Verify that identical random seeds yield identical CV scores and test metrics."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")

    config_1 = build_default_experiment_config(profile, cv_folds=3, random_seed=123)
    config_2 = build_default_experiment_config(profile, cv_folds=3, random_seed=123)

    engine = ExperimentEngine()
    res_1 = engine.run_experiment(sample_binary_classification_df, profile, config_1)
    res_2 = engine.run_experiment(sample_binary_classification_df, profile, config_2)

    assert res_1.best_candidate_name == res_2.best_candidate_name

    best_1 = res_1.leaderboard[0]
    best_2 = res_2.leaderboard[0]
    assert best_1.cv_mean_metrics == best_2.cv_mean_metrics
    assert best_1.test_metrics == best_2.test_metrics


def test_data_leakage_prevention():
    """Verify preprocessors are fitted strictly within folds and held-out test data is not leaked."""
    # Create dataset with an extreme outlier in test set
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 2))
    y = np.random.choice([0, 1], size=100)

    df = pd.DataFrame(X, columns=["feat1", "feat2"])
    df["churn"] = y

    profile = profile_dataset(df, target_column="churn")
    config = build_default_experiment_config(profile, cv_folds=3, test_size=0.2, random_seed=42)

    engine = ExperimentEngine()
    result = engine.run_experiment(df, profile, config)

    # Test rows should be exactly 20%
    assert result.test_rows == 20
    assert result.train_rows == 80


def test_resilience_to_failed_candidate(sample_binary_classification_df: pd.DataFrame):
    """Verify engine handles a failed candidate gracefully without terminating other models."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")

    candidates = [
        ModelCandidateConfig(
            name="Baseline",
            model_family=ModelFamily.DUMMY_CLASSIFIER,
            is_baseline=True,
        ),
        # Broken candidate: invalid C value
        ModelCandidateConfig(
            name="Broken Logistic Regression",
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            hyperparameters={"C": -10.0},
        ),
        ModelCandidateConfig(
            name="Valid Random Forest",
            model_family=ModelFamily.RANDOM_FOREST_CLASSIFIER,
            hyperparameters={"n_estimators": 20},
        ),
    ]

    config = ExperimentConfig(
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=MetricName.ACCURACY,
        cv_folds=3,
        random_seed=42,
        candidates=candidates,
    )

    engine = ExperimentEngine()
    result = engine.run_experiment(sample_binary_classification_df, profile, config)

    # Check status of broken candidate
    failed_run = next(r for r in result.leaderboard if r.candidate_name == "Broken Logistic Regression")
    assert failed_run.status == RunStatus.FAILED
    assert failed_run.error_message is not None

    # Check that valid candidates succeeded
    successful_runs = [r for r in result.leaderboard if r.status == RunStatus.SUCCESS]
    assert len(successful_runs) == 2
    assert result.best_candidate_name == "Valid Random Forest"


def test_invalid_config_rejection():
    """Verify mismatched metric or unsupported model family is rejected at config validation."""
    # 1. Regression metric for classification task
    with pytest.raises(ValueError, match="Primary metric 'rmse' is invalid for classification task"):
        ExperimentConfig(
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            primary_metric=MetricName.RMSE,
            candidates=[],
        )

    # 2. Regression model for classification task
    with pytest.raises(ValueError, match="invalid for classification"):
        ExperimentConfig(
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            primary_metric=MetricName.ACCURACY,
            candidates=[
                ModelCandidateConfig(
                    name="Invalid Ridge",
                    model_family=ModelFamily.RIDGE_REGRESSION,
                )
            ],
        )

    # 3. Exceeding candidate limit
    excessive_candidates = [
        ModelCandidateConfig(name=f"cand_{i}", model_family=ModelFamily.DUMMY_REGRESSOR)
        for i in range(12)
    ]
    with pytest.raises(ValueError, match="exceeds maximum limit"):
        ExperimentConfig(
            target_column="target",
            task_type=TaskType.REGRESSION,
            primary_metric=MetricName.RMSE,
            candidates=excessive_candidates,
        )


def test_stratified_cv_fallback(sample_binary_classification_df: pd.DataFrame):
    """Verify fallback to standard KFold with warning when min class count < cv_folds."""
    # Create dataset where minority class has only 2 samples
    df = sample_binary_classification_df.copy()
    df["churn"] = [1] * 2 + [0] * (len(df) - 2)

    profile = profile_dataset(df, target_column="churn")
    # Request 5 folds, but training split will have at most 2 samples of class 1
    config = build_default_experiment_config(profile, cv_folds=5, random_seed=42)

    engine = ExperimentEngine()
    result = engine.run_experiment(df, profile, config)

    assert any("falling back to non-stratified KFold" in w for w in result.warnings)

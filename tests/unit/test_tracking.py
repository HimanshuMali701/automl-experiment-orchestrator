"""Comprehensive unit tests for MLflow experiment tracking, runs, metrics, and model artifacts."""

from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from automl.profiler import profile_dataset
from automl.tracking import MLflowTracker, compute_dataset_fingerprint
from automl.trainer import (
    ExperimentConfig,
    ExperimentEngine,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    build_default_experiment_config,
)


@pytest.fixture
def temp_tracking_uri(tmp_path: Path) -> str:
    """Provide an isolated local file-based MLflow tracking store."""
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return f"file:///{mlruns_dir.resolve().as_posix()}"


def test_dataset_fingerprint_deterministic_and_safe(
    sample_binary_classification_df: pd.DataFrame,
):
    """Verify dataset fingerprint is reproducible and does not expose raw data values."""
    fp1 = compute_dataset_fingerprint(sample_binary_classification_df)
    fp2 = compute_dataset_fingerprint(sample_binary_classification_df)

    assert fp1 == fp2
    assert fp1.startswith("sha256:")
    assert len(fp1) == 23  # 'sha256:' (7) + 16 chars

    # Modifying column structure changes fingerprint
    df_modified = sample_binary_classification_df.copy()
    df_modified["new_col"] = 1
    fp_mod = compute_dataset_fingerprint(df_modified)
    assert fp_mod != fp1


def test_mlflow_experiment_logging_and_retrieval(
    sample_binary_classification_df: pd.DataFrame,
    temp_tracking_uri: str,
):
    """Verify logging parent-child runs, validation metrics, test metrics, and pipeline artifact."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")
    config = build_default_experiment_config(
        profile,
        primary_metric=MetricName.F1_MACRO,
        cv_folds=3,
        test_size=0.2,
        random_seed=42,
    )

    tracker = MLflowTracker(
        tracking_uri=temp_tracking_uri,
        default_experiment_name="unit-test-tracking",
    )

    engine = ExperimentEngine()
    result = engine.run_experiment(
        sample_binary_classification_df,
        profile,
        config,
        tracker=tracker,
    )

    assert result.mlflow_run_id is not None
    assert result.mlflow_experiment_id is not None

    client = MlflowClient(tracking_uri=temp_tracking_uri)

    # 1. Verify parent run
    parent_run = client.get_run(result.mlflow_run_id)
    assert parent_run.data.tags.get("run_type") == "parent_orchestration"
    assert parent_run.data.tags.get("target_column") == "churn"
    assert parent_run.data.tags.get("primary_metric") == "f1_macro"
    assert "dataset_fingerprint" in parent_run.data.tags

    # Verify reproducibility parameters
    assert parent_run.data.params.get("reproducibility/random_seed") == "42"
    assert parent_run.data.params.get("reproducibility/cv_folds") == "3"
    assert parent_run.data.params.get("reproducibility/test_size") == "0.2"

    # 2. Verify candidate runs (nested under parent)
    child_runs = client.search_runs(
        experiment_ids=[result.mlflow_experiment_id],
        filter_string="tags.run_type = 'candidate_benchmark'",
    )
    assert len(child_runs) == len(config.candidates)

    champion_runs = [r for r in child_runs if r.data.tags.get("is_champion") == "true"]
    assert len(champion_runs) == 1
    champion_run = champion_runs[0]

    # Verify validation metrics (val_) are logged
    assert "val_f1_macro" in champion_run.data.metrics

    # Verify held-out test metric (test_) is logged ONLY for champion after selection
    assert "test_f1_macro" in champion_run.data.metrics

    # Verify non-champion models do NOT have test metrics
    non_champions = [r for r in child_runs if r.data.tags.get("is_champion") == "false"]
    for nc in non_champions:
        assert "test_f1_macro" not in nc.data.metrics

    # 3. Verify fitted champion model artifact is saved and reloadable
    model_uri = f"runs:/{champion_run.info.run_id}/champion_model"
    loaded_pipeline = mlflow.sklearn.load_model(model_uri)
    assert hasattr(loaded_pipeline, "predict")

    # 4. Verify get_experiment_runs DataFrame retrieval
    df_runs = tracker.get_experiment_runs(experiment_id=result.mlflow_experiment_id)
    assert len(df_runs) == len(config.candidates)
    assert "val_f1_macro" in df_runs.columns


def test_failed_candidate_logged_with_error(
    sample_binary_classification_df: pd.DataFrame,
    temp_tracking_uri: str,
):
    """Verify that candidate failures are logged with FAILED status and error summaries."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")

    candidates = [
        ModelCandidateConfig(
            name="Baseline",
            model_family=ModelFamily.DUMMY_CLASSIFIER,
            is_baseline=True,
        ),
        ModelCandidateConfig(
            name="Broken Logistic Regression",
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            hyperparameters={"C": -10.0},  # Invalid C -> raises ValueError
        ),
        ModelCandidateConfig(
            name="Valid Random Forest",
            model_family=ModelFamily.RANDOM_FOREST_CLASSIFIER,
            hyperparameters={"n_estimators": 15},
        ),
    ]

    config = ExperimentConfig(
        target_column="churn",
        task_type=profile.target_profile.detected_task_type,
        primary_metric=MetricName.ACCURACY,
        cv_folds=3,
        random_seed=42,
        candidates=candidates,
    )

    tracker = MLflowTracker(
        tracking_uri=temp_tracking_uri,
        default_experiment_name="test-failure-tracking",
    )

    engine = ExperimentEngine()
    result = engine.run_experiment(
        sample_binary_classification_df,
        profile,
        config,
        tracker=tracker,
    )

    client = MlflowClient(tracking_uri=temp_tracking_uri)
    child_runs = client.search_runs(
        experiment_ids=[result.mlflow_experiment_id],
        filter_string="tags.run_type = 'candidate_benchmark'",
    )

    failed_runs = [r for r in child_runs if r.info.status == "FAILED"]
    assert len(failed_runs) == 1

    failed_run = failed_runs[0]
    assert failed_run.data.tags.get("mlflow.runName") == "Broken Logistic Regression"
    assert "error_summary" in failed_run.data.tags
    assert "C" in failed_run.data.tags["error_summary"]


def test_no_raw_customer_data_in_tracking_metadata(
    sample_binary_classification_df: pd.DataFrame,
    temp_tracking_uri: str,
):
    """Verify that raw rows, customer records, and cell contents are not logged to MLflow."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")
    config = build_default_experiment_config(profile, cv_folds=2, random_seed=42)

    tracker = MLflowTracker(tracking_uri=temp_tracking_uri, default_experiment_name="test-privacy")
    engine = ExperimentEngine()
    result = engine.run_experiment(
        sample_binary_classification_df,
        profile,
        config,
        tracker=tracker,
    )

    client = MlflowClient(tracking_uri=temp_tracking_uri)
    parent = client.get_run(result.mlflow_run_id)

    # Check tags and parameters for sensitive string leaks
    meta_str = str(parent.data.tags) + str(parent.data.params)
    assert "sales" not in meta_str.lower()
    assert "engineering" not in meta_str.lower()
    assert "support" not in meta_str.lower()


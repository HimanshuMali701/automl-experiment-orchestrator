"""Sanity checks for environment configuration, imports, settings, and test fixtures."""

import pandas as pd
from automl import __version__
from automl.core.config import Settings, get_settings
from automl.core.exceptions import AutoMLError, DatasetValidationError
from automl.core.logging import get_logger


def test_package_metadata():
    """Verify package version is defined."""
    assert __version__ == "0.1.0"


def test_settings_initialization():
    """Verify application settings load with sensible defaults."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.app_name == "AutoML Experiment Orchestrator"
    assert settings.random_seed == 42
    assert settings.max_upload_size_mb > 0
    assert settings.max_dataset_rows > 0
    assert settings.max_training_time_seconds > 0
    assert settings.data_dir.exists()
    assert settings.artifacts_dir.exists()


def test_exceptions_formatting():
    """Verify custom exception serialization and details handling."""
    err = DatasetValidationError("Target column missing", details={"column": "churn"})
    assert "Target column missing" in str(err)
    assert err.details["column"] == "churn"
    assert isinstance(err, AutoMLError)


def test_logger_creation():
    """Verify logger instantiation."""
    logger = get_logger("sanity_check")
    assert logger.name == "automl.sanity_check"


def test_conftest_fixtures(
    sample_binary_classification_df: pd.DataFrame,
    sample_regression_df: pd.DataFrame,
    sample_multiclass_df: pd.DataFrame,
):
    """Verify synthetic fixtures provide consistent and well-formed DataFrames."""
    assert not sample_binary_classification_df.empty
    assert "churn" in sample_binary_classification_df.columns
    assert set(sample_binary_classification_df["churn"].unique()) == {0, 1}

    assert not sample_regression_df.empty
    assert "price" in sample_regression_df.columns
    assert sample_regression_df["price"].dtype in ["float64", "float32"]

    assert not sample_multiclass_df.empty
    assert "target_class" in sample_multiclass_df.columns
    assert len(sample_multiclass_df["target_class"].unique()) == 3

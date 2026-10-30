"""Comprehensive unit tests for dataset loading, validation, profiling, and task detection."""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.core.config import Settings
from automl.core.exceptions import DatasetValidationError
from automl.profiler import (
    DataType,
    DatasetProfile,
    ProfileWarning,
    TaskType,
    WarningType,
    load_dataset,
    profile_dataset,
    validate_dataset,
)


# 1. Valid data tests
def test_profile_binary_classification(sample_binary_classification_df: pd.DataFrame):
    """Test profiling on valid mixed-type binary classification dataset."""
    profile = profile_dataset(sample_binary_classification_df, target_column="churn")

    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == len(sample_binary_classification_df)
    assert profile.column_count == len(sample_binary_classification_df.columns)
    assert profile.target_profile is not None
    assert profile.target_profile.detected_task_type == TaskType.BINARY_CLASSIFICATION
    assert profile.target_profile.classes == ["0", "1"] or profile.target_profile.classes == ["1", "0"]
    assert profile.columns["age"].inferred_type == DataType.NUMERIC
    assert profile.columns["age"].numerical_summary is not None
    assert profile.columns["department"].inferred_type == DataType.CATEGORICAL
    assert profile.columns["department"].categorical_summary is not None


def test_profile_regression(sample_regression_df: pd.DataFrame):
    """Test profiling on valid regression dataset with continuous target."""
    profile = profile_dataset(sample_regression_df, target_column="price")

    assert profile.target_profile is not None
    assert profile.target_profile.detected_task_type == TaskType.REGRESSION
    assert profile.target_profile.regression_summary is not None
    assert profile.target_profile.regression_summary.mean is not None
    assert profile.columns["square_feet"].inferred_type == DataType.NUMERIC
    assert profile.columns["location_tier"].inferred_type == DataType.CATEGORICAL


def test_profile_multiclass(sample_multiclass_df: pd.DataFrame):
    """Test profiling on multiclass dataset with string labels."""
    profile = profile_dataset(sample_multiclass_df, target_column="target_class")

    assert profile.target_profile is not None
    assert profile.target_profile.detected_task_type == TaskType.MULTICLASS_CLASSIFICATION
    assert len(profile.target_profile.classes) == 3


def test_profile_from_csv_bytes(sample_regression_df: pd.DataFrame):
    """Test profiling directly from raw CSV bytes."""
    csv_bytes = sample_regression_df.to_csv(index=False).encode("utf-8")
    profile = profile_dataset(csv_bytes, target_column="price")

    assert profile.row_count == len(sample_regression_df)
    assert profile.column_count == len(sample_regression_df.columns)


def test_profile_from_csv_file(sample_regression_df: pd.DataFrame, temp_dir: Path):
    """Test profiling from a saved CSV file on disk."""
    file_path = temp_dir / "housing.csv"
    sample_regression_df.to_csv(file_path, index=False)

    profile = profile_dataset(file_path, target_column="price")
    assert profile.row_count == len(sample_regression_df)
    assert profile.target_profile.name == "price"


# 2. Empty data validation
def test_empty_dataframe_raises():
    """Verify empty DataFrame triggers DatasetValidationError."""
    empty_df = pd.DataFrame()
    with pytest.raises(DatasetValidationError, match="0 rows"):
        validate_dataset(empty_df)


def test_empty_csv_bytes_raises():
    """Verify 0-byte stream triggers DatasetValidationError."""
    with pytest.raises(DatasetValidationError, match="empty"):
        load_dataset(b"")


def test_empty_csv_file_raises(temp_dir: Path):
    """Verify 0-byte file triggers DatasetValidationError."""
    empty_file = temp_dir / "empty.csv"
    empty_file.touch()
    with pytest.raises(DatasetValidationError, match="empty"):
        load_dataset(empty_file)


# 3. Duplicate columns validation
def test_duplicate_columns_in_csv_raises():
    """Verify duplicate column names in CSV header are caught."""
    duplicate_csv = "age,income,age\n25,50000,25\n30,60000,30".encode("utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate column names"):
        load_dataset(duplicate_csv)


def test_duplicate_columns_in_dataframe_raises():
    """Verify duplicate column names in DataFrame are caught."""
    df_dups = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
    with pytest.raises(DatasetValidationError, match="duplicate column names"):
        load_dataset(df_dups)


# 4. Missing and invalid target validation
def test_missing_target_column_raises(sample_binary_classification_df: pd.DataFrame):
    """Verify referencing a non-existent target column raises error."""
    with pytest.raises(DatasetValidationError, match="does not exist in dataset"):
        validate_dataset(sample_binary_classification_df, target_column="non_existent_column")


def test_target_with_null_values_raises():
    """Verify target containing NaNs is rejected."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
        "target": [0, 1, np.nan, 0, 1],
    })
    with pytest.raises(DatasetValidationError, match=r"missing .* values"):
        validate_dataset(df, target_column="target")


def test_constant_target_raises():
    """Verify constant target (single unique value) is rejected."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [1, 1, 1, 1],
    })
    with pytest.raises(DatasetValidationError, match="at least 2 distinct target values"):
        validate_dataset(df, target_column="target")


# 5. Ambiguous target and user override
def test_ambiguous_integer_target():
    """Verify low-cardinality integer target is flagged as ambiguous."""
    # Satisfaction score 1 to 5
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "satisfaction_score": np.random.choice([1, 2, 3, 4, 5], size=100),
    })
    profile = profile_dataset(df, target_column="satisfaction_score")

    assert profile.target_profile is not None
    assert profile.target_profile.is_ambiguous is True
    assert profile.target_profile.detected_task_type == TaskType.MULTICLASS_CLASSIFICATION
    assert TaskType.REGRESSION in profile.target_profile.alternative_task_types
    assert any(w.warning_type == WarningType.AMBIGUOUS_TASK for w in profile.warnings)


def test_user_override_task_type():
    """Verify user override correctly reconfigures task type for ambiguous target."""
    df = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "score": np.random.choice([1, 2, 3, 4, 5], size=100),
    })
    profile = profile_dataset(df, target_column="score", task_type_override=TaskType.REGRESSION)

    assert profile.target_profile.detected_task_type == TaskType.REGRESSION
    assert profile.target_profile.is_ambiguous is False
    assert profile.target_profile.regression_summary is not None


def test_invalid_task_override_raises():
    """Verify invalid task type override (e.g. regression on string labels) raises error."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "category": ["low", "high", "medium", "low"],
    })
    with pytest.raises(DatasetValidationError, match="Cannot override task to 'regression' for non-numeric target"):
        profile_dataset(df, target_column="category", task_type_override=TaskType.REGRESSION)


# 6. Data quality warnings (non-destructive)
def test_data_quality_warnings_generation():
    """Verify detection of constant columns, ID columns, high cardinality, and class imbalance."""
    n = 200
    df = pd.DataFrame({
        "user_id": [f"usr_{i}" for i in range(n)],             # Likely identifier
        "constant_col": [99.0] * n,                             # Constant column
        "rare_churn": [1] * 195 + [0] * 5,                      # Severe imbalance (97.5% vs 2.5%)
        "city": [f"city_{i % 80}" for i in range(n)],           # High cardinality (>50 unique)
        "feature_x": np.random.randn(n),
    })

    profile = profile_dataset(df, target_column="rare_churn")
    warning_types = [w.warning_type for w in profile.warnings]

    assert WarningType.LIKELY_IDENTIFIER in warning_types
    assert WarningType.CONSTANT_COLUMN in warning_types
    assert WarningType.HIGH_CARDINALITY in warning_types
    assert WarningType.SEVERE_CLASS_IMBALANCE in warning_types


# 7. Non-destructive integrity
def test_dataframe_not_mutated():
    """Verify profile_dataset does not modify or drop rows/columns of the input DataFrame."""
    df = pd.DataFrame({
        "constant": [1, 1, 1],
        "id_col": ["a", "b", "c"],
        "target": [0, 1, 0],
    })
    original_cols = list(df.columns)
    original_len = len(df)

    _ = profile_dataset(df, target_column="target")

    assert list(df.columns) == original_cols
    assert len(df) == original_len


# 8. Boundary constraint enforcement
def test_max_rows_constraint_enforced():
    """Verify row limit constraint triggers validation error."""
    custom_settings = Settings(max_dataset_rows=50)
    df = pd.DataFrame({"feat": range(60), "target": [0, 1] * 30})

    with pytest.raises(DatasetValidationError, match="exceeding maximum allowed limit"):
        validate_dataset(df, settings=custom_settings)

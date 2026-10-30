"""Validation rules for loaded tabular DataFrames and target columns."""

from typing import Optional
import numpy as np
import pandas as pd

from automl.core.config import Settings, get_settings
from automl.core.exceptions import DatasetValidationError
from automl.core.logging import get_logger

logger = get_logger("profiler.validator")


def validate_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> None:
    """Validate DataFrame integrity against system bounds and target requirements.

    Args:
        df: Input DataFrame to validate.
        target_column: Optional prediction target column name.
        settings: Application settings containing boundary constraints.

    Raises:
        DatasetValidationError: If any safety constraint or target requirement is violated.
    """
    cfg = settings or get_settings()

    # 1. Empty dataset validation
    if df.empty or len(df) == 0:
        raise DatasetValidationError("Dataset has 0 rows; cannot profile or train on empty data.")
    if len(df.columns) == 0:
        raise DatasetValidationError("Dataset has 0 columns.")

    # 2. Duplicate column check in DataFrame
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique().tolist()
        raise DatasetValidationError(
            f"Dataset contains duplicate column names: {dups}",
            details={"duplicate_columns": dups},
        )

    # 3. Size constraints
    if len(df) > cfg.max_dataset_rows:
        raise DatasetValidationError(
            f"Dataset contains {len(df):,} rows, exceeding maximum allowed limit of {cfg.max_dataset_rows:,}.",
            details={"row_count": len(df), "max_rows": cfg.max_dataset_rows},
        )

    if len(df.columns) > cfg.max_dataset_cols:
        raise DatasetValidationError(
            f"Dataset contains {len(df.columns)} columns, exceeding maximum allowed limit of {cfg.max_dataset_cols}.",
            details={"col_count": len(df.columns), "max_cols": cfg.max_dataset_cols},
        )

    # 4. Target column validation (if specified)
    if target_column is not None:
        if target_column not in df.columns:
            raise DatasetValidationError(
                f"Specified target column '{target_column}' does not exist in dataset columns.",
                details={"target_column": target_column, "available_columns": list(df.columns[:10])},
            )

        target_series = df[target_column]

        # Check for missing values in target
        missing_count = int(target_series.isna().sum())
        if missing_count > 0:
            raise DatasetValidationError(
                f"Target column '{target_column}' contains {missing_count} missing (NaN/null) values. "
                "Supervised learning requires complete target labels.",
                details={"target_column": target_column, "missing_count": missing_count, "row_count": len(df)},
            )

        # Check for constant target
        unique_count = target_series.nunique(dropna=True)
        if unique_count < 2:
            raise DatasetValidationError(
                f"Target column '{target_column}' contains only {unique_count} distinct value(s). "
                "Supervised learning requires at least 2 distinct target values.",
                details={"target_column": target_column, "unique_count": unique_count},
            )

        # Check for unsupported target types (e.g., complex numbers, nested objects)
        first_val = target_series.iloc[0]
        if isinstance(first_val, (list, dict, set, complex)):
            raise DatasetValidationError(
                f"Target column '{target_column}' contains unsupported data type '{type(first_val).__name__}'. "
                "Only scalar numerical, string, or boolean targets are supported.",
                details={"target_column": target_column, "type": type(first_val).__name__},
            )

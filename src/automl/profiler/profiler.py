"""Dataset profiling engine computing statistical summaries, data hygiene warnings, and schema."""

from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Tuple, Union
import numpy as np
import pandas as pd

from automl.core.config import Settings, get_settings
from automl.core.exceptions import DatasetValidationError, ProfilingError
from automl.core.logging import get_logger
from automl.profiler.loader import load_dataset
from automl.profiler.schemas import (
    CategoricalSummary,
    ColumnProfile,
    DatasetProfile,
    DataType,
    NumericalSummary,
    ProfileWarning,
    TargetProfile,
    TaskType,
    WarningType,
)
from automl.profiler.task_detector import detect_target_profile
from automl.profiler.validator import validate_dataset

logger = get_logger("profiler.engine")


def _is_id_column(col_name: str, series: pd.Series, row_count: int) -> bool:
    """Detect whether a column behaves like an arbitrary entity identifier."""
    if row_count < 10:
        return False

    name_lower = col_name.lower()
    id_tokens = {"id", "uuid", "guid", "pk", "key", "identifier", "index"}
    has_id_name = any(token in name_lower.split("_") for token in id_tokens) or name_lower.endswith("id")

    unique_ratio = series.nunique(dropna=True) / row_count
    return (has_id_name and unique_ratio > 0.90) or (unique_ratio == 1.0 and not pd.api.types.is_float_dtype(series))


def _profile_column(
    col_name: str,
    series: pd.Series,
    row_count: int,
) -> Tuple[ColumnProfile, List[ProfileWarning]]:
    """Generate statistical profile and data quality warnings for a single column."""
    warnings: List[ProfileWarning] = []
    missing_count = int(series.isna().sum())
    missing_pct = round((missing_count / row_count) * 100.0, 2)
    non_null_series = series.dropna()
    non_null_count = len(non_null_series)
    unique_count = int(non_null_series.nunique())
    unique_ratio = round(unique_count / max(non_null_count, 1), 4)

    is_constant = (unique_count <= 1)
    if is_constant:
        warnings.append(
            ProfileWarning(
                warning_type=WarningType.CONSTANT_COLUMN,
                column=col_name,
                message=f"Column '{col_name}' has constant zero-variance values ({unique_count} distinct value).",
                metadata={"unique_count": unique_count},
            )
        )

    if missing_pct >= 30.0:
        warnings.append(
            ProfileWarning(
                warning_type=WarningType.HIGH_MISSINGNESS,
                column=col_name,
                message=f"Column '{col_name}' has {missing_pct:.1f}% missing values.",
                metadata={"missing_count": missing_count, "missing_pct": missing_pct},
            )
        )

    # Inferred data type
    raw_dtype = str(series.dtype)
    num_summary = None
    cat_summary = None

    if pd.api.types.is_bool_dtype(series):
        inferred_type = DataType.BOOLEAN
        top_cats = {str(k): int(v) for k, v in non_null_series.value_counts().head(5).items()}
        cat_summary = CategoricalSummary(
            count=non_null_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            top_categories=top_cats,
            is_high_cardinality=False,
        )

    elif pd.api.types.is_datetime64_any_dtype(series):
        inferred_type = DataType.DATETIME

    elif pd.api.types.is_numeric_dtype(series):
        if _is_id_column(col_name, series, row_count):
            inferred_type = DataType.ID_LIKE
            warnings.append(
                ProfileWarning(
                    warning_type=WarningType.LIKELY_IDENTIFIER,
                    column=col_name,
                    message=f"Column '{col_name}' appears to be a unique numeric identifier.",
                    metadata={"unique_ratio": unique_ratio},
                )
            )
        else:
            inferred_type = DataType.NUMERIC

        if non_null_count > 0:
            skew_val = float(non_null_series.skew()) if non_null_count > 2 else 0.0
            num_summary = NumericalSummary(
                count=non_null_count,
                missing_count=missing_count,
                missing_pct=missing_pct,
                mean=round(float(non_null_series.mean()), 4),
                std=round(float(non_null_series.std()), 4) if non_null_count > 1 else 0.0,
                min=round(float(non_null_series.min()), 4),
                p25=round(float(non_null_series.quantile(0.25)), 4),
                p50=round(float(non_null_series.median()), 4),
                p75=round(float(non_null_series.quantile(0.75)), 4),
                max=round(float(non_null_series.max()), 4),
                skewness=round(skew_val, 4) if not np.isnan(skew_val) else 0.0,
                zero_count=int((non_null_series == 0).sum()),
            )

    else:
        # String, object, or category
        if _is_id_column(col_name, series, row_count):
            inferred_type = DataType.ID_LIKE
            warnings.append(
                ProfileWarning(
                    warning_type=WarningType.LIKELY_IDENTIFIER,
                    column=col_name,
                    message=f"Column '{col_name}' appears to be a high-cardinality entity identifier.",
                    metadata={"unique_count": unique_count, "unique_ratio": unique_ratio},
                )
            )
        else:
            # Check for free text vs categorical
            avg_str_len = non_null_series.astype(str).str.len().mean() if non_null_count > 0 else 0
            if avg_str_len > 80 and unique_ratio > 0.5:
                inferred_type = DataType.TEXT
            else:
                inferred_type = DataType.CATEGORICAL

        is_high_cardinality = (unique_count > 50) and (inferred_type == DataType.CATEGORICAL)
        if is_high_cardinality:
            warnings.append(
                ProfileWarning(
                    warning_type=WarningType.HIGH_CARDINALITY,
                    column=col_name,
                    message=f"Column '{col_name}' has high cardinality ({unique_count} distinct categories).",
                    metadata={"unique_count": unique_count},
                )
            )

        top_cats = {str(k): int(v) for k, v in non_null_series.value_counts().head(10).items()}
        cat_summary = CategoricalSummary(
            count=non_null_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            top_categories=top_cats,
            is_high_cardinality=is_high_cardinality,
        )

    profile = ColumnProfile(
        name=col_name,
        inferred_type=inferred_type,
        raw_dtype=raw_dtype,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        is_constant=is_constant,
        numerical_summary=num_summary,
        categorical_summary=cat_summary,
    )

    return profile, warnings


def profile_dataset(
    data: Union[str, Path, bytes, BinaryIO, TextIO, pd.DataFrame],
    target_column: Optional[str] = None,
    task_type_override: Optional[Union[str, TaskType]] = None,
    settings: Optional[Settings] = None,
) -> DatasetProfile:
    """Ingest, validate, and comprehensively profile a tabular dataset.

    Args:
        data: Filepath, bytes, file stream, or existing pandas DataFrame.
        target_column: Optional name of the prediction target column.
        task_type_override: Optional explicit task type override (e.g. TaskType.REGRESSION).
        settings: Application settings for resource constraints.

    Returns:
        DatasetProfile: Fully structured profile with column statistics and data health warnings.

    Raises:
        DatasetValidationError: If data fails structural or safety validation.
        ProfilingError: If profiling computation encounters unexpected failures.
    """
    cfg = settings or get_settings()

    # 1. Load dataset safely
    df = load_dataset(data, max_size_mb=cfg.max_upload_size_mb)

    # 2. Validate structural integrity and target presence
    validate_dataset(df, target_column=target_column, settings=cfg)

    row_count = len(df)
    col_count = len(df.columns)
    logger.info(f"Profiling dataset: rows={row_count:,}, columns={col_count}, target={target_column}")

    all_warnings: List[ProfileWarning] = []

    # 3. Duplicate row analysis
    dup_rows = int(df.duplicated().sum())
    dup_rows_pct = round((dup_rows / row_count) * 100.0, 2)
    if dup_rows > 0:
        all_warnings.append(
            ProfileWarning(
                warning_type=WarningType.DUPLICATE_ROWS,
                message=f"Dataset contains {dup_rows:,} duplicate rows ({dup_rows_pct:.1f}%).",
                metadata={"duplicate_rows": dup_rows, "duplicate_rows_pct": dup_rows_pct},
            )
        )

    # 4. Profile each feature column
    column_profiles: Dict[str, ColumnProfile] = {}
    for col in df.columns:
        try:
            col_profile, col_warnings = _profile_column(str(col), df[col], row_count)
            column_profiles[str(col)] = col_profile
            all_warnings.extend(col_warnings)
        except Exception as e:
            raise ProfilingError(f"Failed to profile column '{col}': {str(e)}") from e

    # 5. Profile target column if provided
    target_profile: Optional[TargetProfile] = None
    if target_column is not None:
        try:
            target_profile, target_warnings = detect_target_profile(
                df[target_column],
                target_name=target_column,
                user_override=task_type_override,
            )
            all_warnings.extend(target_warnings)
        except Exception as e:
            if isinstance(e, (ProfilingError, DatasetValidationError)):
                raise
            raise ProfilingError(f"Failed to analyze target column '{target_column}': {str(e)}") from e

    # Memory usage
    mem_usage = int(df.memory_usage(deep=True).sum())

    return DatasetProfile(
        row_count=row_count,
        column_count=col_count,
        duplicate_rows=dup_rows,
        duplicate_rows_pct=dup_rows_pct,
        memory_usage_bytes=mem_usage,
        columns=column_profiles,
        target_profile=target_profile,
        warnings=all_warnings,
    )

"""Target validation and task type detection with ambiguity analysis and user overrides."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from automl.core.exceptions import DatasetValidationError
from automl.core.logging import get_logger
from automl.profiler.schemas import (
    NumericalSummary,
    ProfileWarning,
    TaskType,
    TargetProfile,
    WarningType,
)

logger = get_logger("profiler.task_detector")


def _is_numeric_series(series: pd.Series) -> bool:
    """Check if series has numeric dtype."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _compute_imbalance(class_counts: Dict[str, int], total_count: int) -> Tuple[Optional[float], Optional[ProfileWarning]]:
    """Evaluate class imbalance and generate a warning if severe."""
    if len(class_counts) < 2:
        return None, None

    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)

    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")
    min_pct = (min_count / total_count) * 100.0

    warning = None
    if imbalance_ratio >= 5.0 or min_pct < 10.0:
        warning = ProfileWarning(
            warning_type=WarningType.SEVERE_CLASS_IMBALANCE,
            message=(
                f"Severe class imbalance detected: majority-to-minority ratio is {imbalance_ratio:.2f}:1 "
                f"(smallest class represents only {min_pct:.1f}% of observations)."
            ),
            level="warning",
            metadata={"imbalance_ratio": imbalance_ratio, "smallest_class_pct": min_pct},
        )

    return round(imbalance_ratio, 2), warning


def detect_target_profile(
    target_series: pd.Series,
    target_name: str,
    user_override: Optional[Union[str, TaskType]] = None,
) -> Tuple[TargetProfile, List[ProfileWarning]]:
    """Analyze target series to infer task type, distribution, and potential ambiguities.

    Args:
        target_series: The target column series.
        target_name: Column name.
        user_override: Optional user-specified TaskType override.

    Returns:
        Tuple[TargetProfile, List[ProfileWarning]]: Target profile and generated non-destructive warnings.
    """
    warnings: List[ProfileWarning] = []
    total_count = len(target_series)
    unique_vals = target_series.unique()
    unique_count = len(unique_vals)

    # Convert string override to enum if needed
    override_enum: Optional[TaskType] = None
    if user_override is not None:
        if isinstance(user_override, str):
            try:
                override_enum = TaskType(user_override.lower())
            except ValueError:
                raise DatasetValidationError(
                    f"Invalid task_type_override '{user_override}'. Valid options: {[t.value for t in TaskType]}"
                )
        elif isinstance(user_override, TaskType):
            override_enum = user_override

    # If user provided override, validate feasibility
    if override_enum is not None:
        if override_enum == TaskType.REGRESSION and not _is_numeric_series(target_series):
            raise DatasetValidationError(
                f"Cannot override task to '{TaskType.REGRESSION.value}' for non-numeric target '{target_name}'."
            )
        if override_enum == TaskType.BINARY_CLASSIFICATION and unique_count != 2:
            raise DatasetValidationError(
                f"Cannot override task to '{TaskType.BINARY_CLASSIFICATION.value}' for target '{target_name}' "
                f"which has {unique_count} distinct values (expected exactly 2)."
            )

    is_ambiguous = False
    ambiguity_reason = None
    alternative_tasks: List[TaskType] = []
    confidence = 0.95

    # 1. Boolean or binary (2 distinct values)
    if pd.api.types.is_bool_dtype(target_series) or unique_count == 2:
        detected_task = TaskType.BINARY_CLASSIFICATION
        confidence = 0.98

    # 2. String, object, or categorical with > 2 unique values
    elif not _is_numeric_series(target_series):
        if unique_count <= 50:
            detected_task = TaskType.MULTICLASS_CLASSIFICATION
            confidence = 0.95
        else:
            raise DatasetValidationError(
                f"Categorical target '{target_name}' has {unique_count} unique values. "
                "High-cardinality categorical targets (>50 distinct classes) are not supported for classification."
            )

    # 3. Numeric target (integer or float)
    else:
        is_integer_like = pd.api.types.is_integer_dtype(target_series) or np.all(
            np.mod(target_series.dropna(), 1) == 0
        )

        if is_integer_like and (3 <= unique_count <= 10):
            # Ambiguous: e.g. satisfaction score 1-5, rating 1-10
            detected_task = TaskType.MULTICLASS_CLASSIFICATION
            is_ambiguous = True
            ambiguity_reason = (
                f"Target has only {unique_count} distinct integer values. It can be treated either as "
                f"multiclass classification or continuous/ordinal regression."
            )
            alternative_tasks = [TaskType.REGRESSION]
            confidence = 0.65
            warnings.append(
                ProfileWarning(
                    warning_type=WarningType.AMBIGUOUS_TASK,
                    column=target_name,
                    message=ambiguity_reason,
                    metadata={"distinct_values": sorted([int(x) for x in unique_vals[:10]])},
                )
            )
        elif is_integer_like and (11 <= unique_count <= 20):
            # Ambiguous leaning regression
            detected_task = TaskType.REGRESSION
            is_ambiguous = True
            ambiguity_reason = (
                f"Target is integer with {unique_count} distinct values; recommended as regression, "
                f"but could alternatively be modeled as multiclass classification."
            )
            alternative_tasks = [TaskType.MULTICLASS_CLASSIFICATION]
            confidence = 0.70
            warnings.append(
                ProfileWarning(
                    warning_type=WarningType.AMBIGUOUS_TASK,
                    column=target_name,
                    message=ambiguity_reason,
                    metadata={"distinct_values_count": unique_count},
                )
            )
        else:
            # Continuous regression
            detected_task = TaskType.REGRESSION
            confidence = 0.95

    # Apply user override if specified
    final_task = override_enum if override_enum is not None else detected_task

    # Build task-specific details
    classes_list = None
    class_counts_dict = None
    class_pcts_dict = None
    imbalance_ratio = None
    regression_summary = None

    if final_task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        counts = target_series.value_counts(dropna=False).to_dict()
        class_counts_dict = {str(k): int(v) for k, v in counts.items()}
        class_pcts_dict = {str(k): round((int(v) / total_count) * 100.0, 2) for k, v in counts.items()}
        classes_list = [str(k) for k in counts.keys()]
        imbalance_ratio, imb_warning = _compute_imbalance(class_counts_dict, total_count)
        if imb_warning:
            warnings.append(imb_warning)
    else:
        # Numeric regression summary
        non_null_series = target_series.dropna()
        skew_val = float(non_null_series.skew()) if len(non_null_series) > 2 else 0.0
        regression_summary = NumericalSummary(
            count=len(non_null_series),
            missing_count=int(target_series.isna().sum()),
            missing_pct=round((target_series.isna().sum() / total_count) * 100.0, 2),
            mean=round(float(non_null_series.mean()), 4),
            std=round(float(non_null_series.std()), 4) if len(non_null_series) > 1 else 0.0,
            min=round(float(non_null_series.min()), 4),
            p25=round(float(non_null_series.quantile(0.25)), 4),
            p50=round(float(non_null_series.median()), 4),
            p75=round(float(non_null_series.quantile(0.75)), 4),
            max=round(float(non_null_series.max()), 4),
            skewness=round(skew_val, 4) if not np.isnan(skew_val) else 0.0,
            zero_count=int((non_null_series == 0).sum()),
        )

    profile = TargetProfile(
        name=target_name,
        detected_task_type=final_task,
        confidence=confidence if override_enum is None else 1.0,
        is_ambiguous=is_ambiguous if override_enum is None else False,
        ambiguity_reason=ambiguity_reason if override_enum is None else None,
        alternative_task_types=alternative_tasks if override_enum is None else [],
        classes=classes_list,
        class_counts=class_counts_dict,
        class_percentages=class_pcts_dict,
        imbalance_ratio=imbalance_ratio,
        regression_summary=regression_summary,
    )

    return profile, warnings

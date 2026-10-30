"""Structured Pydantic schemas for dataset profiles, column statistics, and warnings."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported machine learning task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class DataType(str, Enum):
    """Inferred semantic data types for features."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID_LIKE = "id_like"


class WarningType(str, Enum):
    """Categorization of data health warnings."""
    SEVERE_CLASS_IMBALANCE = "severe_class_imbalance"
    HIGH_CARDINALITY = "high_cardinality"
    CONSTANT_COLUMN = "constant_column"
    LIKELY_IDENTIFIER = "likely_identifier"
    HIGH_MISSINGNESS = "high_missingness"
    AMBIGUOUS_TASK = "ambiguous_task"
    DUPLICATE_ROWS = "duplicate_rows"


class ProfileWarning(BaseModel):
    """Non-destructive diagnostic warning regarding data quality or task ambiguity."""
    warning_type: WarningType
    column: Optional[str] = None
    message: str
    level: str = "warning"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NumericalSummary(BaseModel):
    """Summary statistics for numeric features or continuous regression targets."""
    count: int
    missing_count: int
    missing_pct: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    zero_count: int = 0


class CategoricalSummary(BaseModel):
    """Summary statistics for categorical or discrete features."""
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_ratio: float
    top_categories: Dict[str, int] = Field(default_factory=dict)
    is_high_cardinality: bool = False


class ColumnProfile(BaseModel):
    """Statistical and semantic profile for an individual column."""
    name: str
    inferred_type: DataType
    raw_dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_ratio: float
    is_constant: bool = False
    numerical_summary: Optional[NumericalSummary] = None
    categorical_summary: Optional[CategoricalSummary] = None


class TargetProfile(BaseModel):
    """Profile of the designated prediction target and inferred task recommendation."""
    name: str
    detected_task_type: TaskType
    confidence: float = Field(ge=0.0, le=1.0)
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    alternative_task_types: List[TaskType] = Field(default_factory=list)
    classes: Optional[List[Any]] = None
    class_counts: Optional[Dict[str, int]] = None
    class_percentages: Optional[Dict[str, float]] = None
    imbalance_ratio: Optional[float] = None
    regression_summary: Optional[NumericalSummary] = None


class DatasetProfile(BaseModel):
    """Structured dataset profile containing dataset-level and column-level metrics."""
    row_count: int
    column_count: int
    duplicate_rows: int
    duplicate_rows_pct: float
    memory_usage_bytes: int
    columns: Dict[str, ColumnProfile]
    target_profile: Optional[TargetProfile] = None
    warnings: List[ProfileWarning] = Field(default_factory=list)

"""Structured schemas and enums for failure diagnosis, root cause analysis, and adaptive recovery."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from automl.trainer.schemas import ModelFamily


class FailureCategory(str, Enum):
    """Taxonomy of experiment and candidate model failures."""
    CONVERGENCE = "convergence"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    INVALID_HYPERPARAMETERS = "invalid_hyperparameters"
    PREPROCESSING_DATA = "preprocessing_data"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """Categorized adaptive recovery actions."""
    ADJUST_HYPERPARAMETERS = "adjust_hyperparameters"
    REDUCE_RESOURCES = "reduce_resources"
    INCREASE_TIMEOUT = "increase_timeout"
    SKIP_CANDIDATE = "skip_candidate"
    ADJUST_PREPROCESSING = "adjust_preprocessing"
    TERMINATE_UNRECOVERABLE = "terminate_unrecoverable"


class RecoveryAction(BaseModel):
    """Concrete recovery action to modify an experiment plan safely."""
    action_type: ActionType
    target_candidate: Optional[str] = None
    target_model_family: Optional[ModelFamily] = None
    parameter_adjustments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    is_safe: bool = True


class FailureDiagnosis(BaseModel):
    """Comprehensive diagnostic report analyzing root cause and recovery feasibility."""
    category: FailureCategory
    is_recoverable: bool
    root_cause: str
    affected_candidates: List[str] = Field(default_factory=list)
    suggested_actions: List[RecoveryAction] = Field(default_factory=list)
    diagnostic_details: Dict[str, Any] = Field(default_factory=dict)
    source: str = "deterministic_rules"  # 'deterministic_rules' or 'llm_assisted'


class LLMDiagnosisRecommendation(BaseModel):
    """Strict structured Pydantic schema for optional LLM diagnosis recommendations."""
    category: FailureCategory
    is_recoverable: bool
    root_cause: str
    recommended_action: ActionType
    target_candidate: Optional[str] = None
    parameter_adjustments: Dict[str, Union[int, float, str, bool]] = Field(default_factory=dict)
    rationale: str


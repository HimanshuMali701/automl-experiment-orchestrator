"""Adaptive failure diagnosis, root-cause analysis, and recovery package."""

from automl.recovery.diagnoser import (
    ALLOWED_HYPERPARAMETERS,
    apply_recovery_actions_to_plan,
    categorize_failure,
    consult_llm_for_diagnosis,
    generate_deterministic_recovery,
    validate_recovery_action,
)
from automl.recovery.schemas import (
    ActionType,
    FailureCategory,
    FailureDiagnosis,
    LLMDiagnosisRecommendation,
    RecoveryAction,
)

__all__ = [
    "FailureCategory",
    "ActionType",
    "RecoveryAction",
    "FailureDiagnosis",
    "LLMDiagnosisRecommendation",
    "categorize_failure",
    "generate_deterministic_recovery",
    "validate_recovery_action",
    "apply_recovery_actions_to_plan",
    "consult_llm_for_diagnosis",
    "ALLOWED_HYPERPARAMETERS",
]


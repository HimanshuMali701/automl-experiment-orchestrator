"""Domain-specific exception hierarchy for AutoML Experiment Orchestrator."""

from typing import Any, Dict, Optional


class AutoMLError(Exception):
    """Base exception for all AutoML errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(AutoMLError):
    """Raised when environment or application configuration is invalid."""
    pass


class DatasetValidationError(AutoMLError):
    """Raised when an uploaded dataset violates constraints (e.g., empty, corrupted, too large)."""
    pass


class ProfilingError(AutoMLError):
    """Raised when dataset profiling fails to compute statistics or infer schema."""
    pass


class ExperimentPlanError(AutoMLError):
    """Raised when an experiment plan contains invalid configurations or violates allowlists."""
    pass


class ModelTrainingError(AutoMLError):
    """Raised when model training, cross-validation, or pipeline fitting fails."""
    pass


class TrackingError(AutoMLError):
    """Raised when MLflow experiment logging or metadata persistence fails."""
    pass


class OrchestrationError(AutoMLError):
    """Raised when LangGraph workflow transitions or state updates encounter failures."""
    pass

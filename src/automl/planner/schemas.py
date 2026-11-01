"""Pydantic schemas for LLM-proposed experiment plans and explanations."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from automl.profiler.schemas import TaskType
from automl.trainer.schemas import (
    ImputationStrategy,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    PreprocessingConfig,
    ScalingStrategy,
)


class ProposedModelConfig(BaseModel):
    """A model candidate proposed by the LLM."""
    name: str = Field(description="Descriptive name of the candidate model")
    model_family: ModelFamily = Field(description="Must match an allowlisted model family")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Primitive hyperparameter key-value pairs (int, float, str, bool)",
    )
    is_baseline: bool = Field(default=False, description="Whether this candidate serves as the benchmark baseline")


class ProposedPreprocessingConfig(BaseModel):
    """Preprocessing transformation choices proposed by the LLM."""
    numeric_imputer: ImputationStrategy = Field(
        default=ImputationStrategy.MEDIAN,
        description="Imputation strategy for numerical missing values (mean, median, most_frequent, constant)",
    )
    categorical_imputer: ImputationStrategy = Field(
        default=ImputationStrategy.MOST_FREQUENT,
        description="Imputation strategy for categorical missing values (most_frequent, constant)",
    )
    scaling: ScalingStrategy = Field(
        default=ScalingStrategy.STANDARD,
        description="Feature scaling strategy (standard, minmax, robust, none)",
    )


class LLMExperimentProposal(BaseModel):
    """Structured response schema required from the LLM planner.

    The LLM cannot generate arbitrary code or execute tools. It must output
    strictly conforming JSON adhering to this schema.
    """
    primary_metric: MetricName = Field(
        description="Primary metric to rank models (must match task type)",
    )
    secondary_metrics: List[MetricName] = Field(
        default_factory=list,
        description="Secondary metrics to monitor during evaluation",
    )
    preprocessing: ProposedPreprocessingConfig = Field(
        default_factory=ProposedPreprocessingConfig,
        description="Preprocessing configuration",
    )
    candidates: List[ProposedModelConfig] = Field(
        description="List of 1 to 10 model candidates chosen from the allowlist",
    )
    rationale: str = Field(
        description="Technical justification for chosen models and metrics based on dataset profile",
    )
    data_hygiene_observations: List[str] = Field(
        default_factory=list,
        description="Observations regarding missingness, imbalance, or potential pitfalls",
    )


class PlanValidationResult(BaseModel):
    """Outcome of validating an LLM experiment plan against safety and task constraints."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_proposal: Optional[LLMExperimentProposal] = None


class ExperimentExplanation(BaseModel):
    """Structured narrative explanation grounded strictly in recorded experiment results."""
    summary: str = Field(description="Executive summary of the experiment findings")
    champion_model: str = Field(description="Winning model candidate name")
    key_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Actual measured metrics for the winning model",
    )
    baseline_comparison: str = Field(
        description="Grounded comparison between champion and baseline model",
    )
    tradeoffs_and_observations: List[str] = Field(
        default_factory=list,
        description="Observed tradeoffs based solely on actual run data",
    )


"""Experiment planning module (deterministic rules engine, schema-constrained LLM planner, and explainer)."""

from automl.planner.catalogue import (
    PLANNER_SYSTEM_PROMPT,
    format_compact_profile,
    get_supported_catalogue_description,
)
from automl.planner.explainer import explain_experiment_results
from automl.planner.proposer import (
    convert_proposal_to_experiment_config,
    propose_experiment_plan,
    validate_experiment_proposal,
)
from automl.planner.provider import get_llm_provider
from automl.planner.schemas import (
    ExperimentExplanation,
    LLMExperimentProposal,
    PlanValidationResult,
    ProposedModelConfig,
    ProposedPreprocessingConfig,
)

__all__ = [
    "propose_experiment_plan",
    "explain_experiment_results",
    "validate_experiment_proposal",
    "convert_proposal_to_experiment_config",
    "get_llm_provider",
    "format_compact_profile",
    "get_supported_catalogue_description",
    "PLANNER_SYSTEM_PROMPT",
    "LLMExperimentProposal",
    "ProposedModelConfig",
    "ProposedPreprocessingConfig",
    "PlanValidationResult",
    "ExperimentExplanation",
]

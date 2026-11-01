"""Demonstration of Phase 4: LangChain plan proposal, allowlist validation, and fallback."""

import json
import sys
from pathlib import Path

# Add src/ to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.datasets import load_breast_cancer
from langchain_core.messages import AIMessage

from automl.profiler import profile_dataset
from automl.planner import (
    LLMExperimentProposal,
    explain_experiment_results,
    propose_experiment_plan,
    validate_experiment_proposal,
)
from automl.planner.schemas import ProposedModelConfig
from automl.trainer import ExperimentEngine, MetricName, ModelFamily


class MockChatModel:
    """Mock LangChain chat model to demonstrate planning and explanation without external network."""

    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke(self, *args, **kwargs) -> AIMessage:
        return AIMessage(content=self.response_text)


def run_demo():
    print("=" * 80)
    print("AutoML Experiment Orchestrator - Phase 4 LangChain Integration Demonstration")
    print("=" * 80)

    # 1. Ingest & Profile Dataset (Offline Breast Cancer Benchmark)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    profile = profile_dataset(df, target_column="target")
    print(f"\nDataset Ingested: {len(df)} rows, target='target' ({profile.target_profile.detected_task_type.value})")

    # -------------------------------------------------------------------------
    # SCENARIO 1: VALID LLM-PROPOSED PLAN
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 1: VALID PLAN PROPOSAL (Schema Conforming & Allowlisted)")
    print("-" * 80)

    valid_json = json.dumps({
        "primary_metric": "f1_macro",
        "secondary_metrics": ["accuracy", "roc_auc_ovr"],
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaling": "standard",
        },
        "candidates": [
            {
                "name": "Benchmark Baseline",
                "model_family": "dummy_classifier",
                "hyperparameters": {"strategy": "most_frequent"},
                "is_baseline": True,
            },
            {
                "name": "L2 Regularized Logistic Regression",
                "model_family": "logistic_regression",
                "hyperparameters": {"C": 0.5, "max_iter": 500},
                "is_baseline": False,
            },
            {
                "name": "Constrained Random Forest",
                "model_family": "random_forest_classifier",
                "hyperparameters": {"n_estimators": 40, "max_depth": 6},
                "is_baseline": False,
            },
        ],
        "rationale": "Moderate class balance with 30 continuous features favors L2 logistic regression and shallow ensembles.",
        "data_hygiene_observations": ["Zero missing values detected; standard scaling recommended for distance-sensitive linear models."],
    })

    valid_llm = MockChatModel(response_text=valid_json)
    plan_1, meta_1 = propose_experiment_plan(profile, llm=valid_llm, fallback_to_default=False)

    print(f"Status:             ACCEPTED")
    print(f"Plan Source:        {meta_1['source']}")
    print(f"Primary Metric:     {plan_1.primary_metric.value}")
    print(f"Rationale:          {meta_1['rationale']}")
    print(f"Proposed Models:    {[c.name + ' (' + c.model_family.value + ')' for c in plan_1.candidates]}")

    # -------------------------------------------------------------------------
    # SCENARIO 2: REJECTED PLAN (Allowlist & Safety Violation)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 2: REJECTED PLAN (Unauthorized Model Family & Mismatched Metric)")
    print("-" * 80)

    # Candidate uses 'ridge_regression' (regression model for classification) and arbitrary metric
    invalid_proposal = LLMExperimentProposal(
        primary_metric=MetricName.RMSE,  # Invalid for classification!
        secondary_metrics=[],
        candidates=[
            ProposedModelConfig(
                name="Illegal Regression Model",
                model_family=ModelFamily.RIDGE_REGRESSION,  # Violates classification allowlist!
                is_baseline=False,
            )
        ],
        rationale="Attempting to use an unapproved model family.",
    )

    validation_res = validate_experiment_proposal(invalid_proposal, profile)
    print(f"Status:             REJECTED")
    print(f"Is Valid:           {validation_res.is_valid}")
    print("Violation Errors Caught:")
    for err in validation_res.errors:
        print(f"  [X] {err}")

    # -------------------------------------------------------------------------
    # SCENARIO 3: DETERMINISTIC FALLBACK (Safe Continuity on Unparseable Output or Missing Key)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 3: DETERMINISTIC FALLBACK (Graceful Recovery on Malformed LLM Output)")
    print("-" * 80)

    malformed_llm = MockChatModel(response_text="Error: Model returned conversational text without JSON...")
    plan_3, meta_3 = propose_experiment_plan(profile, llm=malformed_llm, fallback_to_default=True)

    print(f"Status:             FALLBACK TRIGGERED (Application Remains 100% Functional)")
    print(f"Plan Source:        {meta_3['source']}")
    print(f"Fallback Reason:    {meta_3['reason']}")
    print(f"Adopted Fallback:   {[c.name for c in plan_3.candidates]}")

    # -------------------------------------------------------------------------
    # SCENARIO 4: GROUNDED RESULT EXPLANATION (Zero Hallucination)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO 4: GROUNDED RESULT EXPLANATION (From Actual Recorded Benchmark)")
    print("-" * 80)

    engine = ExperimentEngine()
    result = engine.run_experiment(df, profile, plan_1)
    explanation = explain_experiment_results(result, llm=None)

    print(f"Champion Model:     {explanation.champion_model}")
    print(f"Summary:            {explanation.summary}")
    print(f"Baseline Delta:     {explanation.baseline_comparison}")
    print(f"Tradeoffs:          {explanation.tradeoffs_and_observations[0]}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()


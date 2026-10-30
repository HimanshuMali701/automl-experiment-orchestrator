"""Demonstration of end-to-end LangGraph state machine orchestration on tabular data."""

import sys
from pathlib import Path

# Add src/ to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.datasets import load_breast_cancer

from automl.workflow import run_orchestration_workflow


def run_demo():
    print("=" * 75)
    print("AutoML Experiment Orchestrator - Phase 3 LangGraph Workflow Demonstration")
    print("=" * 75)

    # 1. Ingest built-in dataset (offline)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    target_column = "target"
    print(f"\n1. Input Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Target Column: '{target_column}'")

    # 2. Trigger LangGraph state machine execution
    print("\n2. Executing LangGraph Stateful Experiment Workflow...")
    final_state = run_orchestration_workflow(
        dataset_source=df,
        target_column=target_column,
    )

    # 3. Inspect workflow execution trace and lifecycle status
    print("\n" + "=" * 75)
    print("WORKFLOW EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Final Status:     {final_state['workflow_status'].value.upper()}")
    print(f"Error Category:   {final_state['error_category'].value}")
    print(f"Retry Count:      {final_state['retry_count']}")
    print(f"Nodes Executed:   {' -> '.join(final_state['execution_trace'])}")

    # 4. Display profile & plan details
    profile = final_state["profile"]
    plan = final_state["plan"]
    print(f"\nTask Inferred:    {profile.target_profile.detected_task_type.value}")
    print(f"Primary Metric:   {plan.primary_metric.value}")
    print(f"CV Folds:         {plan.cv_folds}")
    print(f"Held-out Test:    {int(plan.test_size * 100)}%")

    # 5. Display Leaderboard
    print("\n" + "=" * 75)
    print(f"LEADERBOARD (Ranked by CV {plan.primary_metric.value.upper()})")
    print("=" * 75)
    print(f"{'Rank':<5} {'Model Candidate':<32} {'Status':<10} {'CV Score':<18} {'Test Score':<12}")
    print("-" * 75)

    metric_key = plan.primary_metric.value
    for rank, cand in enumerate(final_state["candidate_runs"], 1):
        if cand.status.value == "success":
            mean_score = cand.cv_mean_metrics.get(metric_key, 0.0)
            std_score = cand.cv_std_metrics.get(metric_key, 0.0)
            cv_str = f"{mean_score:.4f} ± {std_score:.4f}"
            test_str = (
                f"{cand.test_metrics.get(metric_key, 0.0):.4f}"
                if cand.test_metrics
                else "Held-out"
            )
        else:
            cv_str = "N/A"
            test_str = "N/A"

        tag = " [BASELINE]" if cand.is_baseline else ""
        print(f"{rank:<5} {cand.candidate_name + tag:<32} {cand.status.value:<10} {cv_str:<18} {test_str:<12}")

    print("-" * 75)
    print(f"Selected Champion Model: {final_state['best_candidate_name']}")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()

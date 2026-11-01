"""Demonstration of Phase 5 MLflow Experiment Tracking.

Shows:
1. Running an automated ML experiment with MLflow tracking enabled.
2. Parent-child orchestration run hierarchy.
3. Reproducibility metadata and privacy-safe dataset fingerprinting.
4. Validation metrics per candidate, and held-out test metrics logged strictly for the champion.
5. Champion Scikit-learn Pipeline logged as a versioned artifact.
6. Programmatic querying of experiment runs and leaderboard.
7. Instructions for viewing experiments in the MLflow UI.
"""

from pathlib import Path
import sys

# Add src/ to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from automl.profiler import profile_dataset
from automl.tracking import MLflowTracker
from automl.trainer import (
    ExperimentConfig,
    ExperimentEngine,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    build_default_experiment_config,
)


def create_demo_dataset() -> pd.DataFrame:
    """Generate a realistic synthetic tabular dataset for demonstration."""
    np.random.seed(42)
    n = 250

    age = np.random.randint(18, 70, size=n)
    income = np.random.normal(50000, 15000, size=n).round(2)
    department = np.random.choice(["Engineering", "Sales", "Support", "Marketing"], size=n)
    satisfaction = np.random.uniform(1.0, 5.0, size=n).round(2)

    # Churn probability based on features
    churn_logits = (
        -0.03 * age
        - 0.00002 * income
        - 0.8 * (satisfaction - 3.0)
        + (department == "Support") * 0.5
    )
    churn_prob = 1 / (1 + np.exp(-churn_logits))
    churn = (np.random.uniform(0, 1, size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "department": department,
        "satisfaction_score": satisfaction,
        "churn": churn,
    })

    # Introduce some missing values
    df.loc[np.random.choice(n, 10, replace=False), "income"] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "department"] = np.nan

    return df


def main():
    print("=" * 80)
    print(" PHASE 5 DEMO: MLflow Experiment Tracking & Reproducibility")
    print("=" * 80)

    # 1. Prepare data
    df = create_demo_dataset()
    print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Profile dataset
    profile = profile_dataset(df, target_column="churn")
    print(f"Task Detected: {profile.target_profile.detected_task_type.value}")
    print(f"Target Distribution: {profile.target_profile.class_counts}")

    # 3. Configure experiment with Baseline + Candidates
    config = build_default_experiment_config(
        profile,
        primary_metric=MetricName.F1_MACRO,
        cv_folds=3,
        test_size=0.2,
        random_seed=42,
    )
    print(f"\nExperiment Plan: {len(config.candidates)} models to benchmark")
    for c in config.candidates:
        print(f" - [{c.model_family.value}] {c.name} (baseline={c.is_baseline})")

    # 4. Initialize MLflow Tracker with local store
    mlruns_path = Path("./mlruns").resolve()
    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        default_experiment_name="automl-demo-experiments",
    )
    print(f"\nMLflow Tracker initialized: {tracking_uri}")

    # 5. Run Experiment with Tracking
    print("\nExecuting experiment engine with MLflow logging...")
    engine = ExperimentEngine()
    result = engine.run_experiment(
        df,
        profile,
        config,
        tracker=tracker,
    )

    print("\n" + "=" * 80)
    print(" EXPERIMENT COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    print(f"Parent Run ID:       {result.mlflow_run_id}")
    print(f"MLflow Experiment:   {result.mlflow_experiment_id}")
    print(f"Champion Model:      {result.best_candidate_name}")
    print(f"Total Duration:      {result.total_execution_time_seconds:.2f} seconds")
    print(f"Train/Test Split:    {result.train_rows} train rows / {result.test_rows} held-out test rows")

    print("\nLeaderboard (Cross-Validation Mean Scores):")
    print("-" * 80)
    for rank, cand in enumerate(result.leaderboard, start=1):
        champ_marker = " [CHAMPION]" if cand.candidate_name == result.best_candidate_name else ""
        mean_score = cand.cv_mean_metrics.get(result.primary_metric.value, 0.0)
        std_score = cand.cv_std_metrics.get(result.primary_metric.value, 0.0)
        print(
            f"#{rank} {cand.candidate_name:<30} | "
            f"val_{result.primary_metric.value}: {mean_score:.4f} +/- {std_score:.4f} | "
            f"fit_time: {cand.fit_time_seconds:.3f}s{champ_marker}"
        )
        if cand.candidate_name == result.best_candidate_name and cand.test_metrics:
            test_score = cand.test_metrics.get(result.primary_metric.value, 0.0)
            print(f"   --> Held-out Test Score ({result.primary_metric.value}): {test_score:.4f}")

    # 6. Query MLflow Runs via Tracker API
    print("\n" + "=" * 80)
    print(" MLFLOW RUNS QUERY (via MLflowTracker.get_experiment_runs)")
    print("=" * 80)
    df_runs = tracker.get_experiment_runs(experiment_id=result.mlflow_experiment_id)
    display_cols = [
        "candidate_name",
        "model_family",
        "is_baseline",
        "is_champion",
        "status",
        f"val_{result.primary_metric.value}",
    ]
    available_cols = [c for c in display_cols if c in df_runs.columns]
    print(df_runs[available_cols].to_string(index=False))

    # 7. How to launch UI
    print("\n" + "=" * 80)
    print(" HOW TO VIEW IN MLFLOW UI")
    print("=" * 80)
    print("To inspect interactive run comparisons, artifact trees, and metric charts:")
    print("Run in your terminal:")
    print(f"   mlflow ui --backend-store-uri ./mlruns --port 5000")
    print("Then open: http://127.0.0.1:5000 in your browser.")
    print("=" * 80)


if __name__ == "__main__":
    main()


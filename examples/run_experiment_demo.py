"""Demonstration script comparing baseline and candidate models using Phase 1 & 2 modules."""

import sys
from pathlib import Path

# Add src/ to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.datasets import load_breast_cancer

from automl.profiler import profile_dataset
from automl.trainer import (
    ExperimentEngine,
    MetricName,
    build_default_experiment_config,
)


def run_demo():
    print("=" * 70)
    print("AutoML Experiment Orchestrator - Phase 2 Engine Demonstration")
    print("=" * 70)

    # 1. Load small built-in dataset (offline, no internet required)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    target_name = "target"

    print(f"\n1. Ingested Dataset: {len(df)} rows, {len(df.columns)} columns")

    # 2. Profile dataset using Phase 1 profiler
    profile = profile_dataset(df, target_column=target_name)
    print(f"2. Data Profiling Complete:")
    print(f"   - Detected Task: {profile.target_profile.detected_task_type.value}")
    print(f"   - Class counts: {profile.target_profile.class_counts}")
    print(f"   - Imbalance ratio: {profile.target_profile.imbalance_ratio}:1")

    # 3. Build deterministic experiment plan
    config = build_default_experiment_config(
        profile,
        primary_metric=MetricName.F1_MACRO,
        cv_folds=5,
        test_size=0.2,
        random_seed=42,
    )
    print(f"\n3. Configured Experiment Plan:")
    print(f"   - Primary Metric: {config.primary_metric.value}")
    print(f"   - Cross-Validation: {config.cv_folds}-fold Stratified CV")
    print(f"   - Test Split: {int(config.test_size * 100)}% held-out test set")
    print(f"   - Candidates ({len(config.candidates)}): {[c.name for c in config.candidates]}")

    # 4. Execute experiment engine
    print(f"\n4. Executing Cross-Validation & Model Training...")
    engine = ExperimentEngine()
    result = engine.run_experiment(df, profile, config)

    # 5. Display comparative leaderboard
    print("\n" + "=" * 70)
    print(f"EXPERIMENT LEADERBOARD (Ranked by CV {config.primary_metric.value.upper()})")
    print("=" * 70)
    print(f"{'Rank':<5} {'Model Candidate':<30} {'Status':<10} {'CV F1 (Mean ± Std)':<22} {'Test F1':<10}")
    print("-" * 77)

    for rank, cand in enumerate(result.leaderboard, 1):
        if cand.status.value == "success":
            mean_f1 = cand.cv_mean_metrics.get("f1_macro", 0.0)
            std_f1 = cand.cv_std_metrics.get("f1_macro", 0.0)
            cv_str = f"{mean_f1:.4f} ± {std_f1:.4f}"
            test_f1 = (
                f"{cand.test_metrics.get('f1_macro', 0.0):.4f}"
                if cand.test_metrics
                else "N/A (Held-out)"
            )
        else:
            cv_str = "N/A"
            test_f1 = "N/A"

        baseline_tag = " [BASELINE]" if cand.is_baseline else ""
        print(f"{rank:<5} {cand.candidate_name + baseline_tag:<30} {cand.status.value:<10} {cv_str:<22} {test_f1:<10}")

    print("-" * 77)
    print(f"Selected Best Model: {result.best_candidate_name}")
    print(f"Total Execution Time: {result.total_execution_time_seconds}s")
    print(f"Train / Test Partition: {result.train_rows} / {result.test_rows} rows")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()

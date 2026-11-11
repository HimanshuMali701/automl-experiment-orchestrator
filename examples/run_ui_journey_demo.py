"""Demonstration of Phase 7: Streamlit Full User Journey.

Simulates the end-to-end workflow executed by the Streamlit dashboard:
1. Health Check (Sidebar Status).
2. View 1 & 2: Dataset Upload, Validation & Profiling.
3. View 3: Experiment Configuration & Duplicate Guard.
4. View 4: Real-time Job Progress Polling.
5. View 5: Model Leaderboard & Cross-Validation Metrics.
6. View 6: Candidate Detail & Hyperparameters.
7. View 7: Final Report (Held-out test results separated from CV, grounded explanation, and MLflow run ID).
"""

import io
from pathlib import Path
import sys
import time

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.datasets import load_breast_cancer
from starlette.testclient import TestClient

from automl.api.app import app
from automl.ui.client import APIClient


def main():
    print("=" * 80)
    print(" PHASE 7 DEMO: Streamlit Full User Journey Simulation")
    print("=" * 80)

    # Use in-process test client for zero-dependency execution
    # In live usage, Streamlit connects over HTTP to uvicorn on http://127.0.0.1:8000
    test_client = TestClient(app)

    # Wrap TestClient in APIClient adapter
    class InProcessAPIClient(APIClient):
        def check_health(self):
            res = test_client.get(f"{self.base_url}/health")
            if res.status_code != 200:
                self._handle_response_error(res)
            from automl.api.schemas import HealthResponse
            return HealthResponse.model_validate(res.json())

        def validate_dataset(self, file_bytes, filename, target_column=None, task_type_override=None):
            files = {"file": (filename, file_bytes, "text/csv")}
            data = {}
            if target_column:
                data["target_column"] = target_column
            if task_type_override:
                data["task_type_override"] = task_type_override
            res = test_client.post(f"{self.base_url}/datasets/validate", files=files, data=data)
            if res.status_code != 200:
                self._handle_response_error(res)
            from automl.api.schemas import DatasetValidationResponse
            return DatasetValidationResponse.model_validate(res.json())

        def submit_experiment(self, file_bytes, filename, target_column, primary_metric=None, **kwargs):
            files = {"file": (filename, file_bytes, "text/csv")}
            data = {"target_column": target_column, **kwargs}
            if primary_metric:
                data["primary_metric"] = primary_metric
            res = test_client.post(f"{self.base_url}/experiments", files=files, data=data)
            if res.status_code != 202:
                self._handle_response_error(res)
            from automl.api.schemas import ExperimentJobResponse
            return ExperimentJobResponse.model_validate(res.json())

        def get_experiment_status(self, experiment_id):
            res = test_client.get(f"{self.base_url}/experiments/{experiment_id}")
            if res.status_code != 200:
                self._handle_response_error(res)
            from automl.api.schemas import ExperimentJobResponse
            return ExperimentJobResponse.model_validate(res.json())

        def get_experiment_results(self, experiment_id):
            res = test_client.get(f"{self.base_url}/experiments/{experiment_id}/results")
            if res.status_code != 200:
                self._handle_error(res)
            from automl.api.schemas import ExperimentResultResponse
            return ExperimentResultResponse.model_validate(res.json())

    client = InProcessAPIClient(base_url="")

    # 1. Health check
    print("\n[Sidebar] Checking Backend Connection...")
    health = client.check_health()
    print(f"  Status:    {health.status.upper()}")
    print(f"  App:       {health.app_name} v{health.version}")
    print(f"  Capacity:  {health.active_jobs} active, {health.queued_jobs} queued, {health.completed_jobs} completed")

    # 2. View 1 & 2: Dataset Ingestion & Profiling
    print("\n[View 1 & 2] Ingesting CSV and Computing Statistical Profile...")
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    profile = client.validate_dataset(
        file_bytes=csv_bytes,
        filename="breast_cancer.csv",
        target_column="target",
    )
    print(f"  Is Valid:           {profile.is_valid}")
    print(f"  Row Count:          {profile.row_count:,}")
    print(f"  Column Count:       {profile.column_count}")
    print(f"  Detected Task:      {profile.detected_task_type}")
    print(f"  Recommended Metric: {profile.recommended_primary_metric}")
    print(f"  Numerical Features: {len(profile.numerical_columns)}")
    print(f"  Quality Warnings:   {profile.data_quality_warnings or 'None'}")

    # 3. View 3: Experiment Configuration & Submission
    print("\n[View 3] Configuring Experiment & Guarding Against Duplicate Submissions...")
    print("  Target: 'target' | Task: binary_classification | Metric: f1_macro | Folds: 3 | Test Split: 0.20")
    job = client.submit_experiment(
        file_bytes=csv_bytes,
        filename="breast_cancer.csv",
        target_column="target",
        primary_metric="f1_macro",
        cv_folds=3,
        test_size=0.20,
        random_seed=42,
        use_llm_planner=False,
    )
    print(f"  Job Submitted: ID={job.experiment_id} (Status: {job.status.value.upper()})")

    # 4. View 4: Real-time Job Progress Polling
    print("\n[View 4] Polling Execution Status (Simulating Streamlit Auto-refresh)...")
    start_time = time.time()
    while True:
        status_resp = client.get_experiment_status(job.experiment_id)
        current_status = status_resp.status.value
        elapsed = status_resp.execution_time_seconds or (time.time() - start_time)
        print(f"  Status: {current_status:<10} | Elapsed: {elapsed:.2f}s")
        if current_status in ("completed", "failed"):
            break
        time.sleep(1.0)

    # 5. View 5 & 6: Leaderboard & Candidate Details
    print("\n[View 5 & 6] Retrieving Ranked Leaderboard and Candidate Details...")
    results = client.get_experiment_results(job.experiment_id)
    print(f"  Champion Selected:  {results.best_candidate_name}")
    print(f"  Total Wall Time:    {results.total_execution_time_seconds:.2f}s")
    print(f"  Total Candidates:   {len(results.leaderboard)}")
    print("\n  Leaderboard Table:")
    print("  " + "-" * 75)
    for rank, cand in enumerate(results.leaderboard, start=1):
        champ_flag = " [CHAMPION]" if cand.is_champion else (" [BASELINE]" if cand.is_baseline else "")
        score = cand.cv_mean_metrics.get("f1_macro", 0.0)
        std = cand.cv_std_metrics.get("f1_macro", 0.0)
        print(f"  #{rank} {cand.candidate_name:<28} | F1-Macro: {score:.4f} ±{std:.4f} | Fit: {cand.fit_time_seconds:.3f}s{champ_flag}")

    # 6. View 7: Final Report (Strict Validation vs Held-Out Test Separation)
    print("\n[View 7] Final Report & Generalization Analysis:")
    print("  " + "=" * 75)
    champ = next(c for c in results.leaderboard if c.is_champion)
    cv_score = champ.cv_mean_metrics["f1_macro"]
    test_score = results.champion_test_score
    gap = test_score - cv_score if test_score is not None else 0.0

    print(f"  Champion Model:              {results.best_candidate_name}")
    print(f"  Cross-Validation F1-Macro:   {cv_score:.4f} (Evaluated on Training Partition)")
    print(f"  Held-Out Test F1-Macro:      {test_score:.4f} (Strictly Reserved Test Partition)")
    print(f"  Generalization Gap:          {gap:+.4f}")
    print(f"  MLflow Run ID:               {results.mlflow_run_id}")
    print(f"  MLflow Experiment ID:        {results.mlflow_experiment_id}")

    if results.result_explanation:
        print("\n  Grounded Result Explanation:")
        print(f"  Summary:       {results.result_explanation.get('summary', 'N/A')}")
        print(f"  Comparison:    {results.result_explanation.get('model_comparison', 'N/A')}")
        print(f"  Generalization: {results.result_explanation.get('generalization_check', 'N/A')}")

    print("\n" + "=" * 80)
    print(" INSTRUCTIONS FOR RUNNING IN SEPARATE TERMINALS")
    print("=" * 80)
    print("Terminal 1 (Backend API):")
    print("   uvicorn automl.api.app:app --host 127.0.0.1 --port 8000 --reload")
    print("\nTerminal 2 (Streamlit Dashboard):")
    print("   streamlit run src/automl/ui/app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()


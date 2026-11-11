"""Demonstration of Phase 6: FastAPI Backend Service.

Shows:
1. Health check endpoint (GET /health).
2. Dataset validation and profiling endpoint (POST /datasets/validate).
3. Asynchronous experiment job submission (POST /experiments).
4. Lifecycle status polling (GET /experiments/{experiment_id}).
5. Leaderboard and champion metrics retrieval (GET /experiments/{experiment_id}/results).
6. Instructions for starting the Uvicorn server and opening OpenAPI documentation.
"""

import io
from pathlib import Path
import sys
import time

# Add src/ to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from automl.api.app import app


def main():
    print("=" * 80)
    print(" PHASE 6 DEMO: FastAPI Backend API Service")
    print("=" * 80)

    client = TestClient(app)

    # 1. Health check
    print("\n--- 1. Checking API Health (GET /health) ---")
    health_res = client.get("/health")
    print(f"Status Code: {health_res.status_code}")
    print(f"Payload:     {health_res.json()}")

    # 2. Prepare Sample Dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    print(f"\nPrepared Dataset: {len(df)} rows, 31 columns, target='target'")

    # 3. Validate Dataset
    print("\n--- 2. Validating & Profiling Dataset (POST /datasets/validate) ---")
    val_res = client.post(
        "/datasets/validate",
        files={"file": ("breast_cancer.csv", csv_bytes, "text/csv")},
        data={"target_column": "target"},
    )
    print(f"Status Code: {val_res.status_code}")
    val_data = val_res.json()
    print(f"Is Valid:                  {val_data.get('is_valid')}")
    print(f"Detected Task:             {val_data.get('detected_task_type')}")
    print(f"Recommended Metric:        {val_data.get('recommended_primary_metric')}")
    print(f"Row Count:                 {val_data.get('row_count')}")
    print(f"Numerical Columns Count:   {len(val_data.get('numerical_columns', []))}")
    print(f"Data Quality Warnings:     {val_data.get('data_quality_warnings')}")

    # 4. Submit Asynchronous Experiment Job
    print("\n--- 3. Submitting Asynchronous Experiment (POST /experiments) ---")
    exp_res = client.post(
        "/experiments",
        files={"file": ("breast_cancer.csv", csv_bytes, "text/csv")},
        data={
            "target_column": "target",
            "primary_metric": "f1_macro",
            "cv_folds": 3,
            "test_size": 0.2,
            "random_seed": 42,
            "use_llm_planner": False,
        },
    )
    print(f"Status Code: {exp_res.status_code} (Accepted)")
    job_info = exp_res.json()
    experiment_id = job_info["experiment_id"]
    print(f"Experiment ID:  {experiment_id}")
    print(f"Initial Status: {job_info['status']}")
    print(f"Status URL:     {job_info['status_url']}")
    print(f"Results URL:    {job_info['results_url']}")

    # 5. Poll Job Status
    print("\n--- 4. Polling Experiment Status (GET /experiments/{id}) ---")
    completed = False
    for attempt in range(1, 30):
        poll_res = client.get(f"/experiments/{experiment_id}")
        status_data = poll_res.json()
        status_str = status_data["status"]
        print(f"Attempt #{attempt:02d} | Status: {status_str:<10} | Elapsed: {status_data.get('execution_time_seconds') or 0.0}s")
        if status_str in ("completed", "failed"):
            completed = True
            break
        time.sleep(1.0)

    if not completed:
        print("Polling timed out.")
        return

    # 6. Retrieve Final Results
    print("\n--- 5. Retrieving Final Results (GET /experiments/{id}/results) ---")
    results_res = client.get(f"/experiments/{experiment_id}/results")
    print(f"Status Code: {results_res.status_code}")
    results_data = results_res.json()

    print(f"\nChampion Model:       {results_data.get('best_candidate_name')}")
    print(f"Champion Test Score:  {results_data.get('champion_test_score'):.4f} ({results_data.get('primary_metric')})")
    print(f"Total Execution Time: {results_data.get('total_execution_time_seconds'):.2f}s")
    print(f"MLflow Run ID:        {results_data.get('mlflow_run_id')}")

    print("\nBenchmark Leaderboard:")
    print("-" * 80)
    for rank, cand in enumerate(results_data.get("leaderboard", []), start=1):
        champ_flag = " [CHAMPION]" if cand.get("is_champion") else ""
        cv_score = cand.get("cv_mean_metrics", {}).get(results_data["primary_metric"], 0.0)
        fit_time = cand.get("fit_time_seconds", 0.0)
        print(f"#{rank} {cand['candidate_name']:<30} | {results_data['primary_metric']}: {cv_score:.4f} | fit_time: {fit_time:.3f}s{champ_flag}")

    print("\n" + "=" * 80)
    print(" HOW TO RUN THE FASTAPI SERVER")
    print("=" * 80)
    print("To launch the live API server with automatic reload:")
    print("   uvicorn automl.api.app:app --app-dir src --host 127.0.0.1 --port 8000 --reload")
    print("Or:")
    print("   uvicorn automl.api.app:app --host 127.0.0.1 --port 8000 --reload")
    print("\nInteractive API documentation (Swagger UI):")
    print("   http://127.0.0.1:8000/docs")
    print("\nAlternative API documentation (ReDoc):")
    print("   http://127.0.0.1:8000/redoc")
    print("=" * 80)


if __name__ == "__main__":
    main()


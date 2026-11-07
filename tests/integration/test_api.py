"""Comprehensive API integration tests for FastAPI backend endpoints."""

import io
import time
from typing import Generator
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from automl.api.app import create_app
from automl.api.jobs import JobManager, JobRecord, JobStatus
from automl.api.schemas import HealthResponse
from automl.core.config import Settings


@pytest.fixture
def api_client(tmp_path) -> Generator[TestClient, None, None]:
    """Provide an isolated FastAPI TestClient with temporary storage."""
    test_settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        max_concurrent_jobs=2,
    )
    test_settings.ensure_directories()
    app = create_app(settings=test_settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def csv_bytes(sample_binary_classification_df: pd.DataFrame) -> bytes:
    """Provide binary classification CSV bytes."""
    buf = io.StringIO()
    sample_binary_classification_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_health_endpoint(api_client: TestClient):
    """Verify GET /health returns service status and job capacity."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "active_jobs" in data
    assert "queued_jobs" in data


def test_validate_dataset_success(api_client: TestClient, csv_bytes: bytes):
    """Verify POST /datasets/validate successfully profiles valid CSV."""
    response = api_client.post(
        "/datasets/validate",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
        data={"target_column": "churn"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["row_count"] == 200
    assert data["target_column"] == "churn"
    assert data["detected_task_type"] == "binary_classification"
    assert data["recommended_primary_metric"] == "f1_macro"
    assert "age" in data["column_types"]
    assert len(data["data_quality_warnings"]) >= 0


def test_validate_dataset_empty_file(api_client: TestClient):
    """Verify POST /datasets/validate rejects empty file with HTTP 400."""
    response = api_client.post(
        "/datasets/validate",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_validate_dataset_invalid_extension(api_client: TestClient):
    """Verify POST /datasets/validate rejects non-CSV files with HTTP 400."""
    response = api_client.post(
        "/datasets/validate",
        files={"file": ("data.txt", b"a,b,c\n1,2,3", "text/plain")},
    )
    assert response.status_code == 400
    assert "unsupported file format" in response.json()["detail"].lower()


def test_validate_dataset_duplicate_columns(api_client: TestClient):
    """Verify POST /datasets/validate rejects duplicate column headers with HTTP 422."""
    bad_csv = b"colA,colB,colA\n1,2,3\n4,5,6"
    response = api_client.post(
        "/datasets/validate",
        files={"file": ("dups.csv", bad_csv, "text/csv")},
    )
    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"].lower()


def test_validate_dataset_invalid_task_override(api_client: TestClient, csv_bytes: bytes):
    """Verify POST /datasets/validate rejects invalid task override with HTTP 422."""
    response = api_client.post(
        "/datasets/validate",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
        data={"target_column": "churn", "task_type_override": "quantum_classification"},
    )
    assert response.status_code == 422
    assert "invalid task_type_override" in response.json()["detail"].lower()


def test_get_unknown_experiment_id(api_client: TestClient):
    """Verify 404 when querying unknown experiment ID."""
    unknown_id = "00000000-0000-0000-0000-000000000000"
    res_status = api_client.get(f"/experiments/{unknown_id}")
    assert res_status.status_code == 404
    assert "not found" in res_status.json()["detail"].lower()

    res_results = api_client.get(f"/experiments/{unknown_id}/results")
    assert res_results.status_code == 404
    assert "not found" in res_results.json()["detail"].lower()


def test_get_results_when_job_not_completed(api_client: TestClient):
    """Verify 409 Conflict when retrieving results for incomplete job."""
    from automl.api.jobs import get_job_manager

    manager = get_job_manager()
    dummy_job = JobRecord(
        experiment_id="test-running-job-123",
        status=JobStatus.RUNNING,
        target_column="churn",
        created_at="2026-11-06T00:00:00Z",
    )
    with manager._lock:
        manager._jobs[dummy_job.experiment_id] = dummy_job

    response = api_client.get("/experiments/test-running-job-123/results")
    assert response.status_code == 409
    assert "currently in state 'running'" in response.json()["detail"]


def test_experiment_submission_missing_target(api_client: TestClient, csv_bytes: bytes):
    """Verify 422 when target column does not exist in dataset."""
    response = api_client.post(
        "/experiments",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
        data={"target_column": "nonexistent_target_col"},
    )
    assert response.status_code == 422
    assert "does not exist in dataset" in response.json()["detail"]


def test_experiment_lifecycle_end_to_end(api_client: TestClient, csv_bytes: bytes):
    """Verify full asynchronous experiment lifecycle: submit -> poll -> get results."""
    # 1. Submit experiment
    submit_res = api_client.post(
        "/experiments",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
        data={
            "target_column": "churn",
            "cv_folds": 2,
            "test_size": 0.2,
            "random_seed": 42,
            "use_llm_planner": False,
        },
    )
    assert submit_res.status_code == 202
    job_info = submit_res.json()
    experiment_id = job_info["experiment_id"]
    assert job_info["status"] in ("queued", "running")
    assert "status_url" in job_info
    assert "results_url" in job_info

    # 2. Poll until completed (timeout after 60s)
    max_wait = 60
    start = time.time()
    final_status = None
    while time.time() - start < max_wait:
        poll_res = api_client.get(f"/experiments/{experiment_id}")
        assert poll_res.status_code == 200
        status_data = poll_res.json()
        final_status = status_data["status"]
        if final_status in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert final_status == "completed"

    # 3. Retrieve final benchmark results
    results_res = api_client.get(f"/experiments/{experiment_id}/results")
    assert results_res.status_code == 200
    results_data = results_res.json()

    assert results_data["experiment_id"] == experiment_id
    assert results_data["target_column"] == "churn"
    assert results_data["task_type"] == "binary_classification"
    assert results_data["best_candidate_name"] is not None
    assert results_data["champion_test_score"] is not None
    assert len(results_data["leaderboard"]) > 0
    assert results_data["train_rows"] == 160
    assert results_data["test_rows"] == 40
    assert results_data["mlflow_run_id"] is not None


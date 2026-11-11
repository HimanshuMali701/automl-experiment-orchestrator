"""UI client unit tests and Streamlit AppTest smoke tests."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from automl.api.schemas import (
    CandidateSummaryResponse,
    DatasetValidationResponse,
    ExperimentJobResponse,
    ExperimentResultResponse,
    HealthResponse,
    JobStatus,
)
from automl.ui.client import APIClient, APIClientError, APIConnectionError

APP_FILE = str(Path(__file__).resolve().parents[2] / "src" / "automl" / "ui" / "app.py")


# --- APIClient Unit Tests ---

def test_api_client_health_success(monkeypatch):
    """Verify check_health parses successful HealthResponse."""
    client = APIClient(base_url="http://testserver")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "app_name": "AutoML Experiment Orchestrator",
        "version": "0.1.0",
        "active_jobs": 0,
        "queued_jobs": 0,
        "completed_jobs": 2,
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.check_health()
        assert isinstance(res, HealthResponse)
        assert res.status == "healthy"
        assert res.version == "0.1.0"


def test_api_client_connection_error():
    """Verify APIConnectionError is raised when backend is unreachable."""
    client = APIClient(base_url="http://127.0.0.1:9999")  # Unused port
    with pytest.raises(APIConnectionError) as exc_info:
        client.check_health()
    assert "Cannot connect to AutoML API backend" in str(exc_info.value)


def test_api_client_validate_dataset_success():
    """Verify validate_dataset parses DatasetValidationResponse."""
    client = APIClient(base_url="http://testserver")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "is_valid": True,
        "row_count": 100,
        "column_count": 4,
        "target_column": "target",
        "detected_task_type": "binary_classification",
        "recommended_primary_metric": "f1_macro",
        "data_quality_warnings": ["Minor missingness"],
        "column_types": {"feat1": "numeric", "feat2": "categorical", "target": "categorical"},
        "missing_value_counts": {"feat1": 0, "feat2": 5, "target": 0},
        "numerical_columns": ["feat1"],
        "categorical_columns": ["feat2", "target"],
        "duplicate_rows": 0,
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.validate_dataset(
            file_bytes=b"a,b,target\n1,x,0\n2,y,1",
            filename="data.csv",
            target_column="target",
        )
        assert isinstance(res, DatasetValidationResponse)
        assert res.row_count == 100
        assert res.detected_task_type == "binary_classification"


def test_api_client_submit_experiment_error():
    """Verify APIClientError is raised on 422/429 errors with message."""
    client = APIClient(base_url="http://testserver")

    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": "Target column 'foo' not found.", "error_type": "DatasetValidationError"}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(APIClientError) as exc_info:
            client.submit_experiment(
                file_bytes=b"a,b\n1,2",
                filename="data.csv",
                target_column="foo",
            )
        assert "Target column 'foo' not found." in str(exc_info.value)
        assert exc_info.value.status_code == 422


# --- Streamlit AppTest Smoke Tests ---

def test_streamlit_app_initial_render():
    """Smoke test: verify Streamlit dashboard initializes without uncaught exceptions."""
    at = AppTest.from_file(APP_FILE, default_timeout=15.0)
    at.run()
    assert not at.exception
    # Verify title rendered
    assert len(at.title) >= 1
    assert "AutoML Experiment Orchestrator" in at.title[0].value
    # Verify tabs rendered
    assert len(at.tabs) == 5


def test_streamlit_app_with_validated_dataset():
    """Verify dataset profile view renders when validation_result is in session state."""
    at = AppTest.from_file(APP_FILE, default_timeout=15.0)

    val_res = DatasetValidationResponse(
        is_valid=True,
        row_count=250,
        column_count=5,
        target_column="target",
        detected_task_type="binary_classification",
        recommended_primary_metric="f1_macro",
        data_quality_warnings=["High cardinality in feature x"],
        column_types={"age": "numeric", "country": "categorical", "target": "categorical"},
        missing_value_counts={"age": 2, "country": 0, "target": 0},
        numerical_columns=["age"],
        categorical_columns=["country", "target"],
        duplicate_rows=1,
    )

    at.session_state["validation_result"] = val_res
    at.session_state["uploaded_file_bytes"] = b"age,country,target\n25,US,1\n30,UK,0"
    at.session_state["uploaded_file_name"] = "test_data.csv"
    at.session_state["preview_df"] = pd.DataFrame({"age": [25, 30], "country": ["US", "UK"], "target": [1, 0]})

    at.run()
    assert not at.exception
    # Profile metrics should be rendered in UI
    metric_values = [m.value for m in at.metric]
    assert "250" in metric_values or "250" in [str(m.value) for m in at.metric]


def test_streamlit_app_with_completed_results():
    """Verify leaderboard and final report views render when experiment_results is in session state."""
    at = AppTest.from_file(APP_FILE, default_timeout=15.0)

    cand = CandidateSummaryResponse(
        candidate_name="Random Forest Classifier",
        model_family="random_forest_classifier",
        is_baseline=False,
        is_champion=True,
        status="success",
        cv_mean_metrics={"f1_macro": 0.8850, "accuracy": 0.8900},
        cv_std_metrics={"f1_macro": 0.0210, "accuracy": 0.0150},
        test_metrics={"f1_macro": 0.8710, "accuracy": 0.8800},
        fit_time_seconds=0.35,
    )

    results = ExperimentResultResponse(
        experiment_id="exp-1234-abcd",
        status=JobStatus.COMPLETED,
        target_column="target",
        task_type="binary_classification",
        primary_metric="f1_macro",
        best_candidate_name="Random Forest Classifier",
        champion_test_score=0.8710,
        train_rows=160,
        test_rows=40,
        total_execution_time_seconds=4.25,
        mlflow_run_id="run-mlflow-5678",
        mlflow_experiment_id="1",
        leaderboard=[cand],
        result_explanation={
            "summary": "Random Forest Classifier outperformed baseline.",
            "model_comparison": "Achieved +0.35 gain over dummy classifier.",
            "generalization_check": "Test score is within 0.014 of validation.",
            "caveats": [],
        },
        warnings=[],
    )

    at.session_state["experiment_results"] = results
    at.session_state["active_experiment_id"] = "exp-1234-abcd"

    at.run()
    assert not at.exception
    # Champion name should appear in report
    champion_mentions = [s.value for s in at.success]
    assert any("Random Forest Classifier" in val for val in champion_mentions)

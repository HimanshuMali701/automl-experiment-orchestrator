"""HTTP client wrapping FastAPI backend endpoints for the Streamlit UI."""

import io
from typing import Any, Dict, Optional, Tuple
import httpx
from pydantic import ValidationError

from automl.api.schemas import (
    DatasetValidationResponse,
    ErrorResponse,
    ExperimentJobResponse,
    ExperimentResultResponse,
    HealthResponse,
    JobStatus,
)


class APIClientError(Exception):
    """Base exception for API client errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class APIConnectionError(APIClientError):
    """Raised when the backend API server is unreachable."""
    pass


class APIClient:
    """Robust synchronous client communicating with the FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    def _handle_response_error(self, response: httpx.Response) -> None:
        """Parse structured API error or raise formatted APIClientError."""
        status_code = response.status_code
        try:
            payload = response.json()
            detail = payload.get("detail", response.text)
            err_type = payload.get("error_type", "HTTPError")
        except Exception:
            detail = response.text or f"HTTP {status_code} Error"
            err_type = "HTTPError"

        raise APIClientError(
            message=f"{detail}",
            status_code=status_code,
            details={"error_type": err_type, "raw": detail},
        )

    def check_health(self) -> HealthResponse:
        """Call GET /health to verify backend connectivity and job capacity."""
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(url)
                if res.status_code != 200:
                    self._handle_response_error(res)
                return HealthResponse.model_validate(res.json())
        except httpx.RequestError as exc:
            raise APIConnectionError(
                f"Cannot connect to AutoML API backend at '{self.base_url}'. "
                f"Please verify the backend server is running (e.g. 'uvicorn automl.api.app:app --host 127.0.0.1 --port 8000')."
            ) from exc

    def validate_dataset(
        self,
        file_bytes: bytes,
        filename: str,
        target_column: Optional[str] = None,
        task_type_override: Optional[str] = None,
    ) -> DatasetValidationResponse:
        """Call POST /datasets/validate to non-destructively profile uploaded CSV."""
        url = f"{self.base_url}/datasets/validate"
        files = {"file": (filename, file_bytes, "text/csv")}
        data: Dict[str, Any] = {}
        if target_column:
            data["target_column"] = target_column
        if task_type_override:
            data["task_type_override"] = task_type_override

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, files=files, data=data)
                if res.status_code != 200:
                    self._handle_response_error(res)
                return DatasetValidationResponse.model_validate(res.json())
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Connection failed while validating dataset: {str(exc)}") from exc

    def submit_experiment(
        self,
        file_bytes: bytes,
        filename: str,
        target_column: str,
        primary_metric: Optional[str] = None,
        task_type_override: Optional[str] = None,
        cv_folds: int = 5,
        test_size: float = 0.20,
        random_seed: int = 42,
        use_llm_planner: bool = False,
    ) -> ExperimentJobResponse:
        """Call POST /experiments to enqueue an asynchronous AutoML job."""
        url = f"{self.base_url}/experiments"
        files = {"file": (filename, file_bytes, "text/csv")}
        data: Dict[str, Any] = {
            "target_column": target_column,
            "cv_folds": cv_folds,
            "test_size": test_size,
            "random_seed": random_seed,
            "use_llm_planner": use_llm_planner,
        }
        if primary_metric:
            data["primary_metric"] = primary_metric
        if task_type_override:
            data["task_type_override"] = task_type_override

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, files=files, data=data)
                if res.status_code != 202:
                    self._handle_response_error(res)
                return ExperimentJobResponse.model_validate(res.json())
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Connection failed while submitting experiment: {str(exc)}") from exc

    def get_experiment_status(self, experiment_id: str) -> ExperimentJobResponse:
        """Call GET /experiments/{experiment_id} to query current job execution state."""
        url = f"{self.base_url}/experiments/{experiment_id}"
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.get(url)
                if res.status_code != 200:
                    self._handle_response_error(res)
                return ExperimentJobResponse.model_validate(res.json())
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Connection failed while polling status for job {experiment_id}: {str(exc)}") from exc

    def get_experiment_results(self, experiment_id: str) -> ExperimentResultResponse:
        """Call GET /experiments/{experiment_id}/results to retrieve leaderboard and metrics."""
        url = f"{self.base_url}/experiments/{experiment_id}/results"
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.get(url)
                if res.status_code != 200:
                    self._handle_response_error(res)
                return ExperimentResultResponse.model_validate(res.json())
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Connection failed while retrieving results for job {experiment_id}: {str(exc)}") from exc


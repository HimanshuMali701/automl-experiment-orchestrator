"""Session state management and default initializations for Streamlit UI."""

import os
from typing import Any, Dict, Optional
import streamlit as st

from automl.api.schemas import (
    DatasetValidationResponse,
    ExperimentJobResponse,
    ExperimentResultResponse,
    HealthResponse,
)
from automl.ui.client import APIClient


DEFAULT_API_BASE_URL = os.getenv("AUTOML_API_BASE_URL", "http://127.0.0.1:8000")


def init_session_state() -> None:
    """Initialize all Streamlit session state variables defensively on app load."""
    defaults: Dict[str, Any] = {
        # Backend connectivity
        "api_base_url": DEFAULT_API_BASE_URL,
        "backend_health": None,  # Optional[HealthResponse]
        "backend_connected": False,
        "last_health_check_error": None,

        # Dataset & Profiling state
        "uploaded_file_bytes": None,  # bytes
        "uploaded_file_name": None,   # str
        "preview_df": None,           # pd.DataFrame
        "validation_result": None,    # Optional[DatasetValidationResponse]
        "selected_target": None,      # Optional[str]
        "validation_error": None,     # Optional[str]
        "is_validating": False,       # bool

        # Experiment Configuration & Execution state
        "active_experiment_id": None,      # Optional[str]
        "job_status": None,                # Optional[ExperimentJobResponse]
        "experiment_results": None,        # Optional[ExperimentResultResponse]
        "is_submitting": False,            # bool (idempotency guard against duplicate clicks)
        "submit_error": None,              # Optional[str]
        "polling_error": None,             # Optional[str]

        # Active view navigation
        "active_tab_index": 0,
    }

    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def get_api_client() -> APIClient:
    """Return an APIClient configured with the current session's base URL."""
    base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    return APIClient(base_url=base_url)


def reset_dataset_state() -> None:
    """Clear uploaded dataset and profiling state when a new file is uploaded."""
    st.session_state["uploaded_file_bytes"] = None
    st.session_state["uploaded_file_name"] = None
    st.session_state["preview_df"] = None
    st.session_state["validation_result"] = None
    st.session_state["selected_target"] = None
    st.session_state["validation_error"] = None
    st.session_state["is_validating"] = False


def reset_experiment_state() -> None:
    """Clear active experiment progress, job status, and leaderboard results."""
    st.session_state["active_experiment_id"] = None
    st.session_state["job_status"] = None
    st.session_state["experiment_results"] = None
    st.session_state["is_submitting"] = False
    st.session_state["submit_error"] = None
    st.session_state["polling_error"] = None


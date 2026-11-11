"""Experiment configuration and submission component."""

from typing import Optional
import streamlit as st

from automl.api.schemas import DatasetValidationResponse
from automl.ui.client import APIClientError
from automl.ui.state import get_api_client, reset_experiment_state


# Metric allowlists per task
CLASSIFICATION_METRICS = ["f1_macro", "accuracy", "precision_macro", "recall_macro", "roc_auc_ovr"]
REGRESSION_METRICS = ["rmse", "mae", "r2"]


def render_experiment_config_view() -> None:
    """Render View 3: Experiment Configuration Form and Submission Guard."""
    st.header("3. Configure & Launch Experiment", divider="gray")

    val_res: Optional[DatasetValidationResponse] = st.session_state.get("validation_result")
    file_bytes = st.session_state.get("uploaded_file_bytes")
    file_name = st.session_state.get("uploaded_file_name")

    if not val_res or not file_bytes:
        st.warning("⚠️ Please validate a dataset in Step 1 before configuring experiments.")
        return

    detected_task = val_res.detected_task_type or "binary_classification"
    is_regression = detected_task == "regression"
    available_metrics = REGRESSION_METRICS if is_regression else CLASSIFICATION_METRICS

    default_metric = val_res.recommended_primary_metric
    if default_metric not in available_metrics:
        default_metric = available_metrics[0]

    with st.form("experiment_submission_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.text_input("Target Column", value=val_res.target_column, disabled=True)
            task_type = st.selectbox(
                "Confirmed Task Type",
                options=["binary_classification", "multiclass_classification", "regression"],
                index=["binary_classification", "multiclass_classification", "regression"].index(detected_task),
                help="Prediction problem type.",
            )
            primary_metric = st.selectbox(
                "Primary Evaluation Metric",
                options=available_metrics,
                index=available_metrics.index(default_metric),
                help="Metric used to rank candidates and declare the champion model.",
            )

        with col2:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of k-fold splits used for validation performance estimation.",
            )
            test_size = st.slider(
                "Held-Out Test Set Split Ratio",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                help="Strictly held-out partition used ONLY for final unbiased evaluation of the champion model.",
            )
            random_seed = st.number_input(
                "Random Seed",
                value=42,
                min_value=0,
                max_value=999999,
                help="Seed for reproducible splits and model parameter initialization.",
            )

        st.subheader("🤖 Plan Generation Mode")
        use_llm = st.checkbox(
            "Use Groq LLM Plan Proposer",
            value=False,
            help="If enabled, queries Groq ChatGroq to propose models and preprocessing choices based on the dataset profile. Falls back to deterministic plans on error.",
        )

        st.info(
            "🔒 **Safety & Resource Guardrails:** Arbitrary code execution is strictly prohibited. "
            "Maximum 10 model candidates, bounded training timeout (600s), and strict leakage guards are enforced.",
            icon="🛡️"
        )

        # Duplicate submission guard
        is_submitting = st.session_state.get("is_submitting", False)
        submit_button = st.form_submit_button(
            "🚀 Submit Experiment to Background Queue",
            type="primary",
            use_container_width=True,
            disabled=is_submitting or not st.session_state.get("backend_connected", False),
        )

    if submit_button:
        _handle_submission(
            file_bytes=file_bytes,
            filename=file_name,
            target_column=val_res.target_column,
            primary_metric=primary_metric,
            task_type=task_type,
            cv_folds=cv_folds,
            test_size=test_size,
            random_seed=random_seed,
            use_llm_planner=use_llm,
        )


def _handle_submission(
    file_bytes: bytes,
    filename: str,
    target_column: str,
    primary_metric: str,
    task_type: str,
    cv_folds: int,
    test_size: float,
    random_seed: int,
    use_llm_planner: bool,
) -> None:
    """Submit experiment and lock session state against duplicate clicks."""
    st.session_state["is_submitting"] = True
    reset_experiment_state()
    client = get_api_client()

    with st.spinner("Submitting experiment job to FastAPI background worker pool..."):
        try:
            job_resp = client.submit_experiment(
                file_bytes=file_bytes,
                filename=filename,
                target_column=target_column,
                primary_metric=primary_metric,
                task_type_override=task_type,
                cv_folds=cv_folds,
                test_size=test_size,
                random_seed=random_seed,
                use_llm_planner=use_llm_planner,
            )
            st.session_state["active_experiment_id"] = job_resp.experiment_id
            st.session_state["job_status"] = job_resp
            st.session_state["submit_error"] = None
            st.session_state["is_submitting"] = False
            st.toast(f"Experiment {job_resp.experiment_id[:8]} queued successfully!", icon="🚀")
            st.rerun()

        except APIClientError as exc:
            st.session_state["is_submitting"] = False
            st.session_state["submit_error"] = exc.message
            st.error(f"Failed to submit experiment: {exc.message}")
        except Exception as exc:
            st.session_state["is_submitting"] = False
            st.session_state["submit_error"] = str(exc)
            st.error(f"Unexpected error during submission: {str(exc)}")


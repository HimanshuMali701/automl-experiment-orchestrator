"""Header and backend connection status component."""

import streamlit as st

from automl.ui.client import APIConnectionError
from automl.ui.state import get_api_client


def render_header() -> None:
    """Render top page header and sidebar connection controls."""
    st.title("⚡ AutoML Experiment Orchestrator")
    st.caption("Production-minded automated machine learning experimentation, leakage-free benchmarking & MLflow tracking.")

    # Sidebar: Backend Connection & Server Telemetry
    with st.sidebar:
        st.header("🔌 Backend Connectivity")
        api_url = st.text_input(
            "FastAPI Base URL",
            value=st.session_state.get("api_base_url", "https://automl-backend-trtp.onrender.com"),
            help="Base address of the running FastAPI backend server.",
        )
        if api_url != st.session_state.get("api_base_url"):
            st.session_state["api_base_url"] = api_url.rstrip("/")
            st.rerun()

        # Check health
        client = get_api_client()
        try:
            health = client.check_health()
            st.session_state["backend_health"] = health
            st.session_state["backend_connected"] = True
            st.session_state["last_health_check_error"] = None

            st.success(f"Connected: v{health.version}", icon="🟢")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Jobs", health.active_jobs)
            with col2:
                st.metric("Queued", health.queued_jobs)

        except APIConnectionError as exc:
            st.session_state["backend_health"] = None
            st.session_state["backend_connected"] = False
            st.session_state["last_health_check_error"] = str(exc)
            st.error("Backend Offline", icon="🔴")
            st.caption(
                "Verify your backend service at:\n\n"
                "`https://automl-backend-trtp.onrender.com`\n\n"
                "Or launch local backend in terminal:\n\n"
                "`uvicorn automl.api.app:app --host 127.0.0.1 --port 8000`"
            )

        st.divider()
        st.subheader("🔗 Documentation & Tracking")
        docs_url = f"{st.session_state.get('api_base_url')}/docs"
        redoc_url = f"{st.session_state.get('api_base_url')}/redoc"
        st.markdown(f"- [OpenAPI Swagger Docs]({docs_url})")
        st.markdown(f"- [ReDoc Specification]({redoc_url})")
        st.markdown("- [MLflow UI (Local)](http://127.0.0.1:5000)")

        st.divider()
        if st.button("🔄 Reset UI State", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


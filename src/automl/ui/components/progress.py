"""Real-time experiment job progress and polling component powered by LangGraph state."""

import time
from typing import Any, Dict, List, Optional
import streamlit as st

from automl.api.schemas import ExperimentJobResponse, JobStatus
from automl.ui.client import APIClientError
from automl.ui.state import get_api_client


def render_progress_view() -> None:
    """Render View 4: Real-time Job Execution State & Event-Driven Progress."""
    st.header("4. Experiment Execution Progress", divider="gray")

    active_id: Optional[str] = st.session_state.get("active_experiment_id")

    if not active_id:
        st.info("ℹ️ No active experiment submitted yet. Configure and launch one in the 'Configuration' tab.", icon="ℹ️")
        with st.expander("Or look up existing Experiment ID"):
            lookup_id = st.text_input("Experiment UUID")
            if st.button("Load Experiment") and lookup_id.strip():
                st.session_state["active_experiment_id"] = lookup_id.strip()
                st.rerun()
        return

    client = get_api_client()

    # Query latest status
    try:
        latest_job = client.get_experiment_status(active_id)
        st.session_state["job_status"] = latest_job
        st.session_state["polling_error"] = None
    except APIClientError as exc:
        st.session_state["polling_error"] = exc.message
        st.error(f"Error fetching status for `{active_id}`: {exc.message}")
        return

    job: ExperimentJobResponse = st.session_state["job_status"]

    # Header Card with Status Badge
    col_id, col_stat, col_time = st.columns([3, 2, 2])
    with col_id:
        st.subheader(f"Job: `{job.experiment_id[:8]}...`")
        st.caption(f"Full UUID: `{job.experiment_id}`")
    with col_stat:
        _render_status_badge(job.status)
    with col_time:
        duration = job.execution_time_seconds or 0.0
        st.metric("Elapsed Time", f"{duration:.2f}s")

    # Real Graph Execution Stage & Node Indicators (No fake progress bars)
    st.markdown("#### Real-Time LangGraph Workflow State")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"**Current Stage:** `{job.current_stage or 'Queued'}`")
    with s2:
        st.markdown(f"**Active Node:** `{job.current_node or 'None'}`")
    with s3:
        st.markdown(f"**Completed Steps:** `{len(job.completed_nodes)}`")

    # Completed Nodes Chips
    if job.completed_nodes:
        chips = " ".join([f"`✓ {node}`" for node in job.completed_nodes])
        st.markdown(f"**Executed Nodes:** {chips}")

    # Metadata Details
    m1, m2, m3 = st.columns(3)
    with m1:
        st.write(f"**Target Column:** `{job.target_column}`")
    with m2:
        st.write(f"**Task Type:** `{job.task_type or 'Detecting...'}`")
    with m3:
        st.write(f"**Queued At:** `{job.created_at[:19]}`")

    st.divider()

    # Fetch latest event if running or waiting
    latest_event_summary: Optional[str] = None
    if job.status in (JobStatus.RUNNING, JobStatus.WAITING_FOR_APPROVAL, JobStatus.QUEUED):
        try:
            events = client.get_workflow_events(job.experiment_id)
            if events:
                latest = events[-1]
                latest_event_summary = f"**[{latest.get('timestamp', '')[11:19]}]** {latest.get('summary', '')}"
        except Exception:
            pass

    if latest_event_summary:
        st.markdown(f"📌 **Latest Event:** {latest_event_summary}")

    # Progress States (No fabricated percentages)
    if job.status == JobStatus.QUEUED:
        st.info("⏳ **Experiment is queued.** Waiting for an available worker thread...", icon="⏳")
        st.caption("⚡ Polling backend queue status every 2 seconds...")
        _render_auto_refresh()

    elif job.status == JobStatus.RUNNING:
        st.info(
            f"⚙️ **Graph actively running at node `{job.current_node or 'starting'}`.** "
            f"Stage: *{job.current_stage or 'Processing'}*",
            icon="⚙️",
        )
        st.caption("⚡ Polling real-time graph events every 2 seconds...")
        _render_auto_refresh()

    elif job.status == JobStatus.WAITING_FOR_APPROVAL:
        st.warning(
            "⏸️ **Human Approval Required:** Experiment plan has been generated and validated. "
            "Execution is currently paused at `approval_gate` waiting for human sign-off.",
            icon="⏸️",
        )
        st.info(
            "👉 Please navigate to the **'🕵️ 6. Agent Workflow Inspector'** tab or use the API endpoint "
            f"`{job.approval_url or '/experiments/' + job.experiment_id + '/approval'}` to review and approve/reject the proposed plan.",
            icon="ℹ️",
        )
        _render_auto_refresh()

    elif job.status == JobStatus.REJECTED:
        st.warning(
            f"🚫 **Experiment Plan Rejected:** {job.error_message or 'The plan was rejected by the reviewer.'}",
            icon="🚫",
        )
        st.caption("No models were trained. You can adjust your configuration and launch a new experiment.")

    elif job.status == JobStatus.COMPLETED:
        st.success("🎉 **Experiment completed successfully!** All candidates evaluated and ranked.", icon="✅")

        # Fetch results automatically if not already populated
        if st.session_state.get("experiment_results") is None:
            with st.spinner("Fetching final benchmark leaderboard and metrics..."):
                try:
                    results = client.get_experiment_results(job.experiment_id)
                    st.session_state["experiment_results"] = results
                    st.toast("Benchmark results loaded!", icon="🏆")
                except APIClientError as exc:
                    st.error(f"Could not load experiment results: {exc.message}")

        st.caption("👉 Head over to the **'Model Leaderboard'** or **'Final Report'** tabs to explore the models.")

    elif job.status == JobStatus.FAILED:
        st.error("❌ **Experiment execution failed.**", icon="🚨")
        st.error(f"**Error Details:** {job.error_message or 'Workflow failed without explicit message'}")
        st.caption("Check the **Agent Workflow Inspector** tab for the failure diagnosis, root cause, and recovery attempts.")


def _render_status_badge(status: JobStatus) -> None:
    """Render a colored status badge."""
    if status == JobStatus.QUEUED:
        st.warning("QUEUED", icon="⏳")
    elif status == JobStatus.RUNNING:
        st.info("RUNNING", icon="⚙️")
    elif status == JobStatus.WAITING_FOR_APPROVAL:
        st.warning("WAITING APPROVAL", icon="⏸️")
    elif status == JobStatus.COMPLETED:
        st.success("COMPLETED", icon="✅")
    elif status == JobStatus.REJECTED:
        st.warning("REJECTED", icon="🚫")
    elif status == JobStatus.FAILED:
        st.error("FAILED", icon="❌")


def _render_auto_refresh() -> None:
    """Render auto-refresh controls and trigger rerun if polling."""
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Check Status"):
            st.rerun()
    with col2:
        auto_poll = st.checkbox("Auto-refresh every 2 seconds", value=True)

    if auto_poll:
        time.sleep(2.0)
        st.rerun()

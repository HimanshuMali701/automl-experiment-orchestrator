"""Agent Workflow Inspector component exposing granular LangGraph execution telemetry."""

import json
import time
from typing import Any, Dict, List, Optional
import streamlit as st

from automl.api.schemas import JobStatus
from automl.ui.client import APIClientError
from automl.ui.state import get_api_client
from automl.workflow.events import ALL_WORKFLOW_NODES, NODE_TO_STAGE


def render_workflow_inspector_view() -> None:
    """Render View 6: Agent Workflow Inspector."""
    st.header("🕵️ Agent Workflow Inspector", divider="gray")
    st.caption("Live, fine-grained telemetry into LangGraph state transitions, routing decisions, LLM reflections, and failure recoveries.")

    active_id: Optional[str] = st.session_state.get("active_experiment_id")

    # Experiment lookup or selection
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        lookup_input = st.text_input(
            "Experiment ID",
            value=active_id or "",
            placeholder="Enter experiment UUID...",
            help="View execution graph and events for any active or past experiment.",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Inspect Experiment") and lookup_input.strip():
            st.session_state["active_experiment_id"] = lookup_input.strip()
            st.rerun()

    exp_id = lookup_input.strip() or active_id
    if not exp_id:
        st.info("ℹ️ Select or launch an experiment to inspect its LangGraph execution events.", icon="ℹ️")
        return

    client = get_api_client()

    # Query inspector data and job status
    try:
        inspector_data = client.get_workflow_inspector(exp_id)
        job_status = client.get_experiment_status(exp_id)
    except APIClientError as exc:
        st.error(f"Failed to fetch inspector telemetry for `{exp_id}`: {exc.message}")
        return

    # Polling & Mode Banner
    col_info, col_refresh = st.columns([4, 1])
    with col_info:
        st.info(
            f"⚡ **Viewing Persisted Graph Telemetry** — Experiment `{exp_id}` | "
            f"Status: **{job_status.status.upper()}** | Stage: **{job_status.current_stage or 'N/A'}**",
            icon="🛰️",
        )
    with col_refresh:
        auto_poll = st.checkbox("Auto-poll (2s)", value=(job_status.status in (JobStatus.RUNNING, JobStatus.WAITING_FOR_APPROVAL)))
        if st.button("🔄 Refresh"):
            st.rerun()

    # 1. Summary Metrics Card
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current Stage", job_status.current_stage or "Completed")
    with c2:
        st.metric("Active Node", job_status.current_node or "None (Idle)")
    with c3:
        st.metric("Completed Nodes", f"{len(job_status.completed_nodes)} / {len(ALL_WORKFLOW_NODES)}")
    with c4:
        total_dur = job_status.execution_time_seconds or 0.0
        st.metric("Total Execution Time", f"{total_dur:.2f}s")

    st.divider()

    # 2. Interactive Human Approval Section (if waiting for approval)
    if job_status.status == JobStatus.WAITING_FOR_APPROVAL or job_status.pending_approval:
        _render_approval_panel(client, exp_id)
        st.divider()

    # 3. Node Execution Timeline
    st.subheader("1. Workflow Node Timeline")
    st.caption("Real node execution durations, start/end timestamps, and terminal statuses.")
    timeline_items = inspector_data.get("timeline", [])
    timeline_map = {item["node"]: item for item in timeline_items}

    # Render structured timeline table
    timeline_table = []
    for node_name in ALL_WORKFLOW_NODES:
        stage = NODE_TO_STAGE.get(node_name, node_name)
        if node_name in timeline_map:
            rec = timeline_map[node_name]
            dur = f"{rec['duration_seconds']:.2f}s" if rec.get("duration_seconds") is not None else "-"
            st_time = rec.get("start_time", "")[11:19] if rec.get("start_time") else "-"
            ed_time = rec.get("end_time", "")[11:19] if rec.get("end_time") else "-"
            status_str = rec.get("status", "pending").upper()
            summary = rec.get("summary") or "-"
        else:
            dur = "-"
            st_time = "-"
            ed_time = "-"
            status_str = "SKIPPED / PENDING"
            summary = "Not visited on this execution path"

        # Badge prefix
        if status_str == "COMPLETED":
            badge = "🟢 COMPLETED"
        elif status_str == "RUNNING":
            badge = "🔵 RUNNING"
        elif status_str == "WAITING_FOR_APPROVAL":
            badge = "🟡 WAITING APPROVAL"
        elif status_str == "FAILED":
            badge = "🔴 FAILED"
        else:
            badge = "⚪ SKIPPED"

        timeline_table.append({
            "Node": f"`{node_name}`",
            "Stage": stage,
            "Status": badge,
            "Duration": dur,
            "Start (UTC)": st_time,
            "End (UTC)": ed_time,
            "Summary": summary,
        })

    st.dataframe(
        timeline_table,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # 4. Actual Routing Decisions
    st.subheader("2. Evaluated Routing Decisions")
    st.caption("Every conditional branch evaluated by LangGraph with the exact operational reason.")
    routing_decisions = inspector_data.get("routing_decisions", [])
    if routing_decisions:
        for idx, rd in enumerate(routing_decisions):
            col_edge, col_reason = st.columns([2, 5])
            with col_edge:
                st.markdown(f"**Step {idx + 1}:** `{rd.get('from_node')}` ➔ `{rd.get('to_node')}`")
                st.caption(f"Timestamp: `{rd.get('timestamp', '')[11:19]}`")
            with col_reason:
                st.info(f"💡 **Routing Reason:** {rd.get('reason')}")
    else:
        st.write("No conditional routing decisions recorded yet.")

    st.divider()

    # 5. LLM Involvement & Reflection History
    st.subheader("3. LLM Plan Proposals & Reflection Repair")
    st.caption("Granular tracking of LLM proposals, validation feedback, and reflection loops.")
    llm_interactions = inspector_data.get("llm_interactions", [])
    if llm_interactions:
        for item in llm_interactions:
            t_str = item.get("timestamp", "")[11:19]
            ev_type = item.get("type")
            if ev_type == "llm_proposal":
                st.markdown(f"🤖 **[{t_str}] Proposal Generated:** {item.get('summary')}")
            elif ev_type == "plan_validation":
                details = item.get("details", {})
                if details.get("is_valid"):
                    st.success(f"✅ **[{t_str}] Validation Passed:** {item.get('summary')}")
                else:
                    st.warning(f"⚠️ **[{t_str}] Validation Errors:** {item.get('summary')}")
                    with st.expander("Show Validation Errors"):
                        for err in details.get("errors", []):
                            st.write(f"- `{err}`")
            elif ev_type == "llm_reflection":
                st.info(f"🔄 **[{t_str}] Reflection Loop:** {item.get('summary')}")
    else:
        if job_status.user_constraints and not job_status.user_constraints.get("use_llm_planner"):
            st.write("ℹ️ Deterministic plan proposer was used for this experiment (LLM disabled).")
        else:
            st.write("No LLM interactions recorded for this execution.")

    st.divider()

    # 6. Failure Diagnosis & Recovery History
    st.subheader("4. Failure Diagnosis & Adaptive Recovery")
    st.caption("Diagnostic taxonomy and deterministic hyperparameter adjustments applied upon error.")
    recovery_history = inspector_data.get("recovery_history", [])
    if recovery_history:
        for item in recovery_history:
            t_str = item.get("timestamp", "")[11:19]
            ev_type = item.get("type")
            if ev_type == "failure_diagnosis":
                st.error(f"🔍 **[{t_str}] Failure Diagnosed:** {item.get('summary')}")
            elif ev_type == "recovery_action":
                st.info(f"🔧 **[{t_str}] Adaptive Action Applied:** {item.get('summary')}")
                details = item.get("details", {})
                if details.get("actions"):
                    st.write(f"Actions: `{details['actions']}`")
    else:
        st.success("✅ **Zero Runtime Failures:** All executed pipeline stages succeeded without recovery interventions.")

    st.divider()

    # 7. Expandable Raw Event Log
    st.subheader("5. Granular Event Trace (Chronological)")
    all_events = inspector_data.get("events", [])
    with st.expander(f"View Full Event Log ({len(all_events)} events recorded)", expanded=False):
        filter_type = st.selectbox(
            "Filter Event Types",
            options=["All Events", "Node Events", "Routing Decisions", "LLM / Planning", "Failures & Recovery"],
        )

        for ev in all_events:
            ev_type = ev.get("event_type", "")
            if filter_type == "Node Events" and not ev_type.startswith("node_"):
                continue
            if filter_type == "Routing Decisions" and ev_type != "routing_decision":
                continue
            if filter_type == "LLM / Planning" and ev_type not in ("llm_proposal", "llm_reflection", "plan_validation"):
                continue
            if filter_type == "Failures & Recovery" and ev_type not in ("failure_diagnosis", "recovery_action"):
                continue

            t_str = ev.get("timestamp", "")[11:19]
            ev_node = f"`{ev.get('node')}`" if ev.get("node") else ""
            st.markdown(f"- **`{t_str}`** `[{ev_type}]` {ev_node} — {ev.get('summary')}")
            if ev.get("details"):
                st.json(ev["details"], expanded=False)

    if auto_poll and job_status.status in (JobStatus.RUNNING, JobStatus.WAITING_FOR_APPROVAL):
        time.sleep(2.0)
        st.rerun()


def _render_approval_panel(client: Any, experiment_id: str) -> None:
    """Render interactive plan review and sign-off panel for paused experiments."""
    st.warning("### ⏸️ Human Approval Gate: Plan Review Required", icon="⏸️")
    st.write("The workflow has paused at `approval_gate`. Review the candidate configurations below before permitting model training.")

    try:
        plan_view = client.get_pending_approval(experiment_id)
    except Exception as e:
        # Fall back to raw endpoint if client method raises
        st.error(f"Could not load plan details: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"**Task Type:** `{plan_view.task_type}`")
    with c2:
        st.write(f"**Primary Metric:** `{plan_view.primary_metric}`")
    with c3:
        st.write(f"**CV Folds:** `{plan_view.cv_folds}`")
    with c4:
        st.write(f"**Test Size:** `{plan_view.test_size * 100:.0f}%`")

    st.write(f"**Proposed Candidate Models ({len(plan_view.candidates)}):**")
    for cand in plan_view.candidates:
        with st.expander(f"Candidate: {cand.candidate_name} ({cand.model_family}) {'[BASELINE]' if cand.is_baseline else ''}"):
            st.json(cand.hyperparameters)

    # Action buttons
    col_app, col_rej = st.columns(2)
    with col_app:
        feedback_text = st.text_input("Approval Notes (optional)", key="appr_notes")
        if st.button("✅ Approve Plan & Resume Training", type="primary", use_container_width=True):
            try:
                import httpx
                url = f"{client.base_url}/experiments/{experiment_id}/approve"
                res = httpx.post(url, json={"feedback": feedback_text or "Approved via Inspector UI"})
                if res.status_code == 200:
                    st.success("Plan approved! Workflow execution resumed.")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(f"Approval failed: {res.text}")
            except Exception as exc:
                st.error(f"Approval request failed: {exc}")

    with col_rej:
        reject_reason = st.text_input("Rejection Reason", key="rej_notes")
        if st.button("❌ Reject Plan & Terminate", type="secondary", use_container_width=True):
            try:
                import httpx
                url = f"{client.base_url}/experiments/{experiment_id}/reject"
                res = httpx.post(url, json={"feedback": reject_reason or "Rejected via Inspector UI"})
                if res.status_code == 200:
                    st.warning("Plan rejected. Workflow terminated.")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(f"Rejection failed: {res.text}")
            except Exception as exc:
                st.error(f"Rejection request failed: {exc}")


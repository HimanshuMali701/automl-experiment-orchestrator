"""Model benchmark leaderboard and candidate detail view components."""

from typing import Optional
import pandas as pd
import streamlit as st

from automl.api.schemas import CandidateSummaryResponse, ExperimentResultResponse


def render_leaderboard_view() -> None:
    """Render View 5 (Model Leaderboard) and View 6 (Candidate Deep Dive)."""
    st.header("5. Model Benchmark Leaderboard", divider="gray")

    results: Optional[ExperimentResultResponse] = st.session_state.get("experiment_results")

    if not results or not results.leaderboard:
        st.info("ℹ️ No experiment results available yet. Run an experiment and wait for completion.", icon="ℹ️")
        return

    primary_metric = results.primary_metric

    # 1. Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏆 Champion Model", results.best_candidate_name or "None")
    with c2:
        val_score = None
        champ = next((c for c in results.leaderboard if c.is_champion), None)
        if champ and primary_metric in champ.cv_mean_metrics:
            val_score = champ.cv_mean_metrics[primary_metric]
        st.metric(f"CV {primary_metric.upper()}", f"{val_score:.4f}" if val_score is not None else "N/A")
    with c3:
        st.metric("Total Candidates", len(results.leaderboard))
    with c4:
        st.metric("Total Wall Time", f"{results.total_execution_time_seconds:.2f}s")

    # 2. Main Leaderboard Table
    st.subheader(f"Ranked Leaderboard (Sorted by Validation `{primary_metric}`)")

    table_rows = []
    for rank, cand in enumerate(results.leaderboard, start=1):
        cv_mean = cand.cv_mean_metrics.get(primary_metric, None)
        cv_std = cand.cv_std_metrics.get(primary_metric, 0.0)

        formatted_score = f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_mean is not None else "N/A"
        badge = "🏆 Champion" if cand.is_champion else ("Baseline" if cand.is_baseline else "Candidate")

        table_rows.append({
            "Rank": f"#{rank}",
            "Candidate Name": cand.candidate_name,
            "Role": badge,
            "Model Family": cand.model_family,
            "Status": cand.status,
            f"CV {primary_metric}": formatted_score,
            "Fit Time (s)": f"{cand.fit_time_seconds:.3f}",
        })

    leaderboard_df = pd.DataFrame(table_rows)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    # 3. Visual Performance Comparison Chart
    st.subheader("📊 Cross-Validation Performance Comparison")
    chart_data = []
    for cand in results.leaderboard:
        if primary_metric in cand.cv_mean_metrics:
            chart_data.append({
                "Candidate": cand.candidate_name,
                f"CV {primary_metric}": cand.cv_mean_metrics[primary_metric],
            })

    if chart_data:
        chart_df = pd.DataFrame(chart_data).set_index("Candidate")
        st.bar_chart(chart_df, use_container_width=True)

    # 4. View 6: Candidate Deep Dive & Detail Inspection
    st.header("6. Candidate Detail & Hyperparameters", divider="gray")

    for cand in results.leaderboard:
        expander_title = (
            f"{'🏆 [CHAMPION] ' if cand.is_champion else ''}"
            f"{cand.candidate_name} ({cand.model_family}) - Status: {cand.status.upper()}"
        )
        with st.expander(expander_title, expanded=cand.is_champion):
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.write("**Cross-Validation Metrics (Mean ± Std):**")
                metric_records = []
                for m_name, mean_val in cand.cv_mean_metrics.items():
                    std_val = cand.cv_std_metrics.get(m_name, 0.0)
                    metric_records.append({
                        "Metric": m_name,
                        "Mean Score": f"{mean_val:.4f}",
                        "Std Dev": f"±{std_val:.4f}",
                    })
                if metric_records:
                    st.dataframe(pd.DataFrame(metric_records), use_container_width=True, hide_index=True)
                else:
                    st.caption("No CV metrics recorded (model may have failed or been skipped).")

            with col_b:
                st.write("**Candidate Metadata & Timing:**")
                st.write(f"- **Model Family:** `{cand.model_family}`")
                st.write(f"- **Is Baseline:** `{'Yes' if cand.is_baseline else 'No'}`")
                st.write(f"- **Training Duration:** `{cand.fit_time_seconds:.3f} seconds`")
                st.write(f"- **Run Status:** `{cand.status}`")

                if cand.test_metrics:
                    st.info(
                        f"🔒 **Held-Out Test Score ({primary_metric}):** "
                        f"`{cand.test_metrics.get(primary_metric, 'N/A')}` "
                        "(Evaluated once strictly after model selection)",
                        icon="🎯",
                    )


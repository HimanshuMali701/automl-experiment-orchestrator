"""Final experiment report, held-out test evaluation, and reproducibility component."""

from typing import Optional
import streamlit as st

from automl.api.schemas import ExperimentResultResponse


def render_report_view() -> None:
    """Render View 7: Final Report, Held-Out Test Evaluation, and Reproducibility."""
    st.header("7. Final Experiment Report & Champion Insights", divider="gray")

    results: Optional[ExperimentResultResponse] = st.session_state.get("experiment_results")

    if not results:
        st.info("ℹ️ No completed experiment report to display. Complete an experiment run first.", icon="ℹ️")
        return

    primary_metric = results.primary_metric
    champ_name = results.best_candidate_name or "None"
    champ_cand = next((c for c in results.leaderboard if c.is_champion), None)

    # 1. Champion Showcase Banner
    st.success(
        f"🏆 **Champion Model Selected: `{champ_name}`**  \n"
        f"Automated evaluation identified `{champ_name}` as the top-performing model across cross-validation folds.",
        icon="🥇",
    )

    # 2. STRICT SEPARATION: Validation vs Held-Out Test Metrics
    st.subheader("🎯 Generalization Evaluation (Validation vs. Held-Out Test)")
    st.markdown(
        "> [!IMPORTANT]\n"
        "> **Strict Leakage Prevention:** The held-out test partition is reserved strictly for the final assessment "
        "> of the selected champion model. It was never used during feature selection, cross-validation, or hyperparameter tuning."
    )

    val_score = None
    test_score = results.champion_test_score
    if champ_cand and primary_metric in champ_cand.cv_mean_metrics:
        val_score = champ_cand.cv_mean_metrics[primary_metric]

    c_val, c_test, c_diff = st.columns(3)
    with c_val:
        st.metric(
            label=f"CV Validation {primary_metric.upper()}",
            value=f"{val_score:.4f}" if val_score is not None else "N/A",
            help="Mean cross-validation score across folds on the training partition.",
        )
    with c_test:
        st.metric(
            label=f"Held-Out Test {primary_metric.upper()}",
            value=f"{test_score:.4f}" if test_score is not None else "N/A",
            help="Final unbiased score on the 100% held-out test partition.",
        )
    with c_diff:
        if val_score is not None and test_score is not None:
            delta = test_score - val_score
            delta_str = f"{delta:+.4f}"
            st.metric(
                label="Generalization Gap",
                value=delta_str,
                delta=delta_str,
                help="Test score minus Validation score. Negative indicates mild optimism in CV.",
            )
        else:
            st.metric(label="Generalization Gap", value="N/A")

    # Held-out test metrics table (if secondary metrics exist)
    if champ_cand and champ_cand.test_metrics:
        with st.expander("Detailed Held-Out Test Set Metrics Table", expanded=False):
            test_records = [{"Metric": k, "Test Score": f"{v:.4f}"} for k, v in champ_cand.test_metrics.items()]
            st.table(test_records)

    st.divider()

    # 3. Grounded Result Explanation
    if results.result_explanation:
        expl = results.result_explanation
        st.subheader("💡 Grounded Result Explanation")

        if expl.get("summary"):
            st.markdown(f"**Executive Summary:** {expl['summary']}")

        col_comp, col_gen = st.columns(2)
        with col_comp:
            st.markdown(f"**Model Comparison:**  \n{expl.get('model_comparison', 'N/A')}")
        with col_gen:
            st.markdown(f"**Generalization Check:**  \n{expl.get('generalization_check', 'N/A')}")

        if expl.get("caveats"):
            st.caption(f"**Caveats:** {', '.join(expl['caveats'])}")

    st.divider()

    # 4. Reproducibility & MLflow Tracking
    st.subheader("🔬 Reproducibility & MLflow Artifacts")
    r1, r2 = st.columns(2)
    with r1:
        st.write(f"- **Experiment ID:** `{results.experiment_id}`")
        st.write(f"- **Target Column:** `{results.target_column}`")
        st.write(f"- **Task Type:** `{results.task_type}`")
        st.write(f"- **Training Partition:** `{results.train_rows} rows`")
        st.write(f"- **Held-Out Test Partition:** `{results.test_rows} rows`")

    with r2:
        st.write(f"- **MLflow Experiment ID:** `{results.mlflow_experiment_id or 'default'}`")
        st.write(f"- **MLflow Parent Run ID:** `{results.mlflow_run_id or 'N/A'}`")
        st.write(f"- **Total Duration:** `{results.total_execution_time_seconds:.2f}s`")
        st.markdown(
            "To view detailed MLflow logs, parameters, and model artifacts:  \n"
            "```bash\nmlflow ui --backend-store-uri file:./mlruns\n```"
        )

    # 5. Warnings and Limitations
    if results.warnings:
        st.subheader("⚠️ Experiment Warnings & Limitations")
        for w in results.warnings:
            st.warning(w)


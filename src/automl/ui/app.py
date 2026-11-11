"""Main Streamlit dashboard application for AutoML Experiment Orchestrator."""

import streamlit as st

from automl.ui.components.config import render_experiment_config_view
from automl.ui.components.header import render_header
from automl.ui.components.leaderboard import render_leaderboard_view
from automl.ui.components.progress import render_progress_view
from automl.ui.components.report import render_report_view
from automl.ui.components.validation import render_dataset_view
from automl.ui.state import init_session_state


def main() -> None:
    """Initialize and run the Streamlit AutoML dashboard."""
    st.set_page_config(
        page_title="AutoML Experiment Orchestrator",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_header()

    # Main navigation tabs
    tab_data, tab_config, tab_progress, tab_leaderboard, tab_report = st.tabs([
        "📁 1. Dataset & Profile",
        "⚙️ 2. Configuration",
        "⏳ 3. Progress & Status",
        "🏆 4. Model Leaderboard",
        "📜 5. Final Report",
    ])

    with tab_data:
        render_dataset_view()

    with tab_config:
        render_experiment_config_view()

    with tab_progress:
        render_progress_view()

    with tab_leaderboard:
        render_leaderboard_view()

    with tab_report:
        render_report_view()


if __name__ == "__main__":
    main()


"""Dataset upload, profiling, and validation view components."""

import io
from typing import Optional
import pandas as pd
import streamlit as st

from automl.api.schemas import DatasetValidationResponse
from automl.ui.client import APIClientError, APIConnectionError
from automl.ui.state import get_api_client


def render_dataset_view() -> None:
    """Render View 1 (Upload & Validation) and View 2 (Dataset Profiling)."""
    st.header("1. Dataset Ingestion & Validation", divider="gray")

    # 1. File Uploader
    uploaded_file = st.file_uploader(
        "Upload tabular dataset (CSV format)",
        type=["csv"],
        help="Upload a standard CSV file with feature columns and a target column.",
        key="dataset_file_uploader",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        # Parse preview DataFrame defensively
        if (
            st.session_state.get("uploaded_file_name") != file_name
            or st.session_state.get("uploaded_file_bytes") != file_bytes
        ):
            st.session_state["uploaded_file_bytes"] = file_bytes
            st.session_state["uploaded_file_name"] = file_name
            try:
                preview = pd.read_csv(io.BytesIO(file_bytes), nrows=500)
                st.session_state["preview_df"] = preview
            except Exception as e:
                st.error(f"Failed to parse CSV preview: {str(e)}")
                st.session_state["preview_df"] = None

        preview_df: Optional[pd.DataFrame] = st.session_state.get("preview_df")

        if preview_df is not None:
            st.subheader("Data Preview (First 5 Rows)")
            st.dataframe(preview_df.head(5), use_container_width=True)

            # Target Column Selection
            columns = list(preview_df.columns)
            default_target_idx = len(columns) - 1 if columns else 0

            col_target, col_override, col_btn = st.columns([2, 2, 1.5])
            with col_target:
                selected_target = st.selectbox(
                    "Select Target Column",
                    options=columns,
                    index=default_target_idx,
                    help="Column you wish to predict.",
                    key="target_column_select",
                )
                st.session_state["selected_target"] = selected_target

            with col_override:
                task_override = st.selectbox(
                    "Optional Task Override",
                    options=["Auto-detect", "binary_classification", "multiclass_classification", "regression"],
                    index=0,
                    help="Override inferred task type if needed.",
                    key="task_override_select",
                )

            with col_btn:
                st.write("")  # Vertical spacing
                st.write("")
                validate_clicked = st.button(
                    "🔍 Validate & Profile",
                    type="primary",
                    use_container_width=True,
                    disabled=not st.session_state.get("backend_connected", False),
                )

            if validate_clicked:
                _execute_validation(file_bytes, file_name, selected_target, task_override)

    else:
        st.info("👆 Upload a CSV dataset to inspect column types, missingness, and data quality diagnostics.", icon="ℹ️")

    # Display profiling results if validation succeeded
    val_res: Optional[DatasetValidationResponse] = st.session_state.get("validation_result")
    if val_res is not None:
        _render_dataset_profile(val_res)


def _execute_validation(
    file_bytes: bytes,
    file_name: str,
    target_column: str,
    task_override: str,
) -> None:
    """Call backend POST /datasets/validate and update session state."""
    client = get_api_client()
    override = None if task_override == "Auto-detect" else task_override

    with st.spinner("Validating dataset integrity and computing statistical profile via FastAPI..."):
        try:
            val_res = client.validate_dataset(
                file_bytes=file_bytes,
                filename=file_name,
                target_column=target_column,
                task_type_override=override,
            )
            st.session_state["validation_result"] = val_res
            st.session_state["validation_error"] = None
            st.toast("Dataset validated successfully!", icon="✅")
        except APIClientError as exc:
            st.session_state["validation_result"] = None
            st.session_state["validation_error"] = exc.message
            st.error(f"Validation Error: {exc.message}")
        except Exception as exc:
            st.session_state["validation_result"] = None
            st.session_state["validation_error"] = str(exc)
            st.error(f"Unexpected error: {str(exc)}")


def _render_dataset_profile(profile: DatasetValidationResponse) -> None:
    """Render View 2: Detailed Dataset Statistical Profile."""
    st.header("2. Dataset Profile & Diagnostics", divider="gray")

    # High-level Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Rows", f"{profile.row_count:,}")
    with c2:
        st.metric("Total Columns", profile.column_count)
    with c3:
        st.metric("Detected Task", profile.detected_task_type or "Unspecified")
    with c4:
        st.metric("Recommended Metric", profile.recommended_primary_metric or "N/A")
    with c5:
        st.metric("Duplicate Rows", profile.duplicate_rows)

    # Data Quality Warnings
    if profile.data_quality_warnings:
        st.subheader("⚠️ Data Hygiene Diagnostics")
        for warning in profile.data_quality_warnings:
            st.warning(warning, icon="⚠️")
    else:
        st.success("No critical data hygiene issues detected (no constant features or severe leakage).", icon="✅")

    # Columns Breakdown & Missingness
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Feature Type Distribution")
        type_counts = pd.Series(profile.column_types).value_counts().reset_index()
        type_counts.columns = ["Data Type", "Column Count"]
        st.dataframe(type_counts, use_container_width=True, hide_index=True)

        st.caption(
            f"**Numerical Features ({len(profile.numerical_columns)}):** "
            f"{', '.join(profile.numerical_columns) if profile.numerical_columns else 'None'}"
        )
        st.caption(
            f"**Categorical Features ({len(profile.categorical_columns)}):** "
            f"{', '.join(profile.categorical_columns) if profile.categorical_columns else 'None'}"
        )

    with col_right:
        st.subheader("Missing Values by Column")
        missing_data = pd.DataFrame([
            {"Column": col, "Missing Count": count, "Type": profile.column_types.get(col, "unknown")}
            for col, count in profile.missing_value_counts.items()
            if count > 0
        ])
        if not missing_data.empty:
            st.dataframe(missing_data.sort_values(by="Missing Count", ascending=False), use_container_width=True, hide_index=True)
            # Bar chart of missing values
            chart_df = missing_data.set_index("Column")[["Missing Count"]]
            st.bar_chart(chart_df)
        else:
            st.success("Zero missing values across all columns.", icon="✨")

    # Target Column Distribution (if preview df available)
    preview_df: Optional[pd.DataFrame] = st.session_state.get("preview_df")
    target_col = profile.target_column
    if preview_df is not None and target_col and target_col in preview_df.columns:
        st.subheader(f"Target Distribution: `{target_col}`")
        if profile.detected_task_type in ("binary_classification", "multiclass_classification"):
            class_counts = preview_df[target_col].value_counts().reset_index()
            class_counts.columns = ["Class Label", "Frequency"]
            st.dataframe(class_counts, use_container_width=True, hide_index=True)
            st.bar_chart(class_counts.set_index("Class Label"))
        else:
            st.write(preview_df[target_col].describe().to_frame().T)


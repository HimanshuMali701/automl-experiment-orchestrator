"""Model catalogue, allowlists, and compact profile formatting for LLM planning."""

from typing import Any, Dict, List
from automl.profiler.schemas import DatasetProfile, TaskType
from automl.trainer.schemas import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_MODELS,
    REGRESSION_METRICS,
    REGRESSION_MODELS,
    ImputationStrategy,
    ScalingStrategy,
)


def get_supported_catalogue_description(task_type: TaskType) -> str:
    """Return a concise description of allowlisted models and hyperparameters for the task."""
    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        return (
            "SUPPORTED MODEL FAMILIES (Classification):\n"
            "1. dummy_classifier (Baseline):\n"
            "   - Allowed hyperparameters: {'strategy': 'most_frequent' | 'stratified'}\n"
            "2. logistic_regression:\n"
            "   - Allowed hyperparameters: {'C': float (0.001 to 100.0), 'max_iter': int (100 to 2000), 'solver': 'lbfgs' | 'liblinear'}\n"
            "3. random_forest_classifier:\n"
            "   - Allowed hyperparameters: {'n_estimators': int (10 to 300), 'max_depth': int (2 to 20) or null, 'min_samples_split': int (2 to 10)}\n"
            "4. gradient_boosting_classifier:\n"
            "   - Allowed hyperparameters: {'n_estimators': int (10 to 200), 'learning_rate': float (0.01 to 0.5), 'max_depth': int (2 to 8)}\n\n"
            f"SUPPORTED METRICS: {[m.value for m in CLASSIFICATION_METRICS]}\n"
        )
    else:
        return (
            "SUPPORTED MODEL FAMILIES (Regression):\n"
            "1. dummy_regressor (Baseline):\n"
            "   - Allowed hyperparameters: {'strategy': 'mean' | 'median'}\n"
            "2. ridge_regression:\n"
            "   - Allowed hyperparameters: {'alpha': float (0.01 to 100.0)}\n"
            "3. random_forest_regressor:\n"
            "   - Allowed hyperparameters: {'n_estimators': int (10 to 300), 'max_depth': int (2 to 20) or null, 'min_samples_split': int (2 to 10)}\n"
            "4. gradient_boosting_regressor:\n"
            "   - Allowed hyperparameters: {'n_estimators': int (10 to 200), 'learning_rate': float (0.01 to 0.5), 'max_depth': int (2 to 8)}\n\n"
            f"SUPPORTED METRICS: {[m.value for m in REGRESSION_METRICS]}\n"
        )


def format_compact_profile(profile: DatasetProfile) -> str:
    """Convert DatasetProfile into a compact, privacy-safe text summary for LLM context.

    Contains statistical summaries, column dtypes, missingness, and warnings.
    Does NOT contain raw data records, preserving privacy and minimizing token usage.
    """
    target = profile.target_profile
    target_info = "None"
    if target:
        target_info = (
            f"Name: '{target.name}', Task: '{target.detected_task_type.value}', "
            f"Confidence: {target.confidence:.2f}, Ambiguous: {target.is_ambiguous}"
        )
        if target.imbalance_ratio:
            target_info += f", Imbalance Ratio: {target.imbalance_ratio:.2f}:1"
        if target.classes:
            target_info += f", Classes: {target.classes}"

    columns_summary: List[str] = []
    for col_name, col_prof in profile.columns.items():
        desc = f"'{col_name}': type={col_prof.inferred_type.value}, missing={col_prof.missing_pct}%"
        if col_prof.is_constant:
            desc += " [CONSTANT]"
        if col_prof.inferred_type.value == "id_like":
            desc += " [LIKELY_ID]"
        columns_summary.append(desc)

    warnings_summary = [f"- {w.warning_type.value}: {w.message}" for w in profile.warnings]
    if not warnings_summary:
        warnings_summary = ["None"]

    return (
        f"DATASET OVERVIEW:\n"
        f"- Rows: {profile.row_count:,}\n"
        f"- Columns: {profile.column_count}\n"
        f"- Target Column Profile: {target_info}\n\n"
        f"COLUMN TYPES & MISSINGNESS:\n"
        + "\n".join(f"- {c}" for c in columns_summary[:30])
        + (f"\n- ... and {len(columns_summary) - 30} more columns" if len(columns_summary) > 30 else "")
        + f"\n\nDATA HEALTH WARNINGS:\n"
        + "\n".join(warnings_summary)
    )


PLANNER_SYSTEM_PROMPT = """You are an expert AutoML Experiment Planner.
Your task is to analyze a compact dataset profile and propose a controlled, disciplined machine learning experiment plan.

STRICT BOUNDARIES & SAFETY RULES:
1. You MUST propose ONLY models from the provided allowlisted catalogue.
2. You MUST select a primary metric and secondary metrics that are valid for the detected task.
3. You MUST propose between 2 and 6 model candidates. One candidate MUST be a baseline model.
4. You are STRICTLY FORBIDDEN from outputting Python code, shell commands, or tool invocations.
5. You MUST output strictly conforming JSON matching the provided schema.
"""

PLANNER_REPAIR_SYSTEM_PROMPT = """You are an expert AutoML Experiment Planner repairing a previously proposed experiment plan.
A previously generated proposal was evaluated by the validation engine and rejected due to specific rule or allowlist violations.

STRICT REPAIR INSTRUCTIONS:
1. Fix ONLY the invalid components flagged in the validation errors (e.g. replace unsupported model families with allowlisted alternatives, fix hyperparameter types or values, correct incompatible metrics).
2. Retain all valid candidates and preprocessing configurations from the previous proposal.
3. PRESERVE ALL USER CONSTRAINTS without alteration. Explicit user constraints are mandatory and must never be changed.
4. One candidate MUST be designated as the benchmark baseline model.
5. You MUST output strictly conforming JSON matching the provided schema. Do NOT output Python code, markdown fences, or text outside the JSON.
"""


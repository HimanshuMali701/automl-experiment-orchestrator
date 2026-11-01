"""Grounded experiment result explanation engine preventing metric hallucinations."""

from typing import Any, Dict, List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from automl.core.config import Settings
from automl.core.logging import get_logger
from automl.planner.provider import get_llm_provider
from automl.planner.schemas import ExperimentExplanation
from automl.trainer.schemas import CandidateRunResult, ExperimentResult, RunStatus

logger = get_logger("planner.explainer")

EXPLAINER_SYSTEM_PROMPT = """You are a senior ML engineer explaining actual benchmark results to stakeholders.

CRITICAL INTEGRITY RULES:
1. You MUST use ONLY the exact recorded metric numbers provided in the data table.
2. You are STRICTLY FORBIDDEN from inventing benchmark results, hallucinating unmeasured metrics, or exaggerating performance.
3. Compare the champion model directly with the baseline model using the recorded delta.
4. Output strictly conforming JSON matching the provided schema.
"""


def _generate_deterministic_explanation(result: ExperimentResult) -> ExperimentExplanation:
    """Generate a factual, hallucination-free explanation purely from recorded run data."""
    champion_name = result.best_candidate_name or "Unknown"
    metric_key = result.primary_metric.value

    champion_run: Optional[CandidateRunResult] = next(
        (r for r in result.leaderboard if r.candidate_name == champion_name),
        None,
    )
    baseline_run: Optional[CandidateRunResult] = next(
        (r for r in result.leaderboard if r.is_baseline),
        None,
    )

    champ_cv = champion_run.cv_mean_metrics.get(metric_key, 0.0) if champion_run else 0.0
    base_cv = baseline_run.cv_mean_metrics.get(metric_key, 0.0) if baseline_run else 0.0
    test_score = (
        champion_run.test_metrics.get(metric_key, 0.0)
        if (champion_run and champion_run.test_metrics)
        else None
    )

    diff = champ_cv - base_cv
    sign = "+" if diff >= 0 else ""

    summary = (
        f"Experiment concluded across {len(result.leaderboard)} candidates on {result.train_rows} training rows "
        f"and {result.test_rows} held-out test rows. Champion model '{champion_name}' ranked #1 with a CV "
        f"{metric_key} of {champ_cv:.4f}."
    )
    if test_score is not None:
        summary += f" Held-out test score confirmed at {test_score:.4f}."

    baseline_comp = (
        f"Champion '{champion_name}' ({champ_cv:.4f}) achieved an improvement of {sign}{diff:.4f} "
        f"in CV {metric_key} over the baseline '{baseline_run.candidate_name if baseline_run else 'Baseline'}' ({base_cv:.4f})."
    )

    key_metrics = champion_run.test_metrics or champion_run.cv_mean_metrics if champion_run else {}

    tradeoffs = [
        f"Execution time: {champion_run.fit_time_seconds:.2f}s for champion vs {baseline_run.fit_time_seconds:.2f}s for baseline."
        if (champion_run and baseline_run)
        else f"Total experiment time: {result.total_execution_time_seconds:.2f}s.",
        f"Cross-validation variance: ±{champion_run.cv_std_metrics.get(metric_key, 0.0):.4f} across validation folds."
        if champion_run
        else "No variance recorded.",
    ]

    return ExperimentExplanation(
        summary=summary,
        champion_model=champion_name,
        key_metrics=key_metrics,
        baseline_comparison=baseline_comp,
        tradeoffs_and_observations=tradeoffs,
    )


def explain_experiment_results(
    result: ExperimentResult,
    llm: Optional[Any] = None,
    settings: Optional[Settings] = None,
) -> ExperimentExplanation:
    """Generate a narrative summary grounded strictly in actual recorded run metrics.

    If an LLM is available, produces an enriched explanation using LangChain.
    If no LLM or API key is present, falls back deterministically to exact arithmetic summaries.

    Args:
        result: The ExperimentResult containing recorded leaderboard and test scores.
        llm: Optional LangChain chat model.
        settings: Application settings.

    Returns:
        ExperimentExplanation: Structured, factual summary.
    """
    chat_model = get_llm_provider(settings=settings, custom_llm=llm)

    if chat_model is None:
        return _generate_deterministic_explanation(result)

    # Format recorded data table for prompt
    lines = [
        f"Task: {result.task_type.value}",
        f"Target Column: {result.target_column}",
        f"Primary Metric: {result.primary_metric.value}",
        f"Champion Model: {result.best_candidate_name}",
        f"Train / Test Size: {result.train_rows} / {result.test_rows} rows",
        "\nRECORDED LEADERBOARD DATA:",
    ]
    for r in result.leaderboard:
        lines.append(
            f"- {r.candidate_name} (Status: {r.status.value}, Baseline: {r.is_baseline}): "
            f"CV Metrics: {r.cv_mean_metrics}, "
            f"Test Metrics: {r.test_metrics or 'N/A'}, "
            f"Fit Time: {r.fit_time_seconds}s"
        )
    recorded_table = "\n".join(lines)

    parser = PydanticOutputParser(pydantic_object=ExperimentExplanation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLAINER_SYSTEM_PROMPT),
        (
            "human",
            "Explain the following experiment results strictly adhering to the recorded data:\n\n"
            "{recorded_table}\n\n"
            "{format_instructions}"
        ),
    ])

    formatted = prompt.format_prompt(
        recorded_table=recorded_table,
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response = chat_model.invoke(formatted.to_messages())
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed = parser.parse(raw_text)
        return parsed
    except Exception as e:
        logger.warning(f"LLM explanation generation failed ({str(e)}); falling back to deterministic explanation.")
        return _generate_deterministic_explanation(result)


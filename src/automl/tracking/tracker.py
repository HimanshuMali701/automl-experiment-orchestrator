"""MLflow experiment tracking, run persistence, and model artifact logging."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import mlflow
from mlflow.entities import ViewType
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

from automl.core.config import Settings, get_settings
from automl.core.exceptions import TrackingError
from automl.core.logging import get_logger
from automl.profiler.schemas import DatasetProfile
from automl.tracking.fingerprint import compute_dataset_fingerprint
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    ExperimentResult,
    RunStatus,
)

logger = get_logger("tracking.mlflow")


class MLflowTracker:
    """Production experiment tracker encapsulating MLflow run persistence and model artifact storage."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        default_experiment_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tracking_uri = tracking_uri or self.settings.mlflow_tracking_uri
        self.default_experiment_name = (
            default_experiment_name or self.settings.mlflow_default_experiment_name
        )

        # Set tracking URI globally for MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        logger.info(f"Initialized MLflowTracker with tracking URI: {self.tracking_uri}")

    def get_or_create_experiment(self, experiment_name: Optional[str] = None) -> str:
        """Retrieve existing experiment or create a new one under tracking store.

        Args:
            experiment_name: Optional experiment name override.

        Returns:
            str: MLflow experiment_id.
        """
        name = experiment_name or self.default_experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)

        try:
            exp = self.client.get_experiment_by_name(name)
            if exp is not None:
                if exp.lifecycle_stage == "deleted":
                    self.client.restore_experiment(exp.experiment_id)
                return exp.experiment_id
            return self.client.create_experiment(name)
        except Exception as e:
            logger.warning(f"Error checking experiment '{name}': {str(e)}; creating via set_experiment.")
            experiment = mlflow.set_experiment(name)
            return experiment.experiment_id

    def log_experiment(
        self,
        dataset_source: Any,
        profile: DatasetProfile,
        config: ExperimentConfig,
        result: ExperimentResult,
        final_fitted_pipeline: Optional[Any] = None,
        experiment_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Record an entire experiment execution with parent-child runs in MLflow.

        Integrates strict privacy safeguards, separate validation/test metrics,
        reproducibility configurations, error summaries, and versioned pipeline artifacts.

        Args:
            dataset_source: Dataset reference or DataFrame.
            profile: Phase 1 DatasetProfile.
            config: Phase 2/4 ExperimentConfig.
            result: Phase 2/3 ExperimentResult.
            final_fitted_pipeline: The fitted Scikit-learn Pipeline (preprocessing + model).
            experiment_name: Optional MLflow experiment name.

        Returns:
            Tuple[str, str]: (parent_run_id, experiment_id)
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        exp_id = self.get_or_create_experiment(experiment_name)

        # 1. Compute privacy-safe cryptographic dataset fingerprint
        fingerprint = compute_dataset_fingerprint(dataset_source)

        parent_run_name = f"AutoML_{config.target_column}_{result.task_type.value}"

        try:
            with mlflow.start_run(experiment_id=exp_id, run_name=parent_run_name) as parent_run:
                parent_run_id = parent_run.info.run_id

                # Parent tags
                mlflow.set_tags({
                    "run_type": "parent_orchestration",
                    "dataset_fingerprint": fingerprint,
                    "target_column": config.target_column,
                    "task_type": result.task_type.value,
                    "primary_metric": result.primary_metric.value,
                    "best_candidate_name": result.best_candidate_name or "None",
                    "total_candidates": str(len(result.leaderboard)),
                })

                # Parent parameters (reproducibility & split configuration)
                mlflow.log_params({
                    "reproducibility/random_seed": config.random_seed,
                    "reproducibility/cv_folds": config.cv_folds,
                    "reproducibility/test_size": config.test_size,
                    "data/train_rows": result.train_rows,
                    "data/test_rows": result.test_rows,
                    "data/total_features": profile.column_count - 1,
                    "preprocessing/numeric_imputer": config.preprocessing.numeric_imputer.value,
                    "preprocessing/categorical_imputer": config.preprocessing.categorical_imputer.value,
                    "preprocessing/scaling": config.preprocessing.scaling.value,
                })

                # Parent metrics
                mlflow.log_metric("total_execution_time_seconds", result.total_execution_time_seconds)

                # Summary artifact
                summary_data = {
                    "experiment_id": result.experiment_id,
                    "task_type": result.task_type.value,
                    "target_column": result.target_column,
                    "primary_metric": result.primary_metric.value,
                    "best_candidate_name": result.best_candidate_name,
                    "total_execution_time_seconds": result.total_execution_time_seconds,
                    "dataset_fingerprint": fingerprint,
                    "warnings": result.warnings,
                    "leaderboard": [r.model_dump() for r in result.leaderboard],
                }

                with tempfile.TemporaryDirectory() as tmp_dir:
                    summary_path = Path(tmp_dir) / "experiment_summary.json"
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary_data, f, indent=2)
                    mlflow.log_artifact(str(summary_path))

                # 2. Log each candidate model as a nested child run
                for cand in result.leaderboard:
                    is_champion = (cand.candidate_name == result.best_candidate_name)

                    child_run = mlflow.start_run(
                        experiment_id=exp_id,
                        run_name=cand.candidate_name,
                        nested=True,
                    )
                    child_run_id = child_run.info.run_id

                    try:
                        # Candidate tags
                        tags = {
                            "run_type": "candidate_benchmark",
                            "model_family": cand.model_family.value,
                            "is_baseline": str(cand.is_baseline).lower(),
                            "status": cand.status.value,
                            "is_champion": str(is_champion).lower(),
                            "parent_run_id": parent_run_id,
                        }

                        # Safe parameters (only primitive scalar types)
                        safe_params: Dict[str, Any] = {}
                        for k, v in cand.parameters.items():
                            if isinstance(v, (int, float, str, bool)):
                                safe_params[f"model/{k}"] = v
                            else:
                                safe_params[f"model/{k}"] = str(v)

                        mlflow.set_tags(tags)
                        if safe_params:
                            mlflow.log_params(safe_params)

                        # Timing metric
                        mlflow.log_metric("fit_time_seconds", cand.fit_time_seconds)

                        # Failure handling
                        if cand.status == RunStatus.FAILED:
                            err_msg = cand.error_message or "Unknown candidate failure"
                            mlflow.set_tag("error_summary", err_msg)
                            mlflow.end_run(status="FAILED")
                            continue

                        # Validation metrics (prefixed with val_)
                        for metric_name, score in cand.cv_mean_metrics.items():
                            mlflow.log_metric(f"val_{metric_name}", score)
                        for metric_name, std in cand.cv_std_metrics.items():
                            mlflow.log_metric(f"val_{metric_name}_std", std)

                        # Individual fold metrics
                        for fold in cand.fold_results:
                            for metric_name, score in fold.metrics.items():
                                mlflow.log_metric(f"fold_{fold.fold_index}_{metric_name}", score)

                        # Champion Model: Held-out test metrics & versioned pipeline artifact
                        # (Evaluated and logged ONLY after model selection!)
                        if is_champion:
                            if cand.test_metrics:
                                for metric_name, score in cand.test_metrics.items():
                                    mlflow.log_metric(f"test_{metric_name}", score)

                            if final_fitted_pipeline is not None:
                                try:
                                    mlflow.sklearn.log_model(
                                        sk_model=final_fitted_pipeline,
                                        artifact_path="champion_model",
                                    )
                                    logger.info("Successfully logged champion model artifact to MLflow.")
                                except Exception as e:
                                    logger.warning(f"Could not log sklearn model artifact: {str(e)}")

                        mlflow.end_run(status="FINISHED")

                    except Exception as e:
                        mlflow.end_run(status="FAILED")
                        raise

            # Attach MLflow IDs to result object
            result.mlflow_run_id = parent_run_id
            result.mlflow_experiment_id = exp_id
            logger.info(f"Experiment successfully tracked: parent_run_id={parent_run_id}, exp_id={exp_id}")
            return parent_run_id, exp_id

        except Exception as e:
            logger.error(f"Failed to log experiment to MLflow: {str(e)}")
            raise TrackingError(f"MLflow tracking failed: {str(e)}") from e

    def get_experiment_runs(
        self,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query MLflow tracking service and return runs as a structured pandas DataFrame.

        Filters out parent orchestration wrappers to display candidate model benchmarks.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        exp_id = experiment_id or self.get_or_create_experiment(experiment_name)

        runs = self.client.search_runs(
            experiment_ids=[exp_id],
            filter_string="tags.run_type = 'candidate_benchmark'",
        )

        rows = []
        for r in runs:
            data = {
                "run_id": r.info.run_id,
                "candidate_name": r.data.tags.get("mlflow.runName", "unknown"),
                "model_family": r.data.tags.get("model_family", "unknown"),
                "is_baseline": r.data.tags.get("is_baseline") == "true",
                "is_champion": r.data.tags.get("is_champion") == "true",
                "status": r.info.status,
                "start_time": r.info.start_time,
            }
            # Add metrics
            for k, v in r.data.metrics.items():
                data[k] = v
            # Add parameters
            for k, v in r.data.params.items():
                data[k] = v
            rows.append(data)

        return pd.DataFrame(rows)


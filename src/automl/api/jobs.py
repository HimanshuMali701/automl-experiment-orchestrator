"""Thread-safe background experiment job manager and persistence store."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple
import uuid

import pandas as pd
from pydantic import BaseModel, Field

from automl.api.schemas import (
    CandidateSummaryResponse,
    ExperimentJobResponse,
    ExperimentResultResponse,
    JobStatus,
)
from automl.core.config import Settings, get_settings
from automl.core.exceptions import ResourceExhaustionError
from automl.core.logging import get_logger
from automl.profiler.schemas import TaskType
from automl.trainer.schemas import MetricName
from automl.workflow import WorkflowStatus, run_orchestration_workflow

logger = get_logger("api.jobs")


class JobRecord(BaseModel):
    """Internal model capturing complete job lifecycle and execution output."""
    experiment_id: str
    status: JobStatus = JobStatus.QUEUED
    target_column: str
    task_type: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[ExperimentResultResponse] = None
    use_llm_planner: bool = False


class JobManager:
    """Production-minded local background job execution engine and state manager."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.max_concurrent_jobs = self.settings.max_concurrent_jobs
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_jobs,
            thread_name_prefix="automl_worker",
        )
        self.jobs_dir: Path = self.settings.artifacts_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()
        logger.info(f"Initialized JobManager (max_concurrent_jobs={self.max_concurrent_jobs})")

    def _load_persisted_jobs(self) -> None:
        """Hydrate in-memory job records from disk on startup."""
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    job = JobRecord.model_validate(data)
                    self._jobs[job.experiment_id] = job
            except Exception as e:
                logger.warning(f"Failed to load persisted job file {job_file.name}: {str(e)}")

    def _persist_job(self, job: JobRecord) -> None:
        """Save a job record to disk for persistence and auditability."""
        job_file = self.jobs_dir / f"{job.experiment_id}.json"
        try:
            with open(job_file, "w", encoding="utf-8") as f:
                f.write(job.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to persist job record {job.experiment_id}: {str(e)}")

    def get_job(self, experiment_id: str) -> Optional[JobRecord]:
        """Retrieve a job by its unique experiment ID."""
        with self._lock:
            return self._jobs.get(experiment_id)

    def get_job_counts(self) -> Tuple[int, int, int]:
        """Return (active_running, queued, completed) counts."""
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
            completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
            return running, queued, completed

    def submit_job(
        self,
        dataset_df: pd.DataFrame,
        target_column: str,
        use_llm_planner: bool = False,
        primary_metric: Optional[MetricName] = None,
        task_type_override: Optional[TaskType] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> JobRecord:
        """Validate resource capacity and enqueue a new experiment job."""
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
            if (running + queued) >= self.max_concurrent_jobs:
                logger.warning(
                    f"Job submission rejected: concurrency limit reached "
                    f"(running={running}, queued={queued}, max={self.max_concurrent_jobs})"
                )
                raise ResourceExhaustionError(
                    f"Server is currently at capacity ({running + queued} active jobs). "
                    f"Please retry after existing experiments complete.",
                    details={"running": running, "queued": queued, "limit": self.max_concurrent_jobs},
                )

            experiment_id = str(uuid.uuid4())
            now_str = datetime.now(timezone.utc).isoformat()

            job = JobRecord(
                experiment_id=experiment_id,
                status=JobStatus.QUEUED,
                target_column=target_column,
                task_type=task_type_override.value if task_type_override else None,
                created_at=now_str,
                use_llm_planner=use_llm_planner,
            )
            self._jobs[experiment_id] = job
            self._persist_job(job)

        # Enqueue execution in thread pool
        self._executor.submit(
            self._execute_job,
            experiment_id=experiment_id,
            dataset_df=dataset_df,
            target_column=target_column,
            use_llm_planner=use_llm_planner,
            primary_metric=primary_metric,
            task_type_override=task_type_override,
            cv_folds=cv_folds,
            test_size=test_size,
            random_seed=random_seed,
        )

        logger.info(f"Enqueued experiment job: {experiment_id}")
        return job

    def _execute_job(
        self,
        experiment_id: str,
        dataset_df: pd.DataFrame,
        target_column: str,
        use_llm_planner: bool,
        primary_metric: Optional[MetricName],
        task_type_override: Optional[TaskType],
        cv_folds: int,
        test_size: float,
        random_seed: int,
    ) -> None:
        """Worker executing workflow in background thread."""
        with self._lock:
            job = self._jobs[experiment_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._persist_job(job)

        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting execution of job {experiment_id} for target '{target_column}'")

        try:
            final_state = run_orchestration_workflow(
                dataset_source=dataset_df,
                target_column=target_column,
                task_type_override=task_type_override,
                use_llm_planner=use_llm_planner,
            )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            with self._lock:
                job = self._jobs[experiment_id]
                job.completed_at = end_time.isoformat()
                job.execution_time_seconds = round(duration, 3)

                if final_state["workflow_status"] == WorkflowStatus.SUCCESS:
                    exp_res = final_state.get("experiment_result")
                    if exp_res:
                        leaderboard_summaries = []
                        for cand in exp_res.leaderboard:
                            leaderboard_summaries.append(
                                CandidateSummaryResponse(
                                    candidate_name=cand.candidate_name,
                                    model_family=cand.model_family.value,
                                    is_baseline=cand.is_baseline,
                                    is_champion=(cand.candidate_name == exp_res.best_candidate_name),
                                    status=cand.status.value,
                                    cv_mean_metrics=cand.cv_mean_metrics,
                                    cv_std_metrics=cand.cv_std_metrics,
                                    test_metrics=cand.test_metrics if (cand.candidate_name == exp_res.best_candidate_name) else None,
                                    fit_time_seconds=cand.fit_time_seconds,
                                )
                            )

                        champion_score = None
                        if exp_res.best_candidate_name:
                            champ_cand = next(
                                (c for c in exp_res.leaderboard if c.candidate_name == exp_res.best_candidate_name),
                                None,
                            )
                            if champ_cand and champ_cand.test_metrics:
                                champion_score = champ_cand.test_metrics.get(exp_res.primary_metric.value)

                        explanation_dict = None
                        if final_state.get("result_explanation"):
                            explanation_dict = final_state["result_explanation"].model_dump()

                        job.result = ExperimentResultResponse(
                            experiment_id=experiment_id,
                            status=JobStatus.COMPLETED,
                            target_column=exp_res.target_column,
                            task_type=exp_res.task_type.value,
                            primary_metric=exp_res.primary_metric.value,
                            best_candidate_name=exp_res.best_candidate_name,
                            champion_test_score=champion_score,
                            train_rows=exp_res.train_rows,
                            test_rows=exp_res.test_rows,
                            total_execution_time_seconds=exp_res.total_execution_time_seconds,
                            mlflow_run_id=exp_res.mlflow_run_id,
                            mlflow_experiment_id=exp_res.mlflow_experiment_id,
                            leaderboard=leaderboard_summaries,
                            result_explanation=explanation_dict,
                            warnings=exp_res.warnings,
                        )
                        job.task_type = exp_res.task_type.value

                    job.status = JobStatus.COMPLETED
                    logger.info(f"Job {experiment_id} successfully completed in {duration:.2f}s")

                else:
                    err_msg = final_state.get("error_message") or "Workflow failed without explicit message"
                    job.status = JobStatus.FAILED
                    job.error_message = err_msg
                    logger.warning(f"Job {experiment_id} workflow failed: {err_msg}")

                self._persist_job(job)

        except Exception as e:
            logger.exception(f"Unhandled error in job {experiment_id}: {str(e)}")
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            with self._lock:
                job = self._jobs[experiment_id]
                job.status = JobStatus.FAILED
                job.completed_at = end_time.isoformat()
                job.execution_time_seconds = round(duration, 3)
                job.error_message = f"Experiment processing failed: {str(e)}"
                self._persist_job(job)

    def shutdown(self) -> None:
        """Gracefully terminate thread pool."""
        logger.info("Shutting down JobManager thread pool...")
        self._executor.shutdown(wait=False)


# Global singleton instance
_job_manager: Optional[JobManager] = None
_job_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Return the application JobManager singleton."""
    global _job_manager
    with _job_manager_lock:
        if _job_manager is None:
            _job_manager = JobManager()
        return _job_manager


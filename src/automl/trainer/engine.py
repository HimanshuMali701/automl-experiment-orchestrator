"""Deterministic ML experiment engine orchestrating leakage-free training, CV, and evaluation."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from automl.core.config import Settings, get_settings
from automl.core.exceptions import ExperimentPlanError, ModelTrainingError
from automl.core.logging import get_logger
from automl.profiler.loader import load_dataset
from automl.profiler.schemas import DatasetProfile, TaskType
from automl.trainer.metrics import compute_metrics
from automl.trainer.models import instantiate_model
from automl.trainer.preprocessing import build_preprocessor, extract_feature_columns
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    ExperimentResult,
    FoldResult,
    GREATER_IS_BETTER,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    PreprocessingConfig,
    RunStatus,
)

logger = get_logger("trainer.engine")


def build_default_experiment_config(
    profile: DatasetProfile,
    primary_metric: Optional[MetricName] = None,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> ExperimentConfig:
    """Generate a deterministic baseline and candidate model plan based on dataset profile.

    Requires zero LLM interaction and provides standard, robust ML models.

    Args:
        profile: DatasetProfile from Phase 1.
        primary_metric: Optional override for primary metric.
        cv_folds: Number of CV folds (default: 5).
        test_size: Held-out test set ratio (default: 0.2).
        random_seed: Reproducibility seed.

    Returns:
        ExperimentConfig: Complete, validated experiment configuration.
    """
    if profile.target_profile is None:
        raise ExperimentPlanError("Cannot generate experiment plan without a target column in DatasetProfile.")

    task_type = profile.target_profile.detected_task_type
    target_column = profile.target_profile.name

    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        # Choose primary metric based on balance
        is_imbalanced = (
            profile.target_profile.imbalance_ratio is not None
            and profile.target_profile.imbalance_ratio > 3.0
        )
        default_metric = MetricName.F1_MACRO if is_imbalanced else MetricName.ACCURACY
        selected_metric = primary_metric or default_metric

        secondary = [
            MetricName.ACCURACY,
            MetricName.PRECISION_MACRO,
            MetricName.RECALL_MACRO,
            MetricName.F1_MACRO,
        ]
        if task_type == TaskType.BINARY_CLASSIFICATION:
            secondary.append(MetricName.ROC_AUC_OVR)

        # Remove primary from secondary if duplicate
        secondary = [m for m in secondary if m != selected_metric]

        candidates = [
            ModelCandidateConfig(
                name="Baseline (Majority Class)",
                model_family=ModelFamily.DUMMY_CLASSIFIER,
                hyperparameters={"strategy": "most_frequent"},
                is_baseline=True,
            ),
            ModelCandidateConfig(
                name="Logistic Regression",
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                hyperparameters={"C": 1.0, "max_iter": 1000},
            ),
            ModelCandidateConfig(
                name="Random Forest",
                model_family=ModelFamily.RANDOM_FOREST_CLASSIFIER,
                hyperparameters={"n_estimators": 50, "max_depth": 10},
            ),
            ModelCandidateConfig(
                name="Gradient Boosting",
                model_family=ModelFamily.GRADIENT_BOOSTING_CLASSIFIER,
                hyperparameters={"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
            ),
        ]

    else:
        # Regression
        selected_metric = primary_metric or MetricName.RMSE
        secondary = [m for m in [MetricName.RMSE, MetricName.MAE, MetricName.R2] if m != selected_metric]

        candidates = [
            ModelCandidateConfig(
                name="Baseline (Mean Predictor)",
                model_family=ModelFamily.DUMMY_REGRESSOR,
                hyperparameters={"strategy": "mean"},
                is_baseline=True,
            ),
            ModelCandidateConfig(
                name="Ridge Regression",
                model_family=ModelFamily.RIDGE_REGRESSION,
                hyperparameters={"alpha": 1.0},
            ),
            ModelCandidateConfig(
                name="Random Forest Regressor",
                model_family=ModelFamily.RANDOM_FOREST_REGRESSOR,
                hyperparameters={"n_estimators": 50, "max_depth": 10},
            ),
            ModelCandidateConfig(
                name="Gradient Boosting Regressor",
                model_family=ModelFamily.GRADIENT_BOOSTING_REGRESSOR,
                hyperparameters={"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
            ),
        ]

    return ExperimentConfig(
        target_column=target_column,
        task_type=task_type,
        primary_metric=selected_metric,
        secondary_metrics=secondary,
        test_size=test_size,
        cv_folds=cv_folds,
        random_seed=random_seed,
        candidates=candidates,
        preprocessing=PreprocessingConfig(),
    )


class ExperimentEngine:
    """Orchestrates deterministic model training, leakage-free CV, and evaluation."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def run_experiment(
        self,
        data: Union[str, pd.DataFrame],
        profile: DatasetProfile,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        """Run an end-to-end experiment with cross-validation and held-out test evaluation.

        Args:
            data: Raw DataFrame or path to CSV file.
            profile: DatasetProfile from Phase 1.
            config: Validated ExperimentConfig.

        Returns:
            ExperimentResult: Complete experiment output with ranked leaderboard and fold details.
        """
        start_time = time.time()
        logger.info(
            f"Starting experiment for target '{config.target_column}' "
            f"({config.task_type.value}), primary metric: {config.primary_metric.value}"
        )

        # 1. Load data
        df = load_dataset(data, max_size_mb=self.settings.max_upload_size_mb)
        target_col = config.target_column
        all_metrics = [config.primary_metric] + config.secondary_metrics

        # 2. Extract feature columns (preventing identifier column contamination)
        numeric_cols, cat_cols, dropped_cols = extract_feature_columns(profile, target_col)
        if not numeric_cols and not cat_cols:
            raise ModelTrainingError("No usable predictive features found in dataset.")

        feature_cols = numeric_cols + cat_cols
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        warnings: List[str] = []
        if dropped_cols:
            warnings.append(f"Excluded columns from training: {dropped_cols}")

        # 3. Reserve held-out test set (STRICT LEAKAGE GUARD)
        stratify_split = None
        if config.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            # Check minimum class frequency for stratification
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:
                stratify_split = y
            else:
                warnings.append("Stratified train-test split not feasible (class count < 2); using random split.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=stratify_split,
        )

        train_rows = len(X_train)
        test_rows = len(X_test)
        logger.info(f"Split dataset into Train: {train_rows:,} rows, Held-out Test: {test_rows:,} rows")

        # 4. Cross-validation splitter on training set only
        cv_splitter: Any
        if config.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            min_train_class_count = y_train.value_counts().min()
            if min_train_class_count >= config.cv_folds:
                cv_splitter = StratifiedKFold(
                    n_splits=config.cv_folds,
                    shuffle=True,
                    random_state=config.random_seed,
                )
            else:
                warnings.append(
                    f"Class count in training set ({min_train_class_count}) < cv_folds ({config.cv_folds}); "
                    "falling back to non-stratified KFold."
                )
                cv_splitter = KFold(
                    n_splits=config.cv_folds,
                    shuffle=True,
                    random_state=config.random_seed,
                )
        else:
            cv_splitter = KFold(
                n_splits=config.cv_folds,
                shuffle=True,
                random_state=config.random_seed,
            )

        # 5. Evaluate candidates
        candidate_results: List[CandidateRunResult] = []

        for candidate_cfg in config.candidates:
            elapsed_time = time.time() - start_time
            if elapsed_time > config.max_total_time_seconds:
                warnings.append(f"Reached time limit ({config.max_total_time_seconds}s); skipping '{candidate_cfg.name}'.")
                candidate_results.append(
                    CandidateRunResult(
                        candidate_name=candidate_cfg.name,
                        model_family=candidate_cfg.model_family,
                        is_baseline=candidate_cfg.is_baseline,
                        status=RunStatus.SKIPPED,
                        parameters=candidate_cfg.hyperparameters,
                        error_message="Skipped due to experiment total time limit.",
                    )
                )
                continue

            run_res = self._evaluate_candidate(
                candidate_cfg=candidate_cfg,
                numeric_cols=numeric_cols,
                cat_cols=cat_cols,
                preprocessing_cfg=config.preprocessing,
                X_train=X_train,
                y_train=y_train,
                cv_splitter=cv_splitter,
                task_type=config.task_type,
                metrics=all_metrics,
                random_seed=config.random_seed,
            )
            candidate_results.append(run_res)

        # 6. Rank leaderboard based strictly on CV primary metric
        greater_is_better = GREATER_IS_BETTER[config.primary_metric]
        primary_metric_key = config.primary_metric.value

        def sort_key(res: CandidateRunResult) -> float:
            if res.status != RunStatus.SUCCESS or primary_metric_key not in res.cv_mean_metrics:
                return -float("inf") if greater_is_better else float("inf")
            return res.cv_mean_metrics[primary_metric_key]

        ranked_leaderboard = sorted(candidate_results, key=sort_key, reverse=greater_is_better)

        # 7. Final held-out test evaluation on the selected best model
        best_candidate_name = None
        successful_candidates = [r for r in ranked_leaderboard if r.status == RunStatus.SUCCESS]

        if successful_candidates:
            best_candidate_res = successful_candidates[0]
            best_candidate_name = best_candidate_res.candidate_name

            # Find matching candidate config
            best_cfg = next(c for c in config.candidates if c.name == best_candidate_name)

            # Fit full pipeline on entire X_train and evaluate once on X_test
            try:
                preprocessor = build_preprocessor(numeric_cols, cat_cols, config.preprocessing)
                estimator = instantiate_model(best_cfg, random_seed=config.random_seed)
                final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

                final_pipeline.fit(X_train, y_train)
                y_test_pred = final_pipeline.predict(X_test)
                y_test_proba = (
                    final_pipeline.predict_proba(X_test)
                    if hasattr(final_pipeline, "predict_proba")
                    else None
                )

                test_scores = compute_metrics(
                    task_type=config.task_type,
                    requested_metrics=all_metrics,
                    y_true=y_test,
                    y_pred=y_test_pred,
                    y_proba=y_test_proba,
                )
                best_candidate_res.test_metrics = test_scores
                logger.info(f"Best model '{best_candidate_name}' test metrics: {test_scores}")

            except Exception as e:
                warnings.append(f"Failed final test set evaluation for '{best_candidate_name}': {str(e)}")

        total_exec_time = round(time.time() - start_time, 2)

        return ExperimentResult(
            task_type=config.task_type,
            target_column=config.target_column,
            primary_metric=config.primary_metric,
            best_candidate_name=best_candidate_name,
            leaderboard=ranked_leaderboard,
            total_execution_time_seconds=total_exec_time,
            train_rows=train_rows,
            test_rows=test_rows,
            warnings=warnings,
        )

    def _evaluate_candidate(
        self,
        candidate_cfg: ModelCandidateConfig,
        numeric_cols: List[str],
        cat_cols: List[str],
        preprocessing_cfg: PreprocessingConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_splitter: Any,
        task_type: TaskType,
        metrics: List[MetricName],
        random_seed: int,
    ) -> CandidateRunResult:
        """Fit and cross-validate an individual candidate with strict failure isolation."""
        cand_start = time.time()
        fold_results: List[FoldResult] = []
        metric_fold_accum: Dict[str, List[float]] = {m.value: [] for m in metrics}

        try:
            # Instantiate model
            base_estimator = instantiate_model(candidate_cfg, random_seed=random_seed)

            # Perform cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train)):
                fold_t0 = time.time()
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Build fresh pipeline for each fold: preprocessor fitted ONLY on train fold
                preprocessor = build_preprocessor(numeric_cols, cat_cols, preprocessing_cfg)
                fold_model = clone(base_estimator)
                fold_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", fold_model)])

                fold_pipeline.fit(X_tr, y_tr)
                y_pred = fold_pipeline.predict(X_val)
                y_proba = fold_pipeline.predict_proba(X_val) if hasattr(fold_pipeline, "predict_proba") else None

                fold_scores = compute_metrics(
                    task_type=task_type,
                    requested_metrics=metrics,
                    y_true=y_val,
                    y_pred=y_pred,
                    y_proba=y_proba,
                )

                fold_time = round(time.time() - fold_t0, 3)
                fold_results.append(
                    FoldResult(fold_index=fold_idx, metrics=fold_scores, fit_time_seconds=fold_time)
                )

                for m_name, val in fold_scores.items():
                    metric_fold_accum[m_name].append(val)

            # Calculate mean and std
            cv_mean = {m: round(float(np.mean(vals)), 4) for m, vals in metric_fold_accum.items() if vals}
            cv_std = {m: round(float(np.std(vals)), 4) for m, vals in metric_fold_accum.items() if vals}
            cand_time = round(time.time() - cand_start, 2)

            return CandidateRunResult(
                candidate_name=candidate_cfg.name,
                model_family=candidate_cfg.model_family,
                is_baseline=candidate_cfg.is_baseline,
                status=RunStatus.SUCCESS,
                parameters=candidate_cfg.hyperparameters,
                cv_mean_metrics=cv_mean,
                cv_std_metrics=cv_std,
                fold_results=fold_results,
                fit_time_seconds=cand_time,
            )

        except Exception as e:
            logger.warning(f"Candidate '{candidate_cfg.name}' failed during execution: {str(e)}")
            cand_time = round(time.time() - cand_start, 2)
            return CandidateRunResult(
                candidate_name=candidate_cfg.name,
                model_family=candidate_cfg.model_family,
                is_baseline=candidate_cfg.is_baseline,
                status=RunStatus.FAILED,
                parameters=candidate_cfg.hyperparameters,
                fit_time_seconds=cand_time,
                error_message=str(e),
            )

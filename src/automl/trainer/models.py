"""Model factory for instantiating scikit-learn estimators safely from allowlisted configurations."""

from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from automl.core.exceptions import ModelTrainingError
from automl.trainer.schemas import ModelCandidateConfig, ModelFamily


def instantiate_model(candidate: ModelCandidateConfig, random_seed: int = 42) -> BaseEstimator:
    """Instantiate a scikit-learn estimator from an allowlisted model candidate configuration.

    Args:
        candidate: ModelCandidateConfig with model_family and hyperparameters.
        random_seed: Seed for reproducible initialization.

    Returns:
        BaseEstimator: Configured scikit-learn estimator.

    Raises:
        ModelTrainingError: If the model family is unsupported or parameters are invalid.
    """
    family = candidate.model_family
    params: Dict[str, Any] = dict(candidate.hyperparameters)

    try:
        # Baselines
        if family == ModelFamily.DUMMY_CLASSIFIER:
            strategy = params.get("strategy", "most_frequent")
            return DummyClassifier(strategy=strategy)

        elif family == ModelFamily.DUMMY_REGRESSOR:
            strategy = params.get("strategy", "mean")
            return DummyRegressor(strategy=strategy)

        # Classification models
        elif family == ModelFamily.LOGISTIC_REGRESSION:
            max_iter = params.get("max_iter", 1000)
            c_val = float(params.get("C", 1.0))
            solver = params.get("solver", "lbfgs")
            return LogisticRegression(
                max_iter=max_iter,
                C=c_val,
                solver=solver,
                random_state=random_seed,
            )

        elif family == ModelFamily.RANDOM_FOREST_CLASSIFIER:
            n_estimators = int(params.get("n_estimators", 50))
            max_depth = params.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            min_samples_split = int(params.get("min_samples_split", 2))
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_seed,
                n_jobs=-1,
            )

        elif family == ModelFamily.GRADIENT_BOOSTING_CLASSIFIER:
            n_estimators = int(params.get("n_estimators", 50))
            learning_rate = float(params.get("learning_rate", 0.1))
            max_depth = int(params.get("max_depth", 3))
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_seed,
            )

        # Regression models
        elif family == ModelFamily.RIDGE_REGRESSION:
            alpha = float(params.get("alpha", 1.0))
            return Ridge(
                alpha=alpha,
                random_state=random_seed,
            )

        elif family == ModelFamily.RANDOM_FOREST_REGRESSOR:
            n_estimators = int(params.get("n_estimators", 50))
            max_depth = params.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            min_samples_split = int(params.get("min_samples_split", 2))
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_seed,
                n_jobs=-1,
            )

        elif family == ModelFamily.GRADIENT_BOOSTING_REGRESSOR:
            n_estimators = int(params.get("n_estimators", 50))
            learning_rate = float(params.get("learning_rate", 0.1))
            max_depth = int(params.get("max_depth", 3))
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_seed,
            )

        else:
            raise ModelTrainingError(
                f"Unsupported model family: '{family}'. Model must be in approved allowlist.",
                details={"model_family": str(family)},
            )

    except Exception as e:
        if isinstance(e, ModelTrainingError):
            raise
        raise ModelTrainingError(
            f"Failed to instantiate candidate '{candidate.name}' ({family}): {str(e)}",
            details={"parameters": params},
        ) from e

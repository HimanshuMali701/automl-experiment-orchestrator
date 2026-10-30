"""Shared pytest fixtures providing synthetic tabular datasets and temporary paths."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide an isolated temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_binary_classification_df() -> pd.DataFrame:
    """Provide a reproducible synthetic binary classification dataset with mixed types."""
    np.random.seed(42)
    n_samples = 200

    age = np.random.normal(loc=40.0, scale=12.0, size=n_samples).round(1)
    income = np.random.exponential(scale=50000.0, size=n_samples).round(2)
    credit_score = np.random.randint(300, 850, size=n_samples).astype(float)
    department = np.random.choice(["sales", "engineering", "support", "marketing"], size=n_samples)
    device_type = np.random.choice(["mobile", "desktop", "tablet"], size=n_samples)

    # Inject some missing values realistically
    credit_score[np.random.choice(n_samples, size=10, replace=False)] = np.nan
    department_list = department.tolist()
    for idx in np.random.choice(n_samples, size=5, replace=False):
        department_list[idx] = None

    # Synthetic binary target correlated with age and income
    logit = 0.03 * (age - 40) + 0.00002 * (income - 50000) + np.random.normal(0, 0.5, size=n_samples)
    target = (logit > 0).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "department": department_list,
        "device_type": device_type,
        "churn": target,
    })


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    """Provide a reproducible synthetic regression dataset."""
    np.random.seed(42)
    n_samples = 200

    square_feet = np.random.uniform(500, 3500, size=n_samples).round(1)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples)
    location_tier = np.random.choice(["urban", "suburban", "rural"], size=n_samples)
    distance_to_center = np.random.uniform(0.5, 25.0, size=n_samples).round(2)

    # Price target
    tier_multiplier = {"urban": 1.5, "suburban": 1.1, "rural": 0.8}
    base_price = (
        square_feet * 180
        + bedrooms * 20000
        - distance_to_center * 1500
        + np.array([tier_multiplier[t] * 50000 for t in location_tier])
        + np.random.normal(0, 10000, size=n_samples)
    )

    return pd.DataFrame({
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "location_tier": location_tier,
        "distance_to_center": distance_to_center,
        "price": base_price.round(2),
    })


@pytest.fixture
def sample_multiclass_df() -> pd.DataFrame:
    """Provide a reproducible synthetic 3-class classification dataset."""
    np.random.seed(42)
    n_samples = 150

    feat_a = np.random.normal(5.0, 1.0, size=n_samples)
    feat_b = np.random.normal(10.0, 2.0, size=n_samples)
    category = np.random.choice(["alpha", "beta"], size=n_samples)
    classes = np.random.choice(["class_0", "class_1", "class_2"], size=n_samples)

    return pd.DataFrame({
        "feature_a": feat_a,
        "feature_b": feat_b,
        "category": category,
        "target_class": classes,
    })

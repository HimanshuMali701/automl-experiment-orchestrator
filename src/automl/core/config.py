"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for AutoML Experiment Orchestrator."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project metadata
    app_name: str = "AutoML Experiment Orchestrator"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    # Reproducibility
    random_seed: int = 42

    # Storage paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data")
    artifacts_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "artifacts")
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_default_experiment_name: str = "automl-experiments"

    # Resource bounds & safety guards
    max_upload_size_mb: int = 50
    max_dataset_rows: int = 100_000
    max_dataset_cols: int = 200
    max_training_time_seconds: int = 600
    max_models_per_experiment: int = 10

    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_concurrent_jobs: int = 2

    # Optional LLM integration (Groq)
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_model_name: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0

    def ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings

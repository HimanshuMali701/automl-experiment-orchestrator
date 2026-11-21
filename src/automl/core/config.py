"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
import shutil
from typing import Optional

from pydantic import Field, field_validator
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
    max_upload_size_mb: int = Field(default=50, ge=1, le=500, description="Max upload size in MB")
    max_dataset_rows: int = Field(default=100_000, ge=10, le=1_000_000, description="Max allowed rows")
    max_dataset_cols: int = Field(default=200, ge=2, le=1000, description="Max allowed columns")
    max_training_time_seconds: int = Field(default=600, ge=10, le=7200, description="Per-experiment timeout in seconds")
    max_models_per_experiment: int = Field(default=10, ge=1, le=50, description="Max models per run")

    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1024, le=65535)
    automl_api_base_url: str = Field(
        default="https://automl-backend-trtp.onrender.com",
        description="Base URL for the AutoML backend API server",
    )
    cors_origins: str = Field(
        default="*",
        description="Allowed CORS origins (comma-separated, e.g. 'https://app.onrender.com,http://localhost:8501', or '*' for all)",
    )
    max_concurrent_jobs: int = Field(default=2, ge=1, le=16, description="Max concurrent background jobs")

    # Optional LLM integration (Groq)
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_model_name: str = "openai/gpt-oss-20b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "testing", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid environment '{v}'. Must be one of {allowed}")
        return v.lower()

    def ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_temp_artifacts(self, older_than_seconds: Optional[int] = None) -> None:
        """Safely clean up temporary job files or uploads without removing persistent outputs."""
        jobs_dir = self.artifacts_dir / "jobs"
        if jobs_dir.exists():
            # Clean up corrupted or stale temporary lock/transient files
            for temp_file in jobs_dir.glob("*.tmp"):
                try:
                    temp_file.unlink(missing_ok=True)
                except OSError:
                    pass


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings

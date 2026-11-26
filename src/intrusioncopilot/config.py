"""Central configuration and path resolution for IntrusionCopilot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    log_level: str = Field(default="INFO")
    random_seed: int = Field(default=42)

    # Raw dataset directories (relative to project_root unless absolute)
    unsw_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw" / "UNSW_NB15")
    cicids_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw" / "CICIDS2017")

    # Processed data, artifacts, and reports
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")
    configs_dir: Path = Field(default=PROJECT_ROOT / "configs")
    alert_db_path: Path = Field(default=PROJECT_ROOT / "artifacts" / "alerts.db")

    # LLM configuration (Phase 5)
    llm_provider: Literal["groq", "gemini"] = Field(default="groq")
    groq_api_key: str | None = Field(default=None)
    gemini_api_key: str | None = Field(default=None)

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative paths against the project root."""
        return path if path.is_absolute() else (self.project_root / path).resolve()


settings = Settings()
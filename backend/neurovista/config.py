"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODALITIES = ("t1", "t1ce", "t2", "flair")
CLASS_NAMES = ("background", "necrotic_non_enhancing_core", "edema", "enhancing_tumor")
# Contiguous class ids after remapping BraTS label 4 -> 3.
REGIONS: dict[str, tuple[int, ...]] = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
NUM_CLASSES = 4
RESEARCH_DISCLAIMER = (
    "Research prototype only. This software has not been validated for clinical use and must not be "
    "used for diagnosis, treatment decisions, or patient management. Outputs require qualified review."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NV_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("./var"), description="Root for checkpoints, runs and artifacts")
    checkpoint_path: Path | None = None  # defaults to <data_dir>/checkpoints/best.pt
    device: str = "auto"  # auto | cpu | cuda
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 200
    max_inplane_dim: int = 320
    max_depth: int = 400
    min_inplane_dim: int = 16
    inference_timeout_s: int = 600
    artifact_ttl_minutes: int = 60  # retention for generated results; uploads are never kept
    rate_limit_per_minute: int = 30
    training_enabled: bool = True
    mc_dropout_samples: int = 8
    min_component_voxels: int = 0  # 0 disables small-component filtering
    confidence_high: float = 0.90
    confidence_medium: float = 0.70
    log_level: str = "INFO"

    @property
    def checkpoint_file(self) -> Path:
        return self.checkpoint_path or (self.data_dir / "checkpoints" / "best.pt")

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

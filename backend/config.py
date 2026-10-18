"""Typed settings loaded from config.yaml (+ a few env overrides)."""
import os
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class FusionCfg(BaseModel):
    weights: dict = Field(default_factory=lambda: {"face": 0.4, "voice": 0.3, "text": 0.3})
    mismatch_threshold: float = 0.6
    ignore_neutral: bool = True  # 'neutral vs happy' is not a strong disagreement

    @field_validator("mismatch_threshold")
    @classmethod
    def _unit(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("mismatch_threshold must be within 0..1")
        return v


class SummaryCfg(BaseModel):
    spike_min: float = 0.6     # fused score a non-neutral emotion must reach
    spike_delta: float = 0.3   # ...and exceed its session average by this much
    max_spikes: int = 10


class LimitsCfg(BaseModel):
    max_image_bytes: int = 1_500_000
    max_audio_samples: int = 480_000
    max_text_chars: int = 1000
    min_audio_samples: int = 3200
    allowed_sample_rates: List[int] = [8000, 16000, 22050, 44100, 48000]


class RuntimeCfg(BaseModel):
    predict_timeout_s: float = 5.0
    max_concurrent_predictions: int = 1
    min_frame_interval_ms: int = 500


class DbCfg(BaseModel):
    enabled: bool = True
    database: str = "emotionverse"
    batch_size: int = 20
    flush_interval_s: float = 2.0
    server_selection_timeout_ms: int = 2000


class CorsCfg(BaseModel):
    allowed_origins: List[str] = ["http://localhost:3000"]


# .env at repo root or backend/ (both git-ignored); real env vars win
for _p in (Path(__file__).parent.parent / ".env", Path(__file__).parent / ".env"):
    load_dotenv(_p, override=False)


class Settings(BaseModel):
    fusion: FusionCfg = FusionCfg()
    limits: LimitsCfg = LimitsCfg()
    runtime: RuntimeCfg = RuntimeCfg()
    db: DbCfg = DbCfg()
    cors: CorsCfg = CorsCfg()
    summary: SummaryCfg = SummaryCfg()


def load_settings(path: str | None = None) -> Settings:
    path = path or os.getenv("EMOTIONVERSE_CONFIG") or str(Path(__file__).with_name("config.yaml"))
    data = {}
    if Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
    settings = Settings(**data)
    origins = os.getenv("CORS_ORIGINS")  # comma-separated, set on the platform
    if origins:
        settings.cors.allowed_origins = [o.strip() for o in origins.split(",") if o.strip()]
    if os.getenv("EMOTIONVERSE_DB_ENABLED") in ("0", "false", "False"):
        settings.db.enabled = False
    return settings


settings = load_settings()

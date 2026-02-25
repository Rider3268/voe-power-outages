from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ParserProfile:
    expected_rows: int
    expected_hours: int
    red_ratio_threshold: float
    yellow_ratio_threshold: float
    half_block_ratio_threshold: float
    dark_ratio_threshold: float
    blue_hue_min: int
    blue_hue_max: int
    blue_ratio_threshold: float
    blue_off_saturation_threshold: float
    blue_off_value_threshold: float


@dataclass(frozen=True)
class Settings:
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    region_id: str = "Volyn"
    database_path: str = "./data/voe.db"
    retention_days: int = 30

    enable_scheduler: bool = True
    poll_interval_minutes: int = 30
    poll_align_clock: bool = True

    provider_kind: str = "volyn_json"
    provider_metadata_url: str = ""
    provider_option_key: str = "pw_gpv_image_today"
    provider_image_url_path: str = ""
    provider_updated_at_path: str = ""
    provider_region_path: str = ""
    provider_timeout_seconds: int = 20

    parser_profile: ParserProfile = ParserProfile(
        expected_rows=12,
        expected_hours=24,
        red_ratio_threshold=0.22,
        yellow_ratio_threshold=0.24,
        half_block_ratio_threshold=0.35,
        dark_ratio_threshold=0.20,
        blue_hue_min=90,
        blue_hue_max=125,
        blue_ratio_threshold=0.40,
        blue_off_saturation_threshold=90.0,
        blue_off_value_threshold=185.0,
    )


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    return int(raw)


def _as_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    return float(raw)


def load_settings() -> Settings:
    parser_profile = ParserProfile(
        expected_rows=_as_int(os.getenv("PARSER_EXPECTED_ROWS"), 12),
        expected_hours=_as_int(os.getenv("PARSER_EXPECTED_HOURS"), 24),
        red_ratio_threshold=_as_float(os.getenv("PARSER_RED_RATIO_THRESHOLD"), 0.22),
        yellow_ratio_threshold=_as_float(os.getenv("PARSER_YELLOW_RATIO_THRESHOLD"), 0.24),
        half_block_ratio_threshold=_as_float(os.getenv("PARSER_HALF_BLOCK_RATIO_THRESHOLD"), 0.35),
        dark_ratio_threshold=_as_float(os.getenv("PARSER_DARK_RATIO_THRESHOLD"), 0.20),
        blue_hue_min=_as_int(os.getenv("PARSER_BLUE_HUE_MIN"), 90),
        blue_hue_max=_as_int(os.getenv("PARSER_BLUE_HUE_MAX"), 125),
        blue_ratio_threshold=_as_float(os.getenv("PARSER_BLUE_RATIO_THRESHOLD"), 0.40),
        blue_off_saturation_threshold=_as_float(
            os.getenv("PARSER_BLUE_OFF_SATURATION_THRESHOLD"), 90.0
        ),
        blue_off_value_threshold=_as_float(os.getenv("PARSER_BLUE_OFF_VALUE_THRESHOLD"), 185.0),
    )

    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=_as_int(os.getenv("APP_PORT"), 8000),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        region_id=os.getenv("REGION_ID", "Volyn"),
        database_path=os.getenv("DATABASE_PATH", "./data/voe.db"),
        retention_days=_as_int(os.getenv("RETENTION_DAYS"), 30),
        enable_scheduler=_as_bool(os.getenv("ENABLE_SCHEDULER"), True),
        poll_interval_minutes=_as_int(os.getenv("POLL_INTERVAL_MINUTES"), 30),
        poll_align_clock=_as_bool(os.getenv("POLL_ALIGN_CLOCK"), True),
        provider_kind=os.getenv("PROVIDER_KIND", "volyn_json"),
        provider_metadata_url=os.getenv("PROVIDER_METADATA_URL", ""),
        provider_option_key=os.getenv("PROVIDER_OPTION_KEY", "pw_gpv_image_today"),
        provider_image_url_path=os.getenv("PROVIDER_IMAGE_URL_PATH", ""),
        provider_updated_at_path=os.getenv("PROVIDER_UPDATED_AT_PATH", ""),
        provider_region_path=os.getenv("PROVIDER_REGION_PATH", ""),
        provider_timeout_seconds=_as_int(os.getenv("PROVIDER_TIMEOUT_SECONDS"), 20),
        parser_profile=parser_profile,
    )

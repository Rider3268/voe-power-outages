from __future__ import annotations

from app.config import Settings
from app.providers.base import ScheduleImageProvider
from app.providers.volyn_api import VolynJSONProvider


class UnknownProviderError(RuntimeError):
    pass


def build_provider(settings: Settings) -> ScheduleImageProvider:
    if settings.provider_kind == "volyn_json":
        return VolynJSONProvider(
            metadata_url=settings.provider_metadata_url,
            image_url_path=settings.provider_image_url_path,
            updated_at_path=settings.provider_updated_at_path,
            region_path=settings.provider_region_path,
            default_region_id=settings.region_id,
            timeout_seconds=settings.provider_timeout_seconds,
            option_key=settings.provider_option_key,
        )

    raise UnknownProviderError(f"Unsupported provider kind: {settings.provider_kind}")

from __future__ import annotations

from typing import Protocol

from app.core.models import FetchResult


class ScheduleImageProvider(Protocol):
    async def fetch_latest(self) -> FetchResult:
        """Fetch the latest metadata that includes a schedule image URL."""

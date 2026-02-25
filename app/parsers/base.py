from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.core.models import ParsedSchedule


class ScheduleParser(Protocol):
    def parse(
        self,
        image_bytes: bytes,
        *,
        region_id: str,
        fetched_at_utc: datetime,
        source_updated_at_utc: datetime | None = None,
    ) -> ParsedSchedule:
        """Parse a schedule image into canonical outage matrix representation."""

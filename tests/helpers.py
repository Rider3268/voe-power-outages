from __future__ import annotations

from datetime import datetime, timezone

from app.core.constants import GPV_GROUPS
from app.core.models import ParsedSchedule


def build_matrix(default: str = "yes") -> dict[str, dict[str, str]]:
    return {
        group: {str(hour): default for hour in range(1, 25)}
        for group in GPV_GROUPS
    }


def sample_parsed_schedule() -> ParsedSchedule:
    matrix = build_matrix("yes")
    matrix["GPV1.1"]["23"] = "no"
    matrix["GPV1.1"]["24"] = "no"
    return ParsedSchedule(
        region_id="Volyn",
        last_updated_utc=datetime(2026, 2, 17, 22, 7, 57, tzinfo=timezone.utc),
        fact_update_text="18.02.2026 00:00",
        today_unix=1771365600,
        data=matrix,
    )

from __future__ import annotations

from datetime import timezone

from app.core.constants import GPV_GROUPS, TIME_TYPE, TIME_ZONE
from app.core.models import ParsedSchedule


def normalize_matrix(data: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for group in GPV_GROUPS:
        raw_group = data.get(group, {})
        normalized[group] = {str(hour): raw_group.get(str(hour), "maybe") for hour in range(1, 25)}
    return normalized


def to_sample_payload(schedule: ParsedSchedule) -> dict:
    today_key = str(schedule.today_unix)
    normalized = normalize_matrix(schedule.data)

    return {
        "regionId": schedule.region_id,
        "lastUpdated": schedule.last_updated_utc.astimezone(timezone.utc).isoformat(),
        "fact": {
            "data": {today_key: normalized},
            "update": schedule.fact_update_text,
            "today": schedule.today_unix,
        },
        "preset": {
            "time_zone": TIME_ZONE,
            "time_type": TIME_TYPE,
        },
    }

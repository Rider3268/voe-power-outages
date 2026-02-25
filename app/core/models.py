from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class PowerState(str, Enum):
    YES = "yes"
    MAYBE = "maybe"
    NO = "no"
    FIRST = "first"
    SECOND = "second"


ScheduleMatrix = dict[str, dict[str, str]]


@dataclass(frozen=True)
class ParsedSchedule:
    region_id: str
    last_updated_utc: datetime
    fact_update_text: str
    today_unix: int
    data: ScheduleMatrix


@dataclass(frozen=True)
class FetchResult:
    region_id: str
    image_url: str
    source_updated_at: datetime | None
    source_payload: dict

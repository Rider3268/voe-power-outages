from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import load_settings
from app.parsers.volyn_schedule_parser import VolynScheduleParser


@pytest.mark.skipif(
    not Path("image_samples/no_outages_schedule_example.png").exists(),
    reason="sample images not available",
)
def test_no_outages_sample_parses_all_yes() -> None:
    parser = VolynScheduleParser(load_settings().parser_profile)
    image = Path("image_samples/no_outages_schedule_example.png").read_bytes()

    parsed = parser.parse(
        image,
        region_id="Volyn",
        fetched_at_utc=datetime.now(tz=timezone.utc),
    )

    counts = Counter()
    for row in parsed.data.values():
        counts.update(row.values())

    assert counts == {"yes": 288}


@pytest.mark.skipif(
    not Path("image_samples/schedule_has_outages_example.png").exists(),
    reason="sample images not available",
)
def test_outages_sample_includes_outage_states() -> None:
    parser = VolynScheduleParser(load_settings().parser_profile)
    image = Path("image_samples/schedule_has_outages_example.png").read_bytes()

    parsed = parser.parse(
        image,
        region_id="Volyn",
        fetched_at_utc=datetime.now(tz=timezone.utc),
    )

    counts = Counter()
    for row in parsed.data.values():
        counts.update(row.values())

    assert counts["no"] > 0
    assert counts["first"] > 0
    assert counts["second"] > 0

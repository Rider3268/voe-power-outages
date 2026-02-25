from __future__ import annotations

from app.core.serialization import to_sample_payload
from tests.helpers import sample_parsed_schedule


def test_to_sample_payload_shape() -> None:
    payload = to_sample_payload(sample_parsed_schedule())

    assert payload["regionId"] == "Volyn"
    assert payload["fact"]["today"] == 1771365600
    assert payload["fact"]["data"]["1771365600"]["GPV1.1"]["23"] == "no"
    assert payload["fact"]["data"]["1771365600"]["GPV1.1"]["24"] == "no"
    assert payload["preset"]["time_type"]["yes"] == "Світло є"
    assert payload["preset"]["time_zone"]["1"] == ["00-01", "00:00", "01:00"]

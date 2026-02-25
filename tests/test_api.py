from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.core.serialization import to_sample_payload
from app.main import create_app
from tests.helpers import sample_parsed_schedule


@pytest.mark.asyncio
async def test_latest_endpoint_returns_payload(tmp_path) -> None:
    settings = Settings(
        enable_scheduler=False,
        database_path=str(tmp_path / "api.db"),
        provider_metadata_url="https://example.test/metadata",
    )
    app = create_app(settings)

    parsed = sample_parsed_schedule()
    payload = to_sample_payload(parsed)

    app.state.repository.save_snapshot(
        region_id="Volyn",
        day_unix=parsed.today_unix,
        last_updated_utc=parsed.last_updated_utc,
        fact_update_text=parsed.fact_update_text,
        payload=payload,
        image_url="https://example.test/schedule.png",
        image_sha256="hash-1",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/v1/schedule/latest", params={"regionId": "Volyn"})
    assert response.status_code == 200
    body = response.json()
    assert body["regionId"] == "Volyn"


@pytest.mark.asyncio
async def test_history_endpoint_returns_items(tmp_path) -> None:
    settings = Settings(
        enable_scheduler=False,
        database_path=str(tmp_path / "api.db"),
        provider_metadata_url="https://example.test/metadata",
    )
    app = create_app(settings)

    parsed = sample_parsed_schedule()
    payload = to_sample_payload(parsed)

    app.state.repository.save_snapshot(
        region_id="Volyn",
        day_unix=parsed.today_unix,
        last_updated_utc=parsed.last_updated_utc,
        fact_update_text=parsed.fact_update_text,
        payload=payload,
        image_url="https://example.test/schedule.png",
        image_sha256="hash-1",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/v1/schedule/history", params={"regionId": "Volyn", "limit": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["regionId"] == "Volyn"

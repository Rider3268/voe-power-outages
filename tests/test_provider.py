from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.providers.volyn_api import VolynJSONProvider


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _DummyClient:
    def __init__(self, payload: dict, capture: dict) -> None:
        self._payload = payload
        self._capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, params: dict | None = None) -> _DummyResponse:
        self._capture["url"] = url
        self._capture["params"] = params
        return _DummyResponse(self._payload)


@pytest.mark.asyncio
async def test_provider_fetches_from_options_endpoint(monkeypatch) -> None:
    payload = {
        "success": True,
        "data": {
            "option_key": "pw_gpv_image_today",
            "option_value": "/media/2026/02/abcdfed237f66_GPV.png",
            "upload_time": "2026-02-18 00:00:00",
        },
    }
    capture: dict = {}

    def _client_factory(*args, **kwargs):
        return _DummyClient(payload=payload, capture=capture)

    monkeypatch.setattr("app.providers.volyn_api.httpx.AsyncClient", _client_factory)

    provider = VolynJSONProvider(
        metadata_url="https://api-voe-poweron.inneti.net/api/options",
        image_url_path="",
        updated_at_path="",
        region_path="",
        default_region_id="Volyn",
        option_key="pw_gpv_image_today",
    )

    result = await provider.fetch_latest()

    assert capture["url"] == "https://api-voe-poweron.inneti.net/api/options"
    assert capture["params"] == {"option_key": "pw_gpv_image_today"}
    assert (
        result.image_url
        == "https://api-voe-poweron.inneti.net/media/2026/02/abcdfed237f66_GPV.png"
    )
    assert result.region_id == "Volyn"
    assert result.source_updated_at == datetime(2026, 2, 18, 0, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_provider_supports_explicit_paths(monkeypatch) -> None:
    payload = {
        "result": {
            "region": "Volyn",
            "image": "media/2026/02/xyzdfed237f66_GPV.png",
            "updated": "18.02.2026 00:00",
        }
    }
    capture: dict = {}

    def _client_factory(*args, **kwargs):
        return _DummyClient(payload=payload, capture=capture)

    monkeypatch.setattr("app.providers.volyn_api.httpx.AsyncClient", _client_factory)

    provider = VolynJSONProvider(
        metadata_url="https://api-voe-poweron.inneti.net/api/options",
        image_url_path="result.image",
        updated_at_path="result.updated",
        region_path="result.region",
        default_region_id="Fallback",
        option_key="",
    )

    result = await provider.fetch_latest()

    assert capture["params"] is None
    assert (
        result.image_url
        == "https://api-voe-poweron.inneti.net/media/2026/02/xyzdfed237f66_GPV.png"
    )
    assert result.region_id == "Volyn"
    assert result.source_updated_at == datetime(2026, 2, 18, 0, 0, tzinfo=timezone.utc)

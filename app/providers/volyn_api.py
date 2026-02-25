from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import httpx

from app.core.models import FetchResult


class ProviderError(RuntimeError):
    pass


_MEDIA_PATH_RE = re.compile(
    r"/media/\d{4}/\d{2}/[A-Za-z0-9._%-]+_GPV\.(?:png|jpg|jpeg)",
    flags=re.IGNORECASE,
)
_TIMESTAMP_HINTS = ("upload", "updated", "created", "date", "time")


def _extract_path(payload: Any, path: str) -> Any:
    current = payload
    if not path:
        return None

    for chunk in path.split("."):
        if isinstance(current, dict):
            if chunk not in current:
                return None
            current = current[chunk]
            continue

        if isinstance(current, list):
            try:
                index = int(chunk)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue

        return None

    return current


def _parse_datetime(raw: Any) -> datetime | None:
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)

    if not isinstance(raw, str):
        return None

    value = raw.strip()
    if not value:
        return None

    candidates = [
        lambda: datetime.fromisoformat(value.replace("Z", "+00:00")),
        lambda: datetime.strptime(value, "%d.%m.%Y %H:%M").replace(tzinfo=timezone.utc),
        lambda: datetime.strptime(value, "%d.%m.%Y").replace(tzinfo=timezone.utc),
        lambda: datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc),
        lambda: datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc),
        lambda: datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc),
    ]

    for parser in candidates:
        try:
            parsed = parser()
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue

    return None


def _extract_media_uri_from_string(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None

    if text.startswith("http://") or text.startswith("https://"):
        match = _MEDIA_PATH_RE.search(text)
        if match is not None:
            path = match.group(0)
            host_end = text.find(path)
            if host_end > 0:
                return text[:host_end] + path
        return text

    match = _MEDIA_PATH_RE.search(text)
    if match is None:
        return None

    return match.group(0)


def _walk_values(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk_values(value)
        return

    if isinstance(node, list):
        for item in node:
            yield from _walk_values(item)


def _find_media_uri(payload: Any) -> str | None:
    for _, value in _walk_values(payload):
        if isinstance(value, str):
            found = _extract_media_uri_from_string(value)
            if found is not None:
                return found
    return None


def _find_upload_time(payload: Any) -> datetime | None:
    hinted: list[datetime] = []
    generic: list[datetime] = []

    for key, value in _walk_values(payload):
        parsed = _parse_datetime(value)
        if parsed is None:
            continue

        if isinstance(key, str) and any(token in key.lower() for token in _TIMESTAMP_HINTS):
            hinted.append(parsed)
        else:
            generic.append(parsed)

    if hinted:
        return hinted[0]
    if generic:
        return generic[0]
    return None


def _normalize_image_url(metadata_url: str, raw_image_ref: str) -> str:
    uri = _extract_media_uri_from_string(raw_image_ref) or raw_image_ref.strip()
    if not uri:
        raise ProviderError("Resolved image URI is empty")

    if uri.startswith("http://") or uri.startswith("https://"):
        return uri

    if uri.startswith("media/"):
        uri = "/" + uri

    return urljoin(metadata_url, uri)


@dataclass
class VolynJSONProvider:
    metadata_url: str
    image_url_path: str
    updated_at_path: str
    region_path: str
    default_region_id: str
    timeout_seconds: int = 20
    option_key: str = ""

    async def fetch_latest(self) -> FetchResult:
        if not self.metadata_url:
            raise ProviderError("PROVIDER_METADATA_URL is empty")

        timeout = httpx.Timeout(self.timeout_seconds)
        params = {"option_key": self.option_key} if self.option_key else None

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(self.metadata_url, params=params)
            response.raise_for_status()
            payload = response.json()

        image_value = _extract_path(payload, self.image_url_path)
        if not isinstance(image_value, str) or not image_value.strip():
            image_value = _find_media_uri(payload)

        if not isinstance(image_value, str) or not image_value.strip():
            raise ProviderError(
                "Could not resolve image URI from provider payload. Set PROVIDER_IMAGE_URL_PATH or check payload shape."
            )

        image_url = _normalize_image_url(self.metadata_url, image_value)

        source_updated_raw = _extract_path(payload, self.updated_at_path)
        source_updated = _parse_datetime(source_updated_raw)
        if source_updated is None:
            source_updated = _find_upload_time(payload)

        source_region_raw = _extract_path(payload, self.region_path)
        source_region = source_region_raw if isinstance(source_region_raw, str) else self.default_region_id

        return FetchResult(
            region_id=source_region,
            image_url=image_url,
            source_updated_at=source_updated,
            source_payload=payload if isinstance(payload, dict) else {"raw": payload},
        )

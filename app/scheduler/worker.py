from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from time import perf_counter

import httpx

from app.config import Settings
from app.core.serialization import to_sample_payload
from app.observability.metrics import Metrics
from app.parsers.errors import ParseError
from app.providers.base import ScheduleImageProvider
from app.storage.repository import IngestRunResult, SnapshotRepository


class IngestWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        provider: ScheduleImageProvider,
        parser,
        repository: SnapshotRepository,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.parser = parser
        self.repository = repository
        self.metrics = metrics

        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._logger = logging.getLogger("voe.ingest")

        self.last_run_status: str = "never"
        self.last_run_started_at: datetime | None = None
        self.last_run_finished_at: datetime | None = None
        self.last_error: str | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="ingest-worker")

    async def stop(self) -> None:
        if self._task is None:
            return

        self._stop_event.set()
        await self._task
        self._task = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def run_once(self) -> None:
        started = datetime.now(tz=timezone.utc)
        self.last_run_started_at = started
        region_id = self.settings.region_id
        stage = "fetch"
        status = "success"
        error: str | None = None
        timer_start = perf_counter()

        try:
            fetch_result = await self.provider.fetch_latest()
            region_id = fetch_result.region_id or self.settings.region_id

            image_bytes = await self._download_image(fetch_result.image_url)
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            previous_hash = self.repository.latest_image_hash(region_id)
            if previous_hash == image_hash:
                status = "no_change"
                return

            stage = "parse"
            parsed = self.parser.parse(
                image_bytes,
                region_id=region_id,
                fetched_at_utc=started,
                source_updated_at_utc=fetch_result.source_updated_at,
            )
            payload = to_sample_payload(parsed)

            stage = "store"
            self.repository.save_snapshot(
                region_id=region_id,
                day_unix=parsed.today_unix,
                last_updated_utc=parsed.last_updated_utc,
                fact_update_text=parsed.fact_update_text,
                payload=payload,
                image_url=fetch_result.image_url,
                image_sha256=image_hash,
            )
            self.repository.purge_old_snapshots(self.settings.retention_days)

            self.metrics.mark_snapshot_success(region_id, parsed.last_updated_utc)

        except ParseError as exc:
            status = "parse_error"
            error = str(exc)
            self._logger.exception("Parse error during ingest")
        except httpx.HTTPError as exc:
            status = "fetch_error"
            error = str(exc)
            self._logger.exception("Fetch error during ingest")
        except Exception as exc:  # pragma: no cover
            status = f"{stage}_error"
            error = str(exc)
            self._logger.exception("Unhandled ingest error")
        finally:
            finished = datetime.now(tz=timezone.utc)
            self.last_run_finished_at = finished
            self.last_run_status = status
            self.last_error = error

            self.metrics.mark_ingest_status(status)
            self.metrics.ingest_duration_seconds.observe(perf_counter() - timer_start)

            self.repository.record_ingest_run(
                region_id=region_id,
                started_at_utc=started,
                finished_at_utc=finished,
                result=IngestRunResult(status=status, error_message=error),
            )

    async def _run_loop(self) -> None:
        await self.run_once()

        while not self._stop_event.is_set():
            sleep_seconds = self._next_sleep_seconds()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_seconds)
            except TimeoutError:
                pass

            if self._stop_event.is_set():
                break
            await self.run_once()

    def _next_sleep_seconds(self) -> float:
        interval_seconds = max(self.settings.poll_interval_minutes, 1) * 60
        if not self.settings.poll_align_clock:
            return float(interval_seconds)

        now = datetime.now(tz=timezone.utc).timestamp()
        next_tick = ((int(now) // interval_seconds) + 1) * interval_seconds
        return max(next_tick - now, 1.0)

    async def _download_image(self, image_url: str) -> bytes:
        timeout = httpx.Timeout(self.settings.provider_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            return response.content

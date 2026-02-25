from __future__ import annotations

from datetime import datetime, timezone

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.ingest_runs_total = Counter(
            "voe_ingest_runs_total",
            "Total ingest runs by status",
            labelnames=("status",),
            registry=self.registry,
        )
        self.ingest_duration_seconds = Histogram(
            "voe_ingest_duration_seconds",
            "Duration of ingest runs in seconds",
            registry=self.registry,
        )
        self.snapshot_age_seconds = Gauge(
            "voe_snapshot_age_seconds",
            "Age of latest successful snapshot in seconds",
            labelnames=("region",),
            registry=self.registry,
        )
        self.last_success_epoch = Gauge(
            "voe_last_success_epoch_seconds",
            "Unix timestamp of last successful ingest run",
            labelnames=("region",),
            registry=self.registry,
        )

    def mark_ingest_status(self, status: str) -> None:
        self.ingest_runs_total.labels(status=status).inc()

    def mark_snapshot_success(self, region_id: str, captured_at_utc: datetime) -> None:
        timestamp = captured_at_utc.astimezone(timezone.utc).timestamp()
        now = datetime.now(tz=timezone.utc).timestamp()
        self.last_success_epoch.labels(region=region_id).set(timestamp)
        self.snapshot_age_seconds.labels(region=region_id).set(max(now - timestamp, 0.0))

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

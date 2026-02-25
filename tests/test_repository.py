from __future__ import annotations

from datetime import datetime, timezone

from app.core.serialization import to_sample_payload
from app.storage.repository import IngestRunResult, SnapshotRepository
from tests.helpers import sample_parsed_schedule


def test_repository_snapshot_roundtrip(tmp_path) -> None:
    db = tmp_path / "test.db"
    repo = SnapshotRepository(str(db))
    repo.init_db()

    parsed = sample_parsed_schedule()
    payload = to_sample_payload(parsed)

    repo.save_snapshot(
        region_id="Volyn",
        day_unix=parsed.today_unix,
        last_updated_utc=parsed.last_updated_utc,
        fact_update_text=parsed.fact_update_text,
        payload=payload,
        image_url="https://example.test/schedule.png",
        image_sha256="abc123",
    )

    latest = repo.get_latest_snapshot("Volyn")
    assert latest is not None
    assert latest["regionId"] == "Volyn"

    day = repo.get_day_snapshot("Volyn", parsed.today_unix)
    assert day is not None
    assert day["fact"]["today"] == parsed.today_unix

    assert repo.latest_image_hash("Volyn") == "abc123"


def test_repository_records_ingest_runs(tmp_path) -> None:
    db = tmp_path / "test.db"
    repo = SnapshotRepository(str(db))
    repo.init_db()

    now = datetime.now(tz=timezone.utc)
    repo.record_ingest_run(
        region_id="Volyn",
        started_at_utc=now,
        finished_at_utc=now,
        result=IngestRunResult(status="success"),
    )

    assert repo.ping() is True

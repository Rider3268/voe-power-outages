from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IngestRunResult:
    status: str
    error_message: str | None = None


class SnapshotRepository:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self._lock = threading.Lock()
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_id TEXT NOT NULL,
                    day_unix INTEGER NOT NULL,
                    last_updated_utc TEXT NOT NULL,
                    fact_update_text TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_region_created
                    ON snapshots(region_id, created_at_utc DESC);

                CREATE INDEX IF NOT EXISTS idx_snapshots_region_day
                    ON snapshots(region_id, day_unix);

                CREATE TABLE IF NOT EXISTS ingest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_id TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL,
                    finished_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_ingest_runs_region_started
                    ON ingest_runs(region_id, started_at_utc DESC);
                """
            )
            conn.commit()

    def ping(self) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("SELECT 1")
        return True

    def latest_image_hash(self, region_id: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT image_sha256
                FROM snapshots
                WHERE region_id = ?
                ORDER BY created_at_utc DESC
                LIMIT 1
                """,
                (region_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["image_sha256"])

    def save_snapshot(
        self,
        *,
        region_id: str,
        day_unix: int,
        last_updated_utc: datetime,
        fact_update_text: str,
        payload: dict[str, Any],
        image_url: str,
        image_sha256: str,
    ) -> int:
        created_at = datetime.now(tz=timezone.utc).isoformat()

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO snapshots(
                    region_id,
                    day_unix,
                    last_updated_utc,
                    fact_update_text,
                    raw_json,
                    image_url,
                    image_sha256,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    region_id,
                    day_unix,
                    last_updated_utc.astimezone(timezone.utc).isoformat(),
                    fact_update_text,
                    json.dumps(payload, ensure_ascii=False),
                    image_url,
                    image_sha256,
                    created_at,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def record_ingest_run(
        self,
        *,
        region_id: str,
        started_at_utc: datetime,
        finished_at_utc: datetime,
        result: IngestRunResult,
    ) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ingest_runs(
                    region_id,
                    started_at_utc,
                    finished_at_utc,
                    status,
                    error_message
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    region_id,
                    started_at_utc.astimezone(timezone.utc).isoformat(),
                    finished_at_utc.astimezone(timezone.utc).isoformat(),
                    result.status,
                    result.error_message,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_latest_snapshot(self, region_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT raw_json
                FROM snapshots
                WHERE region_id = ?
                ORDER BY created_at_utc DESC
                LIMIT 1
                """,
                (region_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["raw_json"]))

    def get_day_snapshot(self, region_id: str, day_unix: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT raw_json
                FROM snapshots
                WHERE region_id = ? AND day_unix = ?
                ORDER BY created_at_utc DESC
                LIMIT 1
                """,
                (region_id, day_unix),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["raw_json"]))

    def get_history(self, region_id: str, limit: int) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    region_id,
                    day_unix,
                    last_updated_utc,
                    fact_update_text,
                    image_url,
                    image_sha256,
                    created_at_utc,
                    raw_json
                FROM snapshots
                WHERE region_id = ?
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (region_id, limit),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "id": int(row["id"]),
                    "regionId": str(row["region_id"]),
                    "dayUnix": int(row["day_unix"]),
                    "lastUpdated": str(row["last_updated_utc"]),
                    "factUpdate": str(row["fact_update_text"]),
                    "imageUrl": str(row["image_url"]),
                    "imageSha256": str(row["image_sha256"]),
                    "createdAt": str(row["created_at_utc"]),
                    "payload": json.loads(str(row["raw_json"])),
                }
            )
        return result

    def purge_old_snapshots(self, retention_days: int) -> int:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=retention_days)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM snapshots WHERE created_at_utc < ?",
                (cutoff.astimezone(timezone.utc).isoformat(),),
            )
            conn.commit()
            return int(cursor.rowcount)

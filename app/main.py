from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import Settings, load_settings
from app.observability.metrics import Metrics
from app.parsers.volyn_schedule_parser import VolynScheduleParser
from app.providers.registry import build_provider
from app.scheduler.worker import IngestWorker
from app.storage.repository import SnapshotRepository


class NullWorker:
    last_run_status = "disabled"
    last_run_started_at = None
    last_run_finished_at = None
    last_error = None

    def is_running(self) -> bool:
        return False


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or load_settings()
    configure_logging(app_settings.log_level)

    repository = SnapshotRepository(app_settings.database_path)
    repository.init_db()

    parser = VolynScheduleParser(app_settings.parser_profile)
    provider = build_provider(app_settings)
    metrics = Metrics()

    worker = (
        IngestWorker(
            settings=app_settings,
            provider=provider,
            parser=parser,
            repository=repository,
            metrics=metrics,
        )
        if app_settings.enable_scheduler
        else None
    )

    app = FastAPI(title="voe-power-outages", version="0.1.0")
    app.state.settings = app_settings
    app.state.repository = repository
    app.state.metrics = metrics
    app.state.worker = worker if worker is not None else NullWorker()

    @app.on_event("startup")
    async def _on_startup() -> None:
        if worker is not None:
            await worker.start()

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        if worker is not None:
            await worker.stop()

    app.include_router(api_router)
    return app


app = create_app()

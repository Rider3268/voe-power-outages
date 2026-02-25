from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    worker = request.app.state.worker
    return {
        "status": "ok",
        "scheduler": {
            "enabled": request.app.state.settings.enable_scheduler,
            "running": worker.is_running() if worker else False,
            "lastRunStatus": worker.last_run_status if worker else "disabled",
            "lastRunStartedAt": worker.last_run_started_at.isoformat() if worker and worker.last_run_started_at else None,
            "lastRunFinishedAt": worker.last_run_finished_at.isoformat() if worker and worker.last_run_finished_at else None,
            "lastError": worker.last_error if worker else None,
        },
    }


@router.get("/readyz")
async def readyz(request: Request) -> dict:
    repository = request.app.state.repository
    try:
        repository.ping()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"database not ready: {exc}") from exc

    return {"status": "ready"}


@router.get("/v1/schedule/latest")
async def latest_schedule(
    request: Request,
    region_id: str = Query(alias="regionId", default="Volyn"),
) -> dict:
    payload = request.app.state.repository.get_latest_snapshot(region_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No schedule snapshot available")
    return payload


@router.get("/v1/schedule/day")
async def schedule_day(
    request: Request,
    day_unix: int = Query(alias="dayUnix"),
    region_id: str = Query(alias="regionId", default="Volyn"),
) -> dict:
    payload = request.app.state.repository.get_day_snapshot(region_id, day_unix)
    if payload is None:
        raise HTTPException(status_code=404, detail="No schedule snapshot for requested day")
    return payload


@router.get("/v1/schedule/history")
async def schedule_history(
    request: Request,
    region_id: str = Query(alias="regionId", default="Volyn"),
    limit: int = Query(default=48, ge=1, le=500),
) -> dict:
    history = request.app.state.repository.get_history(region_id, limit)
    return {
        "regionId": region_id,
        "count": len(history),
        "items": history,
    }


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    payload, content_type = request.app.state.metrics.render()
    return Response(content=payload, media_type=content_type)

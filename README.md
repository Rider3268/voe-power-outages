# voe-power-outages

Python service that fetches Volynoblenergo outage schedule images, parses them into a sample-compatible JSON schema, stores snapshots in SQLite, and serves API endpoints for latest and historical schedule data.

## Features

- Deterministic OpenCV-first image parser for GPV table schedules.
- Modular provider interface for fetching latest schedule image URLs.
- SQLite persistence with 30-day retention.
- In-process 30-minute scheduler (24/7 polling).
- FastAPI endpoints (`latest`, `day`, `history`, `healthz`, `readyz`, `metrics`).
- Dockerfile for lightweight container deployment.
- Helm chart in `charts/voe-power-outages` with:
  - ingress disabled by default
  - optional ingress enablement
  - optional Gateway API `HTTPRoute`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

See `.env.example` for defaults. Critical variables:

- `PROVIDER_METADATA_URL`: options endpoint (for Volyn: `https://api-voe-poweron.inneti.net/api/options`).
- `PROVIDER_OPTION_KEY`: key used in request query (for Volyn: `pw_gpv_image_today`).
- `PROVIDER_IMAGE_URL_PATH`: optional dot-path to image URI in payload; if empty provider auto-discovers `/media/..._GPV.png`.
- `PROVIDER_UPDATED_AT_PATH`: optional dot-path to upload/update timestamp; if empty provider auto-discovers timestamp fields.
- `DATABASE_PATH`: SQLite path (default `./data/voe.db` locally, `/data/voe.db` in container/chart).
- `POLL_INTERVAL_MINUTES`: scheduler interval (default `30`).

## API

- `GET /v1/schedule/latest?regionId=Volyn`
- `GET /v1/schedule/day?regionId=Volyn&dayUnix=1771365600`
- `GET /v1/schedule/history?regionId=Volyn&limit=48`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

## Helm

```bash
helm lint charts/voe-power-outages
helm install voe-power-outages charts/voe-power-outages
```

Internal-only defaults are applied unless ingress or gateway are explicitly enabled in values.

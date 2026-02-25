.PHONY: install run test lint helm-lint helm-template

install:
	pip install -e .[dev]

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check app tests

helm-lint:
	helm lint charts/voe-power-outages

helm-template:
	helm template voe-power-outages charts/voe-power-outages

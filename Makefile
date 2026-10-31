PY ?= python

.PHONY: install download train evaluate test lint serve docker drift

install:
	$(PY) -m pip install -r requirements-dev.txt

download:
	$(PY) -m data.download_data

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

lint:
	ruff check .

test:
	$(PY) -m pytest

serve:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

drift:
	$(PY) -m src.drift $(CSV)

.PHONY: help install-dev start stop restart dev test format lint clean

help:
	@echo "OpticalRAG Makefile"
	@echo "  make install-dev  - Install dev dependencies"
	@echo "  make dev          - Dev server (hot reload)"
	@echo "  make start        - Production server"
	@echo "  make test         - Run tests"
	@echo "  make format       - Format code"
	@echo "  make lint         - Lint code"
	@echo "  make clean        - Clean temp files"

install-dev:
	pip install -e ".[dev]"

dev:
	.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 9900

start:
	.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

stop:
	pkill -f "uvicorn app.main:app" 2>/dev/null || true

restart: stop start

test:
	python3 -m pytest tests/ -v --cov=app --cov-report=term-missing

format:
	python3 -m ruff check --select I --fix app/ 2>/dev/null || true
	python3 -m ruff format app/ 2>/dev/null || python3 -m black app/

lint:
	python3 -m ruff check app/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage

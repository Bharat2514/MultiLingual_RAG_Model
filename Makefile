.DEFAULT_GOAL := help
SHELL := /bin/bash

PY      := $(CURDIR)/.venv/bin/python
PIP     := $(CURDIR)/.venv/bin/pip
BACKEND := $(CURDIR)/backend

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- setup -------------------------------------------------------------------

.PHONY: setup
setup: venv install env ## One-shot local setup (venv + deps + .env)
	@echo "✓ Setup complete. Next: make infra && make dev"

.PHONY: venv
venv: ## Create the Python virtualenv
	@test -d .venv || python3 -m venv .venv
	@$(PIP) install --quiet --upgrade pip setuptools wheel

.PHONY: install
install: venv ## Install backend + dev dependencies (no ML extras)
	@$(PIP) install -e "$(BACKEND)[dev]"

.PHONY: install-ml
install-ml: venv ## Install the embedding/reranker stack (torch, FlagEmbedding)
	@$(PIP) install -e "$(BACKEND)[ml]"

.PHONY: install-eval
install-eval: venv ## Install the evaluation harness (RAGAS, pandas)
	@$(PIP) install -e "$(BACKEND)[eval]"

.PHONY: env
env: ## Create .env from the template if absent
	@test -f .env || (cp .env.example .env && echo "✓ Created .env — add ANTHROPIC_API_KEY")

.PHONY: setup-mac
setup-mac: ## Install macOS system deps (Tesseract with Hindi)
	@bash scripts/setup_mac.sh

# --- infrastructure ----------------------------------------------------------

.PHONY: infra
infra: ## Start Qdrant + Postgres + Redis (app runs on the host, uses MPS)
	@docker compose up -d postgres redis qdrant
	@$(MAKE) --no-print-directory wait-infra

.PHONY: wait-infra
wait-infra: ## Block until infra containers report healthy
	@echo "Waiting for infrastructure..."
	@for i in $$(seq 1 60); do \
		unhealthy=$$(docker compose ps --format '{{.Service}} {{.Health}}' \
			| grep -E 'postgres|redis|qdrant' | grep -vc healthy || true); \
		if [ "$$unhealthy" = "0" ]; then echo "✓ Infrastructure healthy"; exit 0; fi; \
		sleep 2; \
	done; \
	echo "✗ Timed out waiting for infrastructure"; docker compose ps; exit 1

.PHONY: up
up: ## Start the full stack in containers (CPU torch)
	@docker compose --profile app up -d --build

.PHONY: down
down: ## Stop all containers
	@docker compose --profile app down

.PHONY: clean
clean: ## Stop containers and delete all volumes (DESTRUCTIVE)
	@docker compose --profile app down -v

.PHONY: logs
logs: ## Tail container logs
	@docker compose --profile app logs -f --tail=100

.PHONY: ps
ps: ## Show container status
	@docker compose ps

# --- development -------------------------------------------------------------

.PHONY: dev
dev: ## Run the API on the host with auto-reload
	@cd $(BACKEND) && $(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: worker
# --pool=solo: Celery's default `prefork` pool forks its workers, and Apple's
# Metal stack does not survive fork() — the first MPS kernel aborts the child
# with SIGABRT. `solo` keeps the model in the main process so the host worker
# gets GPU acceleration. The Linux container uses prefork (see the Dockerfile),
# where CPU torch forks fine and concurrency is worth more.
worker: ## Run the ingestion worker on the host (solo pool, MPS-safe)
	@cd $(BACKEND) && $(PY) -m celery -A app.workers.celery_app:celery_app worker \
		--loglevel=INFO --pool=solo -Q ingestion

.PHONY: web
web: ## Run the Next.js dev server on :3000
	@cd frontend && npm run dev

.PHONY: web-install
web-install: ## Install frontend dependencies
	@cd frontend && npm install --no-audit --no-fund

.PHONY: web-build
web-build: ## Production build of the frontend
	@cd frontend && npm run build

.PHONY: web-lint
web-lint: ## Lint and typecheck the frontend
	@cd frontend && npm run lint && npm run typecheck

.PHONY: db-init
db-init: ## Create database tables
	@cd $(BACKEND) && $(PY) -m app.db.init_db

# --- quality -----------------------------------------------------------------

.PHONY: test
test: ## Run unit tests
	@cd $(BACKEND) && $(PY) -m pytest -m "not integration and not ml and not slow"

.PHONY: test-all
test-all: ## Run the full test suite (needs infra up)
	@cd $(BACKEND) && $(PY) -m pytest

.PHONY: cov
cov: ## Run tests with a coverage report
	@cd $(BACKEND) && $(PY) -m pytest --cov=app --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Lint and type-check (backend, scripts, eval)
	@cd $(BACKEND) && $(PY) -m ruff check . ../scripts ../eval \
		&& $(PY) -m ruff format --check . ../scripts ../eval \
		&& $(PY) -m mypy app

.PHONY: fmt
fmt: ## Auto-format and fix lint errors
	@cd $(BACKEND) && $(PY) -m ruff format . ../scripts ../eval \
		&& $(PY) -m ruff check --fix . ../scripts ../eval

.PHONY: download-models
download-models: ## Pre-fetch bge-m3 + reranker into ./data/models (~2.3 GB)
	@$(PY) scripts/download_models.py

.PHONY: probe-crosslingual
probe-crosslingual: ## Verify Hindi<->English retrieval with the real models
	@$(PY) eval/crosslingual_probe.py

.PHONY: check-llm
check-llm: ## Verify the LLM key, list reachable models, probe Hindi + Hinglish
	@$(PY) scripts/check_llm.py

.PHONY: health
health: ## Curl the readiness endpoint
	@curl -fsS http://localhost:8000/api/v1/health/ready | $(PY) -m json.tool

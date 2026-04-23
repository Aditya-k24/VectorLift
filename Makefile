# =============================================================================
# VectorLift — Makefile
# =============================================================================
# Conventions:
#   • All paths are relative to the repo root (where this Makefile lives).
#   • Phony targets are explicitly declared.
#   • $(MAKE) is used for recursive make calls so flags propagate correctly.
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
PYTHON        ?= python3
PIP           ?= $(PYTHON) -m pip
VENV          ?= .venv
VENV_BIN      := $(VENV)/bin
VENV_PYTHON   := $(VENV_BIN)/python
VENV_PIP      := $(VENV_BIN)/pip

APP_MODULE    := apps.api.main:app
DOCKER_COMPOSE := docker compose
COMPOSE_FILE  := infra/docker/docker-compose.yml
COMPOSE_FILE_DEV := infra/docker/docker-compose.dev.yml

# Test paths
TEST_UNIT    := tests/unit
TEST_INT     := tests/integration
TEST_E2E     := tests/e2e

# Dataset paths
DATA_DIR     := data
MODELS_DIR   := models

# Colors
RESET  := \033[0m
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m
RED    := \033[31m

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@printf "$(BOLD)$(CYAN)VectorLift — Semantic Search + Ranking Engine$(RESET)\n\n"
	@printf "$(BOLD)Usage:$(RESET) make $(CYAN)<target>$(RESET)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
.PHONY: setup
setup: ## Create virtualenv and install all dependencies (dev mode)
	@printf "$(BOLD)$(GREEN)Creating virtual environment...$(RESET)\n"
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	$(MAKE) install-dev
	$(MAKE) pre-commit-install
	@printf "$(BOLD)$(GREEN)Setup complete. Activate with: source $(VENV)/bin/activate$(RESET)\n"

.PHONY: install
install: ## Install production dependencies only
	@printf "$(BOLD)$(GREEN)Installing production dependencies...$(RESET)\n"
	$(VENV_PIP) install -e "."

.PHONY: install-dev
install-dev: ## Install all dependencies including dev extras
	@printf "$(BOLD)$(GREEN)Installing dev dependencies...$(RESET)\n"
	$(VENV_PIP) install -e ".[dev]"

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	$(VENV_BIN)/pre-commit install
	$(VENV_BIN)/pre-commit install --hook-type commit-msg

# ---------------------------------------------------------------------------
# Lint, Format, Typecheck
# ---------------------------------------------------------------------------
.PHONY: lint
lint: ## Run ruff linter
	@printf "$(BOLD)$(CYAN)Running ruff...$(RESET)\n"
	$(VENV_BIN)/ruff check . --fix

.PHONY: format
format: ## Run black formatter
	@printf "$(BOLD)$(CYAN)Running black...$(RESET)\n"
	$(VENV_BIN)/black .

.PHONY: format-check
format-check: ## Check formatting without applying changes
	$(VENV_BIN)/black --check --diff .
	$(VENV_BIN)/ruff check .

.PHONY: typecheck
typecheck: ## Run mypy type checker
	@printf "$(BOLD)$(CYAN)Running mypy...$(RESET)\n"
	$(VENV_BIN)/mypy core retrieval apps pipelines \
		--ignore-missing-imports \
		--show-error-codes

.PHONY: quality
quality: lint format-check typecheck ## Run all code quality checks

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run all tests (unit + integration + e2e)
	@printf "$(BOLD)$(CYAN)Running all tests...$(RESET)\n"
	$(VENV_BIN)/pytest $(TEST_UNIT) $(TEST_INT) $(TEST_E2E) -v

.PHONY: test-unit
test-unit: ## Run unit tests only (no external services required)
	@printf "$(BOLD)$(CYAN)Running unit tests...$(RESET)\n"
	$(VENV_BIN)/pytest $(TEST_UNIT) -v -m unit \
		--cov=core --cov=retrieval --cov=apps \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/unit

.PHONY: test-integration
test-integration: ## Run integration tests (requires live Docker services)
	@printf "$(BOLD)$(CYAN)Running integration tests...$(RESET)\n"
	$(VENV_BIN)/pytest $(TEST_INT) -v -m integration \
		--cov=core --cov=retrieval \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/integration

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests against the live API
	@printf "$(BOLD)$(CYAN)Running e2e tests...$(RESET)\n"
	$(VENV_BIN)/pytest $(TEST_E2E) -v -m e2e

.PHONY: test-watch
test-watch: ## Re-run unit tests on file changes (requires pytest-watch)
	$(VENV_BIN)/ptw $(TEST_UNIT) -- -v -m unit

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: docker-build
docker-build: ## Build Docker images
	@printf "$(BOLD)$(CYAN)Building Docker images...$(RESET)\n"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build --parallel

.PHONY: docker-up
docker-up: ## Start all services (detached)
	@printf "$(BOLD)$(GREEN)Starting services...$(RESET)\n"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d
	@printf "$(GREEN)Services started. Run 'make docker-logs' to tail logs.$(RESET)\n"

.PHONY: docker-up-dev
docker-up-dev: ## Start services in dev mode (with hot-reload volumes)
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(COMPOSE_FILE_DEV) up -d

.PHONY: docker-down
docker-down: ## Stop and remove containers (preserves volumes)
	@printf "$(BOLD)$(YELLOW)Stopping services...$(RESET)\n"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down

.PHONY: docker-down-volumes
docker-down-volumes: ## Stop containers AND remove volumes (destructive)
	@printf "$(BOLD)$(RED)Removing containers and volumes...$(RESET)\n"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v

.PHONY: docker-logs
docker-logs: ## Tail logs for all services
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f --tail=100

.PHONY: docker-logs-api
docker-logs-api: ## Tail API service logs only
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f --tail=100 api

.PHONY: docker-ps
docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

.PHONY: docker-restart
docker-restart: ## Restart all services
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) restart

# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
.PHONY: ingest-sample
ingest-sample: ## Ingest a small sample dataset (DATASET_MODE=dev)
	@printf "$(BOLD)$(CYAN)Ingesting sample data (dev mode)...$(RESET)\n"
	DATASET_MODE=dev $(VENV_PYTHON) -m pipelines.ingestion.ingest \
		--mode dev \
		--output-dir $(DATA_DIR)/processed

.PHONY: ingest-full
ingest-full: ## Ingest the full dataset (DATASET_MODE=full — slow)
	@printf "$(BOLD)$(YELLOW)Ingesting full dataset — this may take a while...$(RESET)\n"
	DATASET_MODE=full $(VENV_PYTHON) -m pipelines.ingestion.ingest \
		--mode full \
		--output-dir $(DATA_DIR)/processed

.PHONY: ingest-small
ingest-small: ## Ingest the small dataset (DATASET_MODE=small)
	DATASET_MODE=small $(VENV_PYTHON) -m pipelines.ingestion.ingest \
		--mode small \
		--output-dir $(DATA_DIR)/processed

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
.PHONY: index-bm25
index-bm25: ## Build BM25 index over processed corpus
	@printf "$(BOLD)$(CYAN)Building BM25 index...$(RESET)\n"
	$(VENV_PYTHON) -m retrieval.bm25.indexer \
		--input-dir $(DATA_DIR)/processed \
		--index-dir $(DATA_DIR)/indexes/bm25

.PHONY: index-dense
index-dense: ## Build dense (vector) index using current bi-encoder model
	@printf "$(BOLD)$(CYAN)Building dense vector index...$(RESET)\n"
	$(VENV_PYTHON) -m retrieval.dense.indexer \
		--input-dir $(DATA_DIR)/processed \
		--index-dir $(DATA_DIR)/indexes/dense \
		--model-path $(MODELS_DIR)/biencoder

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
.PHONY: train-biencoder
train-biencoder: ## Fine-tune the bi-encoder model
	@printf "$(BOLD)$(CYAN)Training bi-encoder...$(RESET)\n"
	$(VENV_PYTHON) -m pipelines.training.train_biencoder \
		--data-dir $(DATA_DIR)/training \
		--output-dir $(MODELS_DIR)/biencoder \
		--epochs 3 \
		--batch-size 32

.PHONY: train-reranker
train-reranker: ## Fine-tune the cross-encoder reranker model
	@printf "$(BOLD)$(CYAN)Training cross-encoder reranker...$(RESET)\n"
	$(VENV_PYTHON) -m pipelines.training.train_reranker \
		--data-dir $(DATA_DIR)/training \
		--output-dir $(MODELS_DIR)/reranker \
		--epochs 5 \
		--batch-size 16

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
.PHONY: evaluate
evaluate: ## Evaluate default (hybrid+rerank) retrieval pipeline
	@printf "$(BOLD)$(CYAN)Running evaluation...$(RESET)\n"
	$(VENV_PYTHON) -m experiments.evaluate \
		--mode hybrid \
		--dataset-mode dev \
		--output-dir experiments/results

.PHONY: evaluate-all
evaluate-all: ## Evaluate all retrieval modes (bm25, dense, hybrid, rerank)
	@printf "$(BOLD)$(CYAN)Running full evaluation suite...$(RESET)\n"
	for mode in bm25 dense hybrid rerank; do \
		printf "$(CYAN)Evaluating $$mode...$(RESET)\n"; \
		$(VENV_PYTHON) -m experiments.evaluate \
			--mode $$mode \
			--dataset-mode small \
			--output-dir experiments/results/$$mode; \
	done
	$(VENV_PYTHON) -m experiments.compare \
		--results-dir experiments/results \
		--output experiments/results/comparison.json

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
.PHONY: run-api
run-api: ## Start the FastAPI development server
	$(VENV_BIN)/uvicorn $(APP_MODULE) \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--reload-dir core \
		--reload-dir retrieval \
		--reload-dir apps \
		--log-level info

.PHONY: run-api-prod
run-api-prod: ## Start the API with Gunicorn + Uvicorn workers
	$(VENV_BIN)/gunicorn $(APP_MODULE) \
		-k uvicorn.workers.UvicornWorker \
		--workers 4 \
		--bind 0.0.0.0:8000 \
		--timeout 120 \
		--graceful-timeout 30 \
		--access-logfile -

.PHONY: run-dashboard
run-dashboard: ## Start the Streamlit analytics dashboard
	$(VENV_BIN)/streamlit run apps/dashboard/app.py \
		--server.port 8501 \
		--server.address 0.0.0.0

# ---------------------------------------------------------------------------
# Database migrations
# ---------------------------------------------------------------------------
.PHONY: db-migrate
db-migrate: ## Run Alembic migrations to head
	$(VENV_BIN)/alembic upgrade head

.PHONY: db-rollback
db-rollback: ## Roll back the last Alembic migration
	$(VENV_BIN)/alembic downgrade -1

.PHONY: db-revision
db-revision: ## Create a new Alembic migration (MSG="description")
	$(VENV_BIN)/alembic revision --autogenerate -m "$(MSG)"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove all build artifacts, caches, and temp files
	@printf "$(BOLD)$(YELLOW)Cleaning build artifacts...$(RESET)\n"
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	@printf "$(GREEN)Clean complete.$(RESET)\n"

.PHONY: clean-models
clean-models: ## Remove downloaded/trained model checkpoints
	@printf "$(BOLD)$(RED)Removing model files...$(RESET)\n"
	rm -rf $(MODELS_DIR)/biencoder $(MODELS_DIR)/reranker

.PHONY: clean-data
clean-data: ## Remove processed data and indexes (not raw data)
	@printf "$(BOLD)$(RED)Removing processed data and indexes...$(RESET)\n"
	rm -rf $(DATA_DIR)/processed $(DATA_DIR)/indexes

.PHONY: clean-all
clean-all: clean clean-models clean-data ## Remove everything (build + models + data)

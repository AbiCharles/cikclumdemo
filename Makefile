# Makefile — A2A Prior Auth Demo

PORT ?= 8080
QPORT ?= 6333
TEST_PATIENT ?= P001
TEST_DRUG ?= Humira
TRIALS ?= 5
WARMUP ?= 1

.PHONY: help up build up-nc logs down restart test health open shell ps clean \
        reingest bench qdrant cards

help:
	@echo "Common commands:"
	@echo "  make up         - Build and start app+qdrant"
	@echo "  make build      - Build app image (cache ok)"
	@echo "  make up-nc      - Build with --no-cache and start"
	@echo "  make logs       - Tail logs"
	@echo "  make down       - Stop containers"
	@echo "  make restart    - Restart services"
	@echo "  make test       - Run CLI test (TEST_PATIENT / TEST_DRUG)"
	@echo "  make bench      - Run latency benchmark (TRIALS / WARMUP)"
	@echo "  make reingest   - Re-create embeddings & load synthetic corpus"
	@echo "  make qdrant     - Open Qdrant dashboard"
	@echo "  make cards      - Show agent cards (retrieval/summarization)"
	@echo "  make health     - GET /health"
	@echo "  make open       - Open UI in browser"
	@echo "  make shell      - Shell into app container"
	@echo "  make ps         - Show running containers"
	@echo "  make clean      - Down + prune build cache"

up:
	@echo ">> Starting stack on port $(PORT)"
	docker compose up --build

build:
	docker compose build

up-nc:
	docker compose build --no-cache
	docker compose up

logs:
	docker compose logs -f

down:
	docker compose down

restart:
	docker compose down
	docker compose up --build

test:
	@echo ">> Testing summarization agent inside container"
	docker compose exec app python -m data.test --patient $(TEST_PATIENT) --drug "$(TEST_DRUG)"

bench:
	@echo ">> Benchmarking (FAST vs REFLECTION) for $(TEST_PATIENT), $(TEST_DRUG)"
	docker compose exec app python -m data.bench --patient $(TEST_PATIENT) --drug "$(TEST_DRUG)" --trials $(TRIALS) --warmup $(WARMUP)

reingest:
	@echo ">> Re-ingesting synthetic corpus into Qdrant"
	docker compose exec app python -m data.debug_qdrant --reingest --limit 10

qdrant:
	@echo ">> Opening Qdrant dashboard http://localhost:$(QPORT)/dashboard"
	-open "http://localhost:$(QPORT)/dashboard" >/dev/null 2>&1 || xdg-open "http://localhost:$(QPORT)/dashboard" || true

cards:
	@echo ">> Retrieval agent card"
	@curl -s http://localhost:$(PORT)/agents/retrieval/.well-known/agent-card.json | jq . || curl -s http://localhost:$(PORT)/agents/retrieval/.well-known/agent-card.json
	@echo ""
	@echo ">> Summarization agent card"
	@curl -s http://localhost:$(PORT)/agents/summarization/.well-known/agent-card.json | jq . || curl -s http://localhost:$(PORT)/agents/summarization/.well-known/agent-card.json

health:
	curl -s http://localhost:$(PORT)/health | jq . || curl -s http://localhost:$(PORT)/health

open:
	@echo ">> Opening UI at http://localhost:$(PORT)/ui"
	-open "http://localhost:$(PORT)/ui" >/dev/null 2>&1 || xdg-open "http://localhost:$(PORT)/ui" || true

shell:
	docker compose exec app /bin/sh

ps:
	docker compose ps

clean:
	@echo ">> Stopping containers and pruning build cache"
	docker compose down
	docker builder prune -f

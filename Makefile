PORT ?= 8080
TEST_PATIENT ?= P001
TEST_DRUG ?= Humira

.PHONY: help up build up-nc logs down restart test health open shell ps clean

help:
	@echo "Common commands:"
	@echo "  make up         - Build and start app+qdrant"
	@echo "  make build      - Build app image (cache ok)"
	@echo "  make up-nc      - Build with --no-cache and start"
	@echo "  make logs       - Tail logs"
	@echo "  make down       - Stop containers"
	@echo "  make restart    - Restart services"
	@echo "  make test       - Run CLI test (TEST_PATIENT / TEST_DRUG)"
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
	@echo ">> Testing summarization agent"
	python -m data.test --patient $(TEST_PATIENT) --drug $(TEST_DRUG)

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
	docker compose down
	docker


.PHONY: help venv install db-up db-down db-reset db-shell schema ingest verify test \
        embed sql-smoke sql semantic hybrid ask ask-explain phase3-verify rechunk

# Pull .env in so targets can use POSTGRES_* / READONLY_* without duplicating
# them here. The leading '-' means "don't fail if .env doesn't exist yet".
-include .env
export

# src-layout: PYTHONPATH avoids needing an editable install, which keeps the
# setup to two commands for anyone cloning this.
PY := PYTHONPATH=src .venv/bin/python
PIP := .venv/bin/pip
READONLY_PASSWORD ?= change_me_readonly

help:
	@echo "TrialSage"
	@echo "  make install    - create .venv and install dependencies"
	@echo "  make db-up      - start Postgres+pgvector, wait until healthy, apply schema"
	@echo "  make db-down    - stop the database (data is preserved)"
	@echo "  make db-reset   - DESTROY the database volume and rebuild from scratch"
	@echo "  make db-shell   - open a psql prompt inside the container"
	@echo "  make ingest     - fetch + parse + load (default: diabetes; AREA=oncology)"
	@echo "  make verify     - print row counts and sample rows"
	@echo "  make test       - run the test suite"
	@echo ""
	@echo "  make embed      - embed eligibility criteria + build the HNSW index"
	@echo "  make sql-smoke  - measure text-to-SQL accuracy against gold SQL"
	@echo "  make sql   Q='...'  - run one text-to-SQL question"
	@echo "  make semantic Q='...'  - run one semantic search"
	@echo "  make ask   Q='...'  - full pipeline: route -> retrieve -> cited answer"
	@echo "  make ask-explain Q='...'  - same, showing the routing decision"
	@echo "  make phase3-verify  - the three example questions, end to end"

venv:
	@test -d .venv || python3 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env from template")

db-up:
	docker compose up -d
	@echo "Waiting for Postgres to become healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' trialsage-db 2>/dev/null)" = "healthy" ]; do sleep 1; done
	@echo "Database healthy."
	@$(MAKE) schema

db-down:
	docker compose down

db-reset:
	docker compose down -v
	@$(MAKE) db-up

db-shell:
	docker compose exec db psql -U $(or $(POSTGRES_USER),trialsage) -d $(or $(POSTGRES_DB),trialsage)

# Applies every migration in order. All of them are idempotent, so re-running
# is safe and is how we pick up schema changes during development.
schema:
	@for f in sql/*.sql; do \
		echo "  applying $$f"; \
		docker compose exec -T db psql -v ON_ERROR_STOP=1 -q \
			-v ro_password="$(READONLY_PASSWORD)" \
			-U $(or $(POSTGRES_USER),trialsage) -d $(or $(POSTGRES_DB),trialsage) < $$f || exit 1; \
	done
	@echo "Schema applied."

ingest:
	$(PY) -m trialsage.ingest.run --area $(or $(AREA),diabetes)

verify:
	$(PY) -m trialsage.ingest.verify

test:
	.venv/bin/pytest -q

embed:
	$(PY) -m trialsage.embed.build_index

sql-smoke:
	PYTHONPATH=src:. .venv/bin/python -m eval.sql_smoke

sql:
	$(PY) -m trialsage.retrieval.cli sql "$(Q)"

semantic:
	$(PY) -m trialsage.retrieval.cli semantic "$(Q)"

ask:
	$(PY) -m trialsage.ask "$(Q)"

ask-explain:
	$(PY) -m trialsage.ask "$(Q)" --explain

phase3-verify:
	PYTHONPATH=src:. .venv/bin/python -m eval.phase3_verify

rechunk:
	$(PY) -m trialsage.ingest.rechunk

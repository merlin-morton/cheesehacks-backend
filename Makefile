# Align backend – build and run
# Use: make [target]. On Windows use Git Bash or: powershell -File build.ps1 [target]

PYTHON   ?= python
PIP      ?= pip
MYSQL    ?= mysql
HOST     ?= 127.0.0.1
PORT     ?= 8000

.PHONY: install db-setup run mysql help

help:
	@echo "Targets:"
	@echo "  make install   - Install Python dependencies"
	@echo "  make db-setup - Create DB and tables (uses MYSQL_* env or .env)"
	@echo "  make run      - Start FastAPI server (uvicorn)"
	@echo "  make mysql    - Open MySQL CLI for MYSQL_DATABASE"

install:
	$(PIP) install -r requirements.txt

db-setup:
	@echo "Applying schema.sql..."
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	$(MYSQL) -h "$${MYSQL_HOST:-localhost}" -u "$${MYSQL_USER:-root}" -p"$${MYSQL_PASSWORD:-}" < schema.sql
	@echo "Database ready."

run:
	$(PYTHON) -m uvicorn routes:app --host $(HOST) --port $(PORT) --reload

mysql:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	$(MYSQL) -h "$${MYSQL_HOST:-localhost}" -u "$${MYSQL_USER:-root}" -p"$${MYSQL_PASSWORD:-}" "$${MYSQL_DATABASE:-align}"

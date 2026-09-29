# NETRA -- SIH PS 127
# Everything runs through here so there is one way to do each thing.

PY   ?= python3
VENV := .venv
BIN  := $(VENV)/bin
RUN  := PYTHONPATH=backend $(BIN)/python -m app.cli

.PHONY: help setup dash-setup warm doctor preflight seed cache-tiles sample cameras preview detect track process run resolve journey analytics eval report api dash demo db stop test clean export import remote bundle-src

help:
	@echo "BEFORE THE DEMO"
	@echo "  make doctor           preflight check -- run this first, and again before you present"
	@echo "  make preflight        judge every video: ready / placeholder / corrupt / unsupported"
	@echo "  make warm             download every model up front (no surprise downloads live)"
	@echo "  make cache-tiles      pre-download map tiles so the demo survives venue wifi"
	@echo "  make seed             fill the database so the dashboard opens populated"
	@echo "  make report           regenerate docs/RESULTS.md — the numbers for the deck"
	@echo ""
	@echo "THE DEMO"
	@echo "  make demo             seed if empty, start workers, start the API"
	@echo "  make dash             the dashboard on :5173"
	@echo "  make stop             stop background workers"
	@echo ""
	@echo "EVERYTHING ELSE"
	@echo "make setup              create .venv and install backend deps"
	@echo "make sample             generate synthetic test footage into data/videos"
	@echo "make cameras            list configured cameras"
	@echo "make preview CAM=CAM01  play a camera feed with the live HUD"
	@echo "make detect CAM=CAM01   run vehicle detection on a feed"
	@echo "make track CAM=CAM01    detect + track + count line crossings"
	@echo "make process CAM=CAM01  full pipeline: detect, track, count, persist"
	@echo "make db                 show what is in the event store"
	@echo "make run                all cameras at once, one process each"
	@echo "make resolve            link sightings across cameras into vehicles"
	@echo "make journey PLATE=X    show a vehicle\x27s reconstructed trip"
	@echo "make analytics          traffic metrics, congestion and alerts"
	@echo "make eval               score association against your ground truth"
	@echo "make api                start the FastAPI backend on :8000"
	@echo "make dash               start the dashboard on :5173"
	@echo "make demo               workers in the background + API in front"
	@echo "make test               run the logic tests (no video needed)"
	@echo "make clean              remove venv and __pycache__"
	@echo ""
	@echo "PROCESSING SOMEWHERE MORE POWERFUL"
	@echo "make bundle-src         zip the project (no venv, no data) to carry to the other machine"
	@echo "make remote             run the whole batch headlessly here and export the results"
	@echo "make export             pack this machine's database + evidence into one archive"
	@echo "make import F=x.tar.gz  merge an archive produced on another machine"

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r backend/requirements.txt
	@echo "\nready. next: make sample"

sample:
	$(RUN) make-sample

cameras:
	$(RUN) cameras

preview:
	@test -n "$(CAM)" || (echo "usage: make preview CAM=CAM01"; exit 1)
	$(RUN) preview --camera $(CAM)

detect:
	@test -n "$(CAM)" || (echo "usage: make detect CAM=CAM01"; exit 1)
	$(RUN) detect --camera $(CAM)

track:
	@test -n "$(CAM)" || (echo "usage: make track CAM=CAM01"; exit 1)
	$(RUN) track --camera $(CAM)

process:
	@test -n "$(CAM)" || (echo "usage: make process CAM=CAM01"; exit 1)
	$(RUN) process --camera $(CAM)

run:
	$(RUN) run

resolve:
	$(RUN) resolve

journey:
	$(RUN) journey $(if $(PLATE),--plate $(PLATE),)

analytics:
	$(RUN) analytics --verbose

report:
	PYTHONPATH=backend $(BIN)/python backend/scripts/report.py --write

eval:
	PYTHONPATH=backend $(BIN)/python backend/scripts/eval_association.py

api:
	$(RUN) serve

dash-setup:
	cd frontend && npm install

dash:
	cd frontend && npm run dev

# Three terminals is clearer, but this is the one-command version: workers get
# backgrounded into data/logs, the API runs in front. Ctrl-C stops the API;
# 'make stop' kills the workers.
# One command. Seeds if the database is empty, starts the workers in the
# background, then runs the API in front. Ctrl-C stops the API; `make stop`
# stops the workers.
demo:
	@mkdir -p data/logs
	@$(RUN) doctor || true
	@echo ""
	@$(BIN)/python -c "import sys; sys.path.insert(0,'backend'); \
	from app.core.config import load_config; from app.store.db import get_sessionmaker, init_db; \
	from app.store.repository import stats; c=load_config(); init_db(c.paths.database); \
	s=get_sessionmaker(c.paths.database)(); sys.exit(0 if stats(s)['sightings'] else 1)" \
	|| $(MAKE) seed
	@echo "\nstarting camera workers in the background..."
	@$(RUN) run > data/logs/workers.log 2>&1 &
	@echo "workers warming up -- models take 15-20s to load."
	@echo "tail data/logs/workers.log to watch. dashboard: make dash\n"
	@sleep 3
	$(RUN) serve

doctor:
	$(RUN) doctor

preflight:
	$(RUN) preflight

# --- processing somewhere more powerful ------------------------------------
# The same four commands the laptop runs, run headlessly and packed up at the
# end. Nothing about the project changes; only where it executes.
remote:
	PYTHONPATH=backend $(BIN)/python backend/scripts/remote_batch.py \
	  $(if $(WORKERS),--workers $(WORKERS),) $(if $(FRAMES),--max-frames $(FRAMES),)

export:
	$(RUN) export

import:
	@test -n "$(F)" || (echo "usage: make import F=data/exports/netra-....tar.gz"; exit 1)
	$(RUN) import $(F)

# What to carry to the other machine: the code and the configs, never the
# venv, node_modules, the database or the footage.
bundle-src:
	@mkdir -p data/exports
	zip -qr data/exports/netra-src.zip backend configs frontend notebooks docs Makefile \
	  README.md QUICKSTART.md \
	  -x '*/node_modules/*' '*/.venv/*' '*/__pycache__/*' '*/dist/*' '*.pyc'
	@ls -lh data/exports/netra-src.zip

warm:
	@echo "downloading detector and ANPR models..."
	$(RUN) detect --camera $(firstword $(shell $(RUN) cameras 2>/dev/null | awk 'NR>2{print $$1}')) \
	  --no-window --max-frames 2 --mode fast >/dev/null 2>&1 || true
	@echo "models cached in data/models"

seed:
	$(RUN) seed

cache-tiles:
	PYTHONPATH=backend $(BIN)/python backend/scripts/cache_tiles.py

stop:
	-@pkill -f "app.cli run" 2>/dev/null || true
	@echo "workers stopped"

db:
	$(RUN) db stats

test:
	PYTHONPATH=backend $(BIN)/python backend/tests/test_tracking.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_store.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_anpr.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_identity.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_trajectory.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_analytics.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_leads.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_preflight.py
	PYTHONPATH=backend $(BIN)/python backend/tests/test_bundle.py

clean:
	rm -rf $(VENV)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# Sumo Companion App Makefile

.PHONY: help preview results daily serve clean install test

# Default basho (update for current tournament)
BASHO ?= 202501
DAY ?= 1

help:
	@echo "Sumo Companion App"
	@echo ""
	@echo "Usage:"
	@echo "  make preview BASHO=202501 DAY=1   Generate preview for a day"
	@echo "  make results BASHO=202501 DAY=1   Generate results for a day"
	@echo "  make daily BASHO=202501 DAY=1     Generate both preview and results"
	@echo "  make serve                        Start local development server"
	@echo "  make install                      Install dependencies"
	@echo "  make clean                        Remove generated files"
	@echo "  make test                         Run tests"
	@echo ""
	@echo "Examples:"
	@echo "  make preview BASHO=202501 DAY=5"
	@echo "  make daily BASHO=202503 DAY=10"

install:
	pip install -r requirements.txt

preview:
	python -m src.companion.daily_pipeline --basho $(BASHO) --day $(DAY) --mode preview

results:
	python -m src.companion.daily_pipeline --basho $(BASHO) --day $(DAY) --mode results

daily:
	python -m src.companion.daily_pipeline --basho $(BASHO) --day $(DAY) --mode both

# Generate all days for a basho (useful for backfilling)
all-days:
	@for day in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		echo "Generating day $$day..."; \
		python -m src.companion.daily_pipeline --basho $(BASHO) --day $$day --mode both || true; \
	done

serve:
	@echo "Starting local server at http://localhost:8000"
	python -m http.server 8000 --directory site

clean:
	rm -rf site/2*
	rm -rf data/*.parquet
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test:
	python -m pytest tests/ -v

# Development helpers
lint:
	python -m flake8 src/companion/

format:
	python -m black src/companion/

# Create data directory
setup-dirs:
	mkdir -p site/css site/js data

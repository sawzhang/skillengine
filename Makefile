.PHONY: test lint build release publish clean

# Load .env if present (exports UV_PUBLISH_TOKEN etc.)
ifneq (,$(wildcard .env))
  include .env
  export
endif

## Development ----------------------------------------------------------------

test:  ## Run all tests
	uv run pytest --tb=short -q

lint:  ## Lint + format check + type check
	uv run ruff check src/
	uv run ruff format --check src/
	uv run mypy src/

## Release --------------------------------------------------------------------

# Usage: make release v=0.3.0
release: _check-version _check-token lint test build publish tag push  ## Full release: lint → test → build → publish → tag → push
	@echo ""
	@echo "  Published skillengine==$(v) to PyPI"
	@echo "  https://pypi.org/project/skillengine/$(v)/"

_check-version:
ifndef v
	$(error Usage: make release v=0.3.0)
endif

_check-token:
ifndef UV_PUBLISH_TOKEN
	$(error UV_PUBLISH_TOKEN not set. Add it to .env or export it)
endif
ifeq ($(UV_PUBLISH_TOKEN),pypi-YOUR_TOKEN_HERE)
	$(error UV_PUBLISH_TOKEN is still the placeholder. Update .env with your real token)
endif

build: clean  ## Build sdist + wheel
	uv build
	@ls -lh dist/

publish:  ## Publish to PyPI (requires UV_PUBLISH_TOKEN)
	uv publish --token $(UV_PUBLISH_TOKEN)

tag:  ## Create git tag (requires v=X.Y.Z)
	git tag v$(v)

push:  ## Push commits + tags to origin
	git push origin master --tags

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info src/*.egg-info

## Helpers --------------------------------------------------------------------

version:  ## Show current version
	@uv run python -c "import skillengine; print(skillengine.__version__)"

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

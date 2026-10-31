#!/usr/bin/env bash
# Run the CI steps in a clean Ubuntu container: proves the install works on Linux without Apple GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD":/src:ro -w /work ubuntu:24.04 bash -lc '
  set -euo pipefail
  apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-rus curl ca-certificates git >/dev/null
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null && export PATH="$HOME/.local/bin:$PATH"
  cp -r /src/. /work && rm -rf /work/.venv /work/data /work/runs
  uv sync -q
  uv run ruff check .
  uv run pytest -q -p no:warnings -m "not real and not ollama"
'

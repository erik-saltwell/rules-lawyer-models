#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root as parent of this script's directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found in PATH" >&2
  exit 1
fi

if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found at repo root: $REPO_ROOT" >&2
  exit 1
fi

echo "Running: uv sync --frozen --inexact"
uv sync --frozen --inexact

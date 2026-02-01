#!/usr/bin/env bash
set -euo pipefail

# clean_rebuild.sh
#
# Default: project-clean + host cache clean + rebuild + up
# Optional: --clean-machine => ALSO removes *all* Docker images/containers/volumes/build cache system-wide
#
# Usage:
#   ./clean_rebuild.sh
#   ./clean_rebuild.sh --clean-machine
#
# Env overrides:
#   COMPOSE_FILE=compose.yaml
#   SERVICE=rules-lawyer-models

MODE="${1:-}"
COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml}"
SERVICE="${SERVICE:-rules-lawyer-models}"

# Find repo root (works even if you run from a subdir)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "ERROR: compose file not found: ${REPO_ROOT}/${COMPOSE_FILE}" >&2
  exit 1
fi

echo "==> Repo: ${REPO_ROOT}"
echo "==> Compose: ${COMPOSE_FILE}"
echo "==> Service: ${SERVICE}"
echo

echo "==> 1) docker compose down (remove orphans)"
docker compose -f "${COMPOSE_FILE}" down --remove-orphans || true

echo
echo "==> 2) Remove project-local caches (repo directories)"
rm -rf \
  "${REPO_ROOT}/.ruff_cache" \
  "${REPO_ROOT}/.mypy_cache" \
  "${REPO_ROOT}/.pytest_cache" \
  "${REPO_ROOT}/.tox" \
  "${REPO_ROOT}/.nox" \
  "${REPO_ROOT}/.cache" \
  "${REPO_ROOT}/.venv" \
  "${REPO_ROOT}/__pycache__" \
  "${REPO_ROOT}/**/__pycache__" 2>/dev/null || true

echo
echo "==> 3) Remove host caches used by this project"
# These are the ones you've been mounting into /home/dev/.cache/...
rm -rf \
  "${HOME}/.cache/uv" \
  "${HOME}/.cache/pip" \
  "${HOME}/.cache/huggingface" \
  "${HOME}/.cache/unsloth_compiled_cache" \
  "${HOME}/.cache/ruff" \
  "${HOME}/.cache/pre-commit" 2>/dev/null || true

mkdir -p \
  "${HOME}/.cache/uv" \
  "${HOME}/.cache/pip" \
  "${HOME}/.cache/huggingface" \
  "${HOME}/.cache/unsloth_compiled_cache" \
  "${HOME}/.cache/ruff" \
  "${HOME}/.cache/pre-commit"

echo
echo "==> 4) Clean Docker build cache (builder cache)"
docker builder prune -af || true

if [[ "${MODE}" == "--clean-machine" ]]; then
  echo
  echo "==> 5) CLEAN MACHINE MODE: remove *all* Docker resources system-wide"
  echo "    (images, stopped containers, networks, and volumes)"
  docker system prune -af --volumes || true
else
  echo
  echo "==> 5) Skip full Docker nuke (run with --clean-machine to wipe everything)"
fi

echo
echo "==> 6) Full rebuild (no cache, pull base images)"
docker compose -f "${COMPOSE_FILE}" build --no-cache --pull "${SERVICE}"

echo
echo "==> 7) docker compose up"
docker compose -f "${COMPOSE_FILE}" up -d

echo
echo "==> 8) Status"
docker compose -f "${COMPOSE_FILE}" ps

#!/usr/bin/env bash
set -euo pipefail

log() { printf "\n[postCreate] %s\n" "$*"; }

log "Starting devcontainer post-create setup..."

# ----------------------------
# Basic dirs
# ----------------------------
log "Ensuring ~/.ssh exists..."
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh" || true

# ----------------------------
# Git config: identity (only set if missing)
# ----------------------------
# NOTE: These defaults match your current devcontainer.json.
# If you ever share this repo, consider setting these via env vars instead.
DEFAULT_GIT_NAME="Erik Saltzman"
DEFAULT_GIT_EMAIL="eriksalt@gmail.com"

current_name="$(git config --global --get user.name || true)"
current_email="$(git config --global --get user.email || true)"

if [[ -z "${current_name}" ]]; then
  log "Setting git user.name -> ${DEFAULT_GIT_NAME}"
  git config --global user.name "${DEFAULT_GIT_NAME}"
else
  log "git user.name already set -> ${current_name}"
fi

if [[ -z "${current_email}" ]]; then
  log "Setting git user.email -> ${DEFAULT_GIT_EMAIL}"
  git config --global user.email "${DEFAULT_GIT_EMAIL}"
else
  log "git user.email already set -> ${current_email}"
fi

# ----------------------------
# Git signing (SSH)
# ----------------------------
log "Configuring git SSH commit signing..."

git config --global gpg.format ssh
git config --global commit.gpgsign true

# The key file you mount into the container.
SIGNING_KEY_PATH="/.devcontainer/git_signing_key.pub"
ALLOWED_SIGNERS_PATH="/.devcontainer/allowed_signers"

if [[ ! -f "${SIGNING_KEY_PATH}" ]]; then
  log "WARNING: ${SIGNING_KEY_PATH} not found. SSH signing key not configured."
else
  git config --global user.signingkey "${SIGNING_KEY_PATH}"
  log "user.signingkey -> ${SIGNING_KEY_PATH}"
fi

if [[ ! -f "${ALLOWED_SIGNERS_PATH}" ]]; then
  log "WARNING: ${ALLOWED_SIGNERS_PATH} not found. gpg.ssh.allowedSignersFile not configured."
else
  git config --global gpg.ssh.allowedSignersFile "${ALLOWED_SIGNERS_PATH}"
  log "gpg.ssh.allowedSignersFile -> ${ALLOWED_SIGNERS_PATH}"
fi

# ----------------------------
# uv sync (project deps)
# ----------------------------
# Your container already installs torch at the platform layer.
# --inexact ensures uv doesn't try to remove platform packages that aren't in the lock.
log "Running: uv sync --frozen --inexact"
uv sync --frozen --inexact

log "Post-create complete."

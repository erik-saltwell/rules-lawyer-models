# syntax=docker/dockerfile:1.6
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

# Build-time pin for Unsloth installer (can be a commit hash)
ARG UNSLOTH_INSTALLER_REF=main

# Minimal OS tools + micro editor (plus tini for correct signal handling)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git openssh-client wget build-essential python3-dev pkg-config curl tini micro \
    && rm -rf /var/lib/apt/lists/*

# Install uv by copying the binaries from Astral’s official image
# Best practice: pin this tag later (instead of :latest) once you're happy.
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

# Project conventions + uv behavior in containers (build-time defaults)
ENV WORKDIR=/workspace \
    UV_LINK_MODE=copy \
    UV_PYTHON_CACHE_DIR=/root/.cache/uv/python \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv


WORKDIR ${WORKDIR}

RUN mkdir -p /opt/uv/python && chmod 0755 /opt/uv /opt/uv/python

# Create venv with the exact Python version (uv will download Python if needed)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python 3.11.14

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install PyTorch (CUDA 12.8 build) as a platform-level dependency
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install project deps from lock file only (cache-friendly)
COPY pyproject.toml uv.lock ./

# Keep torch even though it's installed separately (not necessarily in uv.lock)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --frozen --no-install-project --inexact

# Install Unsloth (platform layer) using our robust installer wrapper.
COPY docker/install_unsloth.py /usr/local/bin/install_unsloth.py

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    python -c "import torch; print('torch:', torch.__version__, 'cuda build:', torch.version.cuda)"; \
    UNSLOTH_INSTALLER_REF="${UNSLOTH_INSTALLER_REF:-main}" python /usr/local/bin/install_unsloth.py

# Copy shell helpers into a stable location in the image
COPY .devcontainer/shell/uv_guard.sh /etc/profile.d/uv_guard.sh

RUN chmod 644 /etc/profile.d/uv_guard.sh \
    && printf '\n# uv guard\n[ -f /etc/profile.d/uv_guard.sh ] && . /etc/profile.d/uv_guard.sh\n' >> /etc/bash.bashrc \
    && if [ -f /etc/zsh/zshrc ]; then \
    printf '\n# uv guard\nsource /etc/profile.d/uv_guard.sh\n' >> /etc/zsh/zshrc ; \
    fi

# Create /usr/local/bin/uv-shell (safe sync wrapper)
RUN cat >/usr/local/bin/uv-shell <<'EOF' \
    && chmod 0755 /usr/local/bin/uv-shell
#!/usr/bin/env bash
set -euo pipefail

# Find the git repo root (works from anywhere inside the repo)
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${root}" ]]; then
  echo "ERROR: uv-shell must be run inside a git checkout (cannot find repo root)." >&2
  exit 1
fi

cd "${root}"

if [[ ! -f pyproject.toml ]]; then
  echo "ERROR: pyproject.toml not found at repo root: ${root}" >&2
  exit 1
fi

echo "Running: uv sync --frozen --inexact"
uv sync --frozen --inexact
EOF

# --- Non-root dev user that matches the host UID/GID ---
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN set -eux; \
    # Find an existing group *by GID* (not by name)
    existing_group="$(getent group | awk -F: -v gid="${USER_GID}" '$3==gid {print $1; exit}')"; \
    if [ -z "${existing_group}" ]; then \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    existing_group="${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
    useradd --uid "${USER_UID}" --gid "${existing_group}" -m "${USERNAME}"; \
    fi; \
    install -d -o "${USER_UID}" -g "${USER_GID}" -m 0700 "/home/${USERNAME}/.ssh"; \
    install -d -o "${USER_UID}" -g "${USER_GID}" "/home/${USERNAME}/.cache"; \
    chown -R "${USER_UID}:${USER_GID}" /opt/venv

# Runtime defaults for caches (so non-root can write)
ENV HOME=/home/${USERNAME} \
    UV_CACHE_DIR=/home/${USERNAME}/.cache/uv \
    UV_PYTHON_CACHE_DIR=/home/${USERNAME}/.cache/uv/python \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PIP_CACHE_DIR=/home/${USERNAME}/.cache/pip


# Default user for the container is non-root
USER ${USERNAME}

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash"]

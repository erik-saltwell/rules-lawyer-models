# syntax=docker/dockerfile:1.6
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

# Minimal OS tools + micro editor (plus tini for correct signal handling)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git tini micro \
    && rm -rf /var/lib/apt/lists/*

# Install uv by copying the binaries from Astral’s official image
# Best practice: pin this tag later (instead of :latest) once you're happy.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Project conventions + uv behavior in containers
ENV WORKDIR=/workspace \
    UV_LINK_MODE=copy \
    UV_PYTHON_CACHE_DIR=/root/.cache/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR ${WORKDIR}

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

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash"]

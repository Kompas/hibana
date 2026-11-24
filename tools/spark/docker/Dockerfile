# Hibana (the Kompas fork, branch spark) at a pinned commit. Adapted from dedene/Hibana-Qwen3.8-Flash-Next-DGX-Spark
# (MIT, Copyright (c) 2026 Peter Dedene): the patches step is gone, the spark branch carries them.
# Hibana at a pinned commit, with its locked GB10 environment (uv.lock: torch 2.13 cu130, FlashInfer 0.6.17, Triton).
# Hibana runs on the host with uv by default; this puts the same `uv sync` in an image. The CUDA 13 devel base gives
# the /usr/local/cuda/bin/nvcc that Hibana's preflight wants and its first-use kernel builds need.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d

COPY --from=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /usr/local/bin/uv

ARG HIBANA_REPO=https://github.com/Kompas/hibana.git
ARG HIBANA_COMMIT

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# A uv-managed Python 3.12 (pyproject: python-preference only-managed), kept in the image, not in a cache volume.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1

WORKDIR /opt/hibana
RUN test -n "$HIBANA_COMMIT" && git init -q . \
    && git remote add origin "$HIBANA_REPO" \
    && git fetch -q --depth 1 origin "$HIBANA_COMMIT" \
    && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$HIBANA_COMMIT"
RUN uv sync --frozen

ENV PATH=/opt/hibana/.venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TRITON_CACHE_DIR=/root/.cache/triton
# No GPU at build time: the launcher is torch-free, so check that it and the locked torch build are there.
RUN hibana --help >/dev/null \
    && python tools/spark/tests/test_tool_args_anyof.py \
    && python -c "import importlib.metadata as m; v = m.version('torch'); assert v.startswith('2.13.0') and 'cu130' in v, v; print('torch', v)"

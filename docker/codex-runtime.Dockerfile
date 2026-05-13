# syntax=docker/dockerfile:1
FROM node:22-slim

# Install system dependencies with cache mount and minimal packages
# Removed: build-essential (only needed for compiling, not runtime)
# Added: --no-install-recommends to skip extra packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-dev \
    python3-minimal \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
    ln -sf /opt/venv/bin/pip3 /usr/local/bin/pip3
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV VIRTUAL_ENV=/opt/venv

# Install Astral uv and npm-backed agent CLIs in one layer
# Use npm cache mount for faster installs
RUN --mount=type=cache,target=/root/.npm \
    mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx && \
    npm install -g @openai/codex @anthropic-ai/claude-code @mariozechner/pi-coding-agent playwright && \
    npx playwright install --with-deps chromium && \
    node --version && npm --version && npx --version && uv --version && uvx --version && python --version && python3 --version && pip --version && pip3 --version && codex --version && claude --version && pi --version

RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 https://github.com/NousResearch/hermes-agent.git /usr/local/lib/hermes-agent && \
    cd /usr/local/lib/hermes-agent && \
    uv venv venv --python python3 && \
    uv pip install --python /usr/local/lib/hermes-agent/venv/bin/python -e . && \
    printf '#!/usr/bin/env bash\nunset PYTHONPATH\nunset PYTHONHOME\nexec /usr/local/lib/hermes-agent/venv/bin/hermes "$@"\n' > /usr/local/bin/hermes && \
    chmod +x /usr/local/bin/hermes && \
    (hermes --version || hermes --help >/dev/null)

# Create working directory and user in one layer
WORKDIR /workspace
RUN useradd -m -s /bin/bash codex && \
    mkdir -p /home/codex/.codex/sessions && \
    mkdir -p /home/codex/.claude && \
    mkdir -p /home/codex/.hermes && \
    chown -R codex:codex /home/codex /workspace ${PLAYWRIGHT_BROWSERS_PATH} ${VIRTUAL_ENV}

USER codex
ENV CODEX_HOME=/home/codex
ENV HOME=/home/codex
ENV NODE_PATH="/usr/local/lib/node_modules"
ENV PATH="/opt/venv/bin:/usr/local/bin:$PATH"

CMD ["bash"]

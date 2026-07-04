# syntax=docker/dockerfile:1
FROM node:26.4.0-alpine3.24

# Install system dependencies with cache mount and minimal packages
# Removed: build-essential (only needed for compiling, not runtime)
# Added: --no-install-recommends to skip extra packages
RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \
    apk update && apk add \
    build-base \
    bash \
    git \
    curl \
    ca-certificates \
    sudo \
    python3 \
    python3-dev \
    py3-pip \
    chromium

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
    ln -sf /opt/venv/bin/pip3 /usr/local/bin/pip3
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV VIRTUAL_ENV=/opt/venv
ARG TEICH_INSTALL_LANGFUSE=0

# Install Astral uv and npm-backed agent CLIs in one layer
# Use npm cache mount for faster installs
RUN --mount=type=cache,target=/root/.npm \
    mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx && \
    npm install -g @openai/codex @anthropic-ai/claude-code @mariozechner/pi-coding-agent playwright && \

    node --version && npm --version && npx --version && uv --version && uvx --version && python --version && python3 --version && pip --version && pip3 --version && codex --version && claude --version && pi --version

RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 https://github.com/NousResearch/hermes-agent.git /usr/local/lib/hermes-agent && \
    cd /usr/local/lib/hermes-agent && \
    uv venv venv --python python3 && \
    uv pip install --python /usr/local/lib/hermes-agent/venv/bin/python -e . && \
    printf '#!/usr/bin/env bash\nunset PYTHONPATH\nunset PYTHONHOME\nexec /usr/local/lib/hermes-agent/venv/bin/hermes "$@"\n' > /usr/local/bin/hermes && \
    chmod +x /usr/local/bin/hermes && \
    (hermes --version || hermes --help >/dev/null)

# Bake Langfuse tooling only for the tracing-enabled runtime image. The default
# runtime image leaves this off so normal Teich runs do not depend on Langfuse
# plugin, pip, or git availability during Docker rebuilds.
RUN HOME=/opt/codex-langfuse CODEX_HOME=/opt/codex-langfuse/.codex \
    sh -c 'mkdir -p "$CODEX_HOME" \
      && if [ "$TEICH_INSTALL_LANGFUSE" = "1" ]; then \
        codex plugin marketplace add langfuse/codex-observability-plugin \
        && codex plugin add tracing@codex-observability-plugin; \
      fi \
      && chmod -R a+rX /opt/codex-langfuse'

# Langfuse hook + SDK for Claude Code, pinned to a known-good commit so the
# baked hook script is reproducible. The SDK goes in the venv because Claude
# strips it from a hook's PATH, so the hook calls it by full path.
RUN if [ "$TEICH_INSTALL_LANGFUSE" = "1" ]; then \
      /opt/venv/bin/pip install --no-cache-dir "langfuse>=4.0,<5" \
      && git clone https://github.com/langfuse/Claude-Observability-Plugin.git \
          /opt/claude-langfuse-plugin \
      && git -C /opt/claude-langfuse-plugin checkout 597af67d6c6b369f3e55db6cfa2ebe444f1af46c \
      && rm -rf /opt/claude-langfuse-plugin/.git \
      && chmod -R a+rX /opt/claude-langfuse-plugin; \
    else \
      mkdir -p /opt/claude-langfuse-plugin; \
    fi

# Create working directory and user in one layer
WORKDIR /workspace
RUN adduser -D -s /bin/bash codex && \
    mkdir -p /home/codex/.codex/sessions && \
    mkdir -p /home/codex/.claude && \
    mkdir -p /home/codex/.hermes && \
    printf 'codex ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/codex && \
    chmod 0440 /etc/sudoers.d/codex && \
    printf '#!/usr/bin/env bash\nif [ "$1" = "install" ]; then shift; pkgs=(); for i in "$@"; do if [[ "$i" != -* ]]; then pkgs+=("$i"); fi; done; exec sudo /sbin/apk add "${pkgs[@]}"; elif [ "$1" = "update" ]; then exec sudo /sbin/apk update; else exec sudo /sbin/apk "$@"; fi\n' > /usr/local/bin/apt-get && \
    printf '#!/usr/bin/env bash\nif [ "$1" = "install" ]; then shift; pkgs=(); for i in "$@"; do if [[ "$i" != -* ]]; then pkgs+=("$i"); fi; done; exec sudo /sbin/apk add "${pkgs[@]}"; elif [ "$1" = "update" ]; then exec sudo /sbin/apk update; else exec sudo /sbin/apk "$@"; fi\n' > /usr/local/bin/apt && \
    chmod +x /usr/local/bin/apt-get /usr/local/bin/apt && \
    chown -R codex:codex /home/codex /workspace ${PLAYWRIGHT_BROWSERS_PATH} ${VIRTUAL_ENV}

USER codex
ENV CODEX_HOME=/home/codex
ENV HOME=/home/codex
ENV NODE_PATH="/usr/local/lib/node_modules"
ENV PATH="/opt/venv/bin:/usr/local/bin:$PATH"

CMD ["bash"]

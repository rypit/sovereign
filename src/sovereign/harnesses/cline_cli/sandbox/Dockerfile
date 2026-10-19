# Sandbox image for the cline_cli harness (ADR 0009): everything the agent
# needs and nothing else. Built on first sandboxed use as
# sovereign/cline-sandbox:latest; override via the harness's sandbox.image.
FROM node:22-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g cline

# The task workdir is bind-mounted here; run as the unprivileged node user.
USER node
WORKDIR /workspace

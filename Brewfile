# Sovereign bootstrap dependencies. Install/verify with:
#   brew bundle            # install
#   brew bundle check      # verify satisfied
# or run: python3 scripts/setup.py
#
# Only the toolchain needed to bootstrap Sovereign itself lives here.
# Per-integration dependencies (llama.cpp, Docker Desktop, Node/Cline, ...)
# live in each integration's own Brewfile (src/sovereign/services/*/Brewfile,
# src/sovereign/harnesses/*/Brewfile) and are installed by
# `sovereign provision` — run automatically by setup.py, at boot for anything
# declared in sovereign.yaml, or by hand (`sovereign provision -f stack.yaml`).

brew "uv"               # Python environment + dependency manager

# Alternatives / future (uncomment as needed):
# cask "orbstack"       # lighter Docker Desktop replacement (swap for docker-desktop)

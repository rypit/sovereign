"""Coding-harness integrations.

Each harness lives in its own folder with a ``config.py`` + ``manager.py`` and
registers itself via :mod:`sovereign.core.registry`. This module will import every
harness package so registration happens on ``import sovereign.harnesses`` — none
exist yet (they arrive in the harness track, roughly alongside Phase 11).
"""

"""Benchmarking subsystem (§6b) — a second consumer of the Orchestrator-as-library.

Benchmarks are deliberately outside ``sovereign.yaml``: a bench spec (``spec.py``)
is loaded imperatively by ``sovereign bench run``, never at boot. ``Job``
(``runner.py``) is a distinct run-to-completion type — services never finish, so
this is not a stretched service state machine (§0 core taxonomy).
"""

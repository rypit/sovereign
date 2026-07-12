#!/usr/bin/env python3
"""Documentation consistency checks — stdlib-only, exits 1 on any failure.

Three independent checks, each printing every failure it finds before
exiting non-zero:

1. **`§N` citation validity** — every ``§N``/``§Nx``/``§N.M`` citation found
   in ``src/**/*.py`` must resolve to a heading in
   ``docs/sovereign-implementation-plan-v1.1.md`` (exact match, or — for a
   dotted citation like ``§11.5`` — its base section ``§11`` existing as a
   heading), or be explicitly listed in ``KNOWN_EXTERNAL`` below. Citations
   like ``§3a`` in the worker/telemetry code refer to a *different* plan
   document (the PR #20 design doc, not v1.1) — those are the
   ``KNOWN_EXTERNAL`` cases, each pointing at the ADR that now carries that
   decision instead of weakening this check to a no-op.
2. **ADR well-formedness** — every ``docs/decisions/NNNN-*.md`` has a
   parseable ``NNNN-kebab-title.md`` filename and a well-formed ``Status:``
   line (``accepted`` or ``superseded-by-NNNN`` naming an ADR that exists).
3. **Architecture-doc / depgraph parity** — the rule ids listed in
   ``docs/architecture.md``'s "Dependency rules" section match
   ``scripts/depgraph.py``'s ``ARCH_RULES`` ids exactly (imported via
   ``importlib`` rather than re-parsed, since both scripts are stdlib-only).

Usage:
    uv run python scripts/check_docs.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "sovereign"
PLAN_V1_1 = REPO_ROOT / "docs" / "sovereign-implementation-plan-v1.1.md"
ARCHITECTURE_MD = REPO_ROOT / "docs" / "architecture.md"
DECISIONS_DIR = REPO_ROOT / "docs" / "decisions"

# §-citations that refer to a document other than plan v1.1 (or to a
# plan section that was since renumbered/retired) but are still valid
# because an ADR now carries that decision. Add an entry here — with a
# comment explaining the provenance — rather than loosening the checker.
KNOWN_EXTERNAL: dict[str, str] = {
    # PR #20's own design doc used a local "§3. Engine embedding > Hard
    # gaps" numbering distinct from plan v1.1's §3 (Repository Structure).
    # ADR 0006 now records that decision.
    "3a": "docs/decisions/0006-engine-gap-policy.md",
}

CITATION_RE = re.compile(r"§(\d+[a-zA-Z]*(?:\.\d+)?)")
HEADING_RE = re.compile(r"^#{2,3}\s+(\d+[a-zA-Z]*(?:\.\d+)?)\b")
ADR_FILENAME_RE = re.compile(r"^(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
STATUS_RE = re.compile(r"^Status:\s*(accepted|superseded-by-(\d{4}))\s*$")


def _load_depgraph():
    """Import scripts/depgraph.py as a module without depending on package layout."""
    spec = importlib.util.spec_from_file_location(
        "sovereign_depgraph", REPO_ROOT / "scripts" / "depgraph.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plan_headings(text: str) -> set[str]:
    return {m.group(1) for line in text.splitlines() if (m := HEADING_RE.match(line))}


def citation_is_valid(citation: str, headings: set[str]) -> bool:
    if citation in headings:
        return True
    if citation in KNOWN_EXTERNAL:
        return True
    if "." in citation:
        base = citation.split(".", 1)[0]
        if base in headings:
            return True
    return False


def check_citations() -> list[str]:
    errors: list[str] = []
    headings = plan_headings(PLAN_V1_1.read_text(encoding="utf-8"))
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in CITATION_RE.finditer(line):
                citation = m.group(1)
                if not citation_is_valid(citation, headings):
                    rel = path.relative_to(REPO_ROOT)
                    errors.append(
                        f"{rel}:{lineno}: §{citation} does not resolve to a heading in "
                        f"{PLAN_V1_1.relative_to(REPO_ROOT)} and is not in KNOWN_EXTERNAL"
                    )
    return errors


def check_adrs() -> list[str]:
    errors: list[str] = []
    if not DECISIONS_DIR.is_dir():
        return [f"{DECISIONS_DIR.relative_to(REPO_ROOT)} does not exist"]

    adr_paths = sorted(
        p for p in DECISIONS_DIR.glob("*.md") if p.name != "README.md"
    )
    numbers = set()
    for path in adr_paths:
        rel = path.relative_to(REPO_ROOT)
        m = ADR_FILENAME_RE.match(path.name)
        if not m:
            errors.append(f"{rel}: filename does not match NNNN-kebab-title.md")
            continue
        numbers.add(m.group(1))

    for path in adr_paths:
        rel = path.relative_to(REPO_ROOT)
        m = ADR_FILENAME_RE.match(path.name)
        if not m:
            continue
        text = path.read_text(encoding="utf-8")
        status_line = next(
            (line for line in text.splitlines() if line.startswith("Status:")), None
        )
        if status_line is None:
            errors.append(f"{rel}: missing a `Status:` line")
            continue
        sm = STATUS_RE.match(status_line.strip())
        if not sm:
            errors.append(
                f"{rel}: malformed Status line {status_line!r} "
                "(expected 'accepted' or 'superseded-by-NNNN')"
            )
            continue
        target = sm.group(2)
        if target is not None and target not in numbers:
            errors.append(
                f"{rel}: superseded-by-{target} names an ADR number that doesn't exist"
            )
    return errors


def check_architecture_rule_parity() -> list[str]:
    if not ARCHITECTURE_MD.exists():
        return [f"{ARCHITECTURE_MD.relative_to(REPO_ROOT)} does not exist"]

    depgraph = _load_depgraph()
    rule_ids = {rule.id for rule in depgraph.ARCH_RULES}

    text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    # Rule ids are documented as `` `rule-id` `` at the start of a bullet in
    # the "Dependency rules" section, e.g. "- **`config-golden-rule`** — ...".
    doc_ids = set(re.findall(r"- \*\*`([a-z0-9-]+)`\*\*", text))

    missing_from_doc = rule_ids - doc_ids
    stale_in_doc = doc_ids - rule_ids
    errors: list[str] = []
    if missing_from_doc:
        errors.append(
            f"{ARCHITECTURE_MD.relative_to(REPO_ROOT)} is missing rule id(s) "
            f"present in depgraph.ARCH_RULES: {sorted(missing_from_doc)}"
        )
    if stale_in_doc:
        errors.append(
            f"{ARCHITECTURE_MD.relative_to(REPO_ROOT)} documents rule id(s) no "
            f"longer in depgraph.ARCH_RULES: {sorted(stale_in_doc)}"
        )
    return errors


def main() -> int:
    all_errors: list[str] = [
        *check_citations(),
        *check_adrs(),
        *check_architecture_rule_parity(),
    ]

    if all_errors:
        print(f"check_docs: {len(all_errors)} problem(s):")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print("check_docs: clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

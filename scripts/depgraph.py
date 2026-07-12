#!/usr/bin/env python3
"""Generate an AST-based internal dependency graph of the ``sovereign`` package.

Parses every module under ``src/sovereign`` with the stdlib ``ast`` module,
extracts each file's *internal* imports (targets under the ``sovereign``
package — third-party and stdlib imports are ignored), and emits a single
Markdown report with an embedded Mermaid chart plus coupling signals: import
cycles (strongly connected components) and a fan-in/fan-out table. The goal is
to make tightly coupled areas obvious at a glance.

Imports that appear only inside ``if TYPE_CHECKING:`` blocks are classified as
type-only and rendered as dashed edges. They are excluded from cycle detection
and the fan-in/fan-out table.

Stdlib-only (no networkx / pydeps) so it runs on a bare checkout. Output
ordering is fully sorted, so re-running produces no spurious diffs.

Usage:
    uv run python scripts/depgraph.py
    uv run python scripts/depgraph.py --root src/sovereign --out docs/dependency-graph.md
    uv run python scripts/depgraph.py --check

``--check`` runs the same analysis in-memory (nothing is written) and exits
1 on the first violation found: a runtime import cycle, an ``ARCH_RULES``
layering violation, or a stale ``docs/dependency-graph.md``. See
``docs/architecture.md`` — its "Dependency rules" section must list the same
rule ids as ``ARCH_RULES`` below (``scripts/check_docs.py`` asserts this).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "src" / "sovereign"
DEFAULT_OUT = REPO_ROOT / "docs" / "dependency-graph.md"

# The dotted prefix that marks an import as internal (worth graphing).
PACKAGE = "sovereign"


@dataclasses.dataclass(frozen=True)
class ArchRule:
    """A layering rule evaluated over the runtime edge list.

    ``scope`` is an fnmatch-style pattern (dotted module names, ``*`` wildcard)
    selecting which source modules the rule applies to. ``allowed`` is the
    exhaustive set of import patterns those modules may use — anything else is
    a violation. Exactly one of ``allowed`` should be set per rule; keep rules
    simple (allow-list only) so they stay easy to read as data.
    """

    id: str
    description: str
    scope: tuple[str, ...]
    allowed: tuple[str, ...]
    exempt: tuple[str, ...] = ()

    def applies_to(self, module: str) -> bool:
        if any(fnmatch.fnmatchcase(module, pat) for pat in self.exempt):
            return False
        return any(fnmatch.fnmatchcase(module, pat) for pat in self.scope)

    def permits(self, target: str) -> bool:
        """Is an import of ``target`` allowed under this rule?

        Two rule shapes, distinguished by ``allowed``'s contents: an
        allow-list (only these patterns are permitted — everything else is a
        violation) or, when every entry is negated (``"!pattern"``), a
        deny-list (everything is permitted except these patterns).
        """
        if self.allowed and all(pat.startswith("!") for pat in self.allowed):
            forbidden = [pat[1:] for pat in self.allowed]
            return not any(fnmatch.fnmatchcase(target, pat) for pat in forbidden)
        return any(fnmatch.fnmatchcase(target, pat) for pat in self.allowed)


# Layering rules, declared as data. Keep in sync with the "Dependency rules"
# section of docs/architecture.md (rule ids must match exactly).
ARCH_RULES: tuple[ArchRule, ...] = (
    ArchRule(
        id="config-golden-rule",
        description=(
            "config.py, **/config.py, and core/base_config.py may only import "
            "core.units and core.base_config (plus stdlib/Pydantic) — config "
            "must never own subprocess/os/docker."
        ),
        scope=("sovereign.config", "*.config", "sovereign.core.base_config"),
        allowed=("sovereign.core.units", "sovereign.core.base_config"),
    ),
    ArchRule(
        id="workers-leaf",
        description=(
            "workers/* may only import other workers/* modules and "
            "core.procmem — worker modules stay importable without engine "
            "bindings or the rest of the control plane."
        ),
        scope=("sovereign.workers.*",),
        allowed=("sovereign.workers.*", "sovereign.core.procmem"),
    ),
    ArchRule(
        id="hf-leaf",
        description=(
            "services/inference/hf.py may only import core.errors and "
            "core.state — it stays a leaf nothing above services/ needs to "
            "import directly."
        ),
        scope=("sovereign.services.inference.hf",),
        allowed=("sovereign.core.errors", "sovereign.core.state"),
    ),
    ArchRule(
        id="runtime-no-bench",
        description="runtime/* must never import bench/*.",
        scope=("sovereign.runtime.*",),
        allowed=("!sovereign.bench.*",),
    ),
    ArchRule(
        id="bench-single-door",
        description=(
            "Only bench/cleanroom.py may import runtime.orchestrator — every "
            "other bench module reaches the orchestrator (if at all) through "
            "that one door."
        ),
        scope=("sovereign.bench.*",),
        allowed=("!sovereign.runtime.orchestrator",),
        exempt=("sovereign.bench.cleanroom",),
    ),
    ArchRule(
        id="core-single-door",
        description=(
            "Nothing in core/ imports services/ or harnesses/ at runtime "
            "except core.registry — the one sanctioned door from the "
            "contract layer into concrete integrations."
        ),
        scope=("sovereign.core.*",),
        allowed=("!sovereign.services.*", "!sovereign.harnesses.*"),
        exempt=("sovereign.core.registry",),
    ),
)

# Rule ids we've deliberately decided to keep despite a violation, e.g.
# ("workers-leaf", "sovereign.workers.foo", "sovereign.core.bar"). Empty: the
# current graph is clean against every rule above.
GRANDFATHERED: frozenset[tuple[str, str, str]] = frozenset()


def discover_modules(root: Path) -> dict[str, Path]:
    """Map every module's dotted name to its file path.

    ``src/sovereign/services/inference/base.py`` -> ``sovereign.services.inference.base``
    and ``.../mlx_lm/__init__.py`` -> ``sovereign.services.inference.mlx_lm``.
    """
    modules: dict[str, Path] = {}
    src_root = root.parent  # so the package dir itself becomes the first part
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(src_root)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


def _resolve(target: str, known: set[str]) -> str | None:
    """Resolve an imported dotted name to the deepest known module it names.

    ``sovereign.core.registry.route_entry`` names the ``...registry`` module
    (``route_entry`` is an attribute, not a submodule), so walk parents until
    one is a known module. Returns ``None`` for external targets.
    """
    if target == PACKAGE or target.startswith(PACKAGE + "."):
        parts = target.split(".")
        while parts:
            candidate = ".".join(parts)
            if candidate in known:
                return candidate
            parts.pop()
    return None


def _collect_imports_from_nodes(
    nodes: list[ast.stmt],
    known: set[str],
) -> set[str]:
    """Collect resolved internal imports from a flat list of AST statements."""
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (resolved := _resolve(alias.name, known)) is not None:
                    found.add(resolved)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — none exist today, but be safe
                continue
            module = node.module or ""
            resolved_any = False
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                if (resolved := _resolve(full, known)) is not None:
                    found.add(resolved)
                    resolved_any = True
            if not resolved_any and (resolved := _resolve(module, known)) is not None:
                found.add(resolved)
    return found


def _is_type_checking_guard(node: ast.If) -> bool:
    """Return True if an ``if`` node is an ``if TYPE_CHECKING:`` guard.

    Matches both bare ``TYPE_CHECKING`` and ``typing.TYPE_CHECKING``.
    """
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
    ):
        return True
    return False


def extract_edges(
    name: str, path: Path, known: set[str]
) -> tuple[set[str], set[str]]:
    """Return ``(runtime_edges, type_only_edges)`` for module ``name``.

    Imports that appear at module top level but inside an ``if TYPE_CHECKING:``
    block are classified as type-only. If the same target appears both at
    runtime and under TYPE_CHECKING, it is classified as runtime.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body  # top-level statements only

    type_only_stmts: list[ast.stmt] = []
    runtime_stmts: list[ast.stmt] = []

    for node in body:
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            # Only the if-body; the else-branch (if any) runs at runtime.
            type_only_stmts.extend(node.body)
        else:
            # For non-TYPE_CHECKING nodes we need to walk recursively to catch
            # imports nested in try/except, with-blocks, etc.
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    runtime_stmts.append(child)

    type_only = _collect_imports_from_nodes(type_only_stmts, known)
    runtime = _collect_imports_from_nodes(runtime_stmts, known)

    # If a module appears in both, runtime wins.
    type_only -= runtime

    runtime.discard(name)
    type_only.discard(name)

    return runtime, type_only


# EdgeGraph holds two parallel adjacency sets: runtime and type-only.
EdgeGraph = dict[str, tuple[set[str], set[str]]]


def build_graph(modules: dict[str, Path]) -> EdgeGraph:
    known = set(modules)
    return {
        name: extract_edges(name, path, known)
        for name, path in modules.items()
    }


def runtime_graph(graph: EdgeGraph) -> dict[str, set[str]]:
    """Extract the runtime-only adjacency dict."""
    return {name: runtime for name, (runtime, _) in graph.items()}


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return non-trivial strongly connected components (Tarjan's algorithm)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                sccs.append(sorted(component))

    for node in sorted(graph):
        if node not in index:
            strongconnect(node)
    return sorted(sccs, key=lambda c: (-len(c), c))


def _group(name: str) -> str:
    """Top-level grouping key for a module (the package immediately under root)."""
    parts = name.split(".")
    return parts[1] if len(parts) > 2 else "top-level"


def build_labels(modules: dict[str, Path], root: Path) -> dict[str, str]:
    """Human-readable node label per module, e.g. ``services/inference/base.py``.

    Package ``__init__.py`` files keep their real name (``bench/__init__.py``)
    rather than collapsing to a misleading ``bench.py``.
    """
    src_root = root.parent
    return {
        name: str(path.relative_to(src_root).relative_to(PACKAGE))
        for name, path in modules.items()
    }


def _node_id(name: str) -> str:
    """Mermaid-safe node id (letters, digits, underscores only)."""
    return "n_" + name.replace(".", "_")


def render_mermaid(
    graph: EdgeGraph, labels: dict[str, str], cycle_nodes: set[str]
) -> str:
    lines = ["```mermaid", "graph LR"]

    groups: dict[str, list[str]] = {}
    for name in graph:
        groups.setdefault(_group(name), []).append(name)

    for group in sorted(groups):
        lines.append(f"  subgraph {group}")
        for name in sorted(groups[group]):
            lines.append(f'    {_node_id(name)}["{labels[name]}"]')
        lines.append("  end")

    # Collect all edges sorted for deterministic output.
    runtime_edges = sorted(
        (src, dst) for src, (rt, _) in graph.items() for dst in rt
    )
    type_only_edges = sorted(
        (src, dst) for src, (_, to) in graph.items() for dst in to
    )

    # Emit runtime edges first (solid), then type-only (dashed).
    # Track link indices for cycle highlighting (runtime only).
    cycle_link_indices: list[int] = []
    link_index = 0

    for src, dst in runtime_edges:
        lines.append(f"  {_node_id(src)} --> {_node_id(dst)}")
        if src in cycle_nodes and dst in cycle_nodes:
            cycle_link_indices.append(link_index)
        link_index += 1

    for src, dst in type_only_edges:
        lines.append(f"  {_node_id(src)} -.-> {_node_id(dst)}")
        link_index += 1

    # Highlight cycle members and the edges between them.
    if cycle_nodes:
        lines.append("  classDef cycle stroke:#e53935,stroke-width:2px,color:#e53935;")
        members = ",".join(_node_id(n) for n in sorted(cycle_nodes))
        lines.append(f"  class {members} cycle;")
    if cycle_link_indices:
        idx = ",".join(str(i) for i in cycle_link_indices)
        lines.append(f"  linkStyle {idx} stroke:#e53935,stroke-width:2px;")

    lines.append("```")
    return "\n".join(lines)


def render_report(graph: EdgeGraph, labels: dict[str, str]) -> str:
    rt_graph = runtime_graph(graph)
    cycles = find_cycles(rt_graph)
    cycle_nodes = {n for scc in cycles for n in scc}

    runtime_edge_count = sum(len(rt) for rt, _ in graph.values())
    type_only_edge_count = sum(len(to) for _, to in graph.values())
    total_edges = runtime_edge_count + type_only_edge_count

    fan_out = {name: len(rt) for name, (rt, _) in graph.items()}
    fan_in: dict[str, int] = {name: 0 for name in graph}
    for rt, _ in graph.values():
        for dst in rt:
            fan_in[dst] += 1

    generated = dt.datetime.now().strftime("%Y-%m-%d")

    out: list[str] = []
    out.append("# Internal dependency graph")
    out.append("")
    out.append(
        f"_Generated {generated} by `scripts/depgraph.py` — "
        f"{len(graph)} modules, {total_edges} internal import edges "
        f"({runtime_edge_count} runtime, {type_only_edge_count} type-annotation-only). "
        "Regenerate with `make graph`._"
    )
    out.append("")
    out.append(
        "Nodes are modules under `sovereign`, grouped by top-level package. "
        "Only imports internal to the package are shown. "
        "Solid arrows (`-->`) are runtime imports; "
        "dashed arrows (`-.->`) are type-annotation-only imports "
        "(`if TYPE_CHECKING:` blocks). "
        "Type-only edges are excluded from cycle detection and fan-in/fan-out. "
        "Modules that participate in a runtime import cycle are outlined in red."
    )
    out.append("")
    out.append(render_mermaid(graph, labels, cycle_nodes))
    out.append("")

    out.append("## Import cycles")
    out.append("")
    if cycles:
        out.append(
            "Strongly connected components (each is a group of modules that "
            "transitively import each other at runtime — the tightest coupling there is):"
        )
        out.append("")
        for scc in cycles:
            out.append(f"- {' → '.join(labels[n] for n in scc)} → …")
    else:
        out.append("None detected ✅")
    out.append("")

    out.append("## Coupling (fan-in / fan-out)")
    out.append("")
    out.append(
        "Sorted by total coupling (fan-in + fan-out). "
        "Counts runtime edges only. "
        "High fan-in = a hub many modules depend on; "
        "high fan-out = a module that pulls in a lot."
    )
    out.append("")
    out.append("| Module | Fan-in | Fan-out | Total |")
    out.append("| --- | ---: | ---: | ---: |")
    ranked = sorted(
        graph,
        key=lambda n: (-(fan_in[n] + fan_out[n]), n),
    )
    for name in ranked:
        fi, fo = fan_in[name], fan_out[name]
        out.append(f"| `{labels[name]}` | {fi} | {fo} | {fi + fo} |")
    out.append("")

    return "\n".join(out)


def check_arch_rules(graph: EdgeGraph) -> list[str]:
    """Evaluate ``ARCH_RULES`` over the runtime edge list.

    Returns one human-readable ``"<rule-id>: <message>"`` string per
    violation, empty if the graph is clean (module-level self-imports never
    occur and are skipped defensively).
    """
    rt = runtime_graph(graph)
    violations: list[str] = []
    for module in sorted(rt):
        for rule in ARCH_RULES:
            if not rule.applies_to(module):
                continue
            for target in sorted(rt[module]):
                if target == module or rule.permits(target):
                    continue
                if (rule.id, module, target) in GRANDFATHERED:
                    continue
                violations.append(
                    f"{rule.id}: `{module}` imports `{target}` — {rule.description}"
                )
    return violations


def check_cycles(graph: EdgeGraph) -> list[str]:
    """Return one message per runtime import cycle (empty if none)."""
    cycles = find_cycles(runtime_graph(graph))
    return [f"import-cycle: {' -> '.join(scc)} -> ..." for scc in cycles]


def check_freshness(graph: EdgeGraph, labels: dict[str, str], out_path: Path) -> str | None:
    """Return a violation message if ``out_path`` is stale, else ``None``.

    Regenerates the report in-memory and compares it to the checked-in file,
    ignoring the "_Generated <date>" line (which always differs run to run).
    """

    def _strip_generated(text: str) -> str:
        return "\n".join(
            line for line in text.splitlines() if not line.startswith("_Generated ")
        )

    if not out_path.exists():
        return f"docs-freshness: {out_path} does not exist — run `make graph`"

    current = _strip_generated(out_path.read_text(encoding="utf-8"))
    fresh = _strip_generated(render_report(graph, labels))
    if current.strip() != fresh.strip():
        return f"docs-freshness: {out_path} is stale — run `make graph` and commit the result"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="package root to analyze (default: src/sovereign)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Markdown file to write (default: docs/dependency-graph.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check mode: exit 1 on any runtime import cycle, ARCH_RULES "
            "violation, or stale docs/dependency-graph.md. Writes nothing."
        ),
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        parser.error(f"root is not a directory: {root}")

    modules = discover_modules(root)
    graph = build_graph(modules)
    labels = build_labels(modules, root)

    if args.check:
        violations = [
            *check_cycles(graph),
            *check_arch_rules(graph),
        ]
        freshness = check_freshness(graph, labels, args.out)
        if freshness is not None:
            violations.append(freshness)

        if violations:
            print(f"depgraph --check: {len(violations)} violation(s):")
            for v in violations:
                print(f"  - {v}")
            return 1
        print(f"depgraph --check: clean ({len(modules)} modules).")
        return 0

    report = render_report(graph, labels)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(modules)} modules).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

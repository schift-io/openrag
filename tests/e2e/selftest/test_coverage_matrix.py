"""Offline coverage-matrix & policy self-tests across E2E Tiers 1-4."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.selftest

_E2E_DIR = Path(__file__).resolve().parent.parent
_ALL_FEATURES: tuple[str, ...] = tuple(
    [f"f{i:02d}" for i in range(1, 35)] + ["f36", "f37"]
)
_FEATURE_SET = frozenset(_ALL_FEATURES)
_TIER_DIRS = ("tier1", "tier2", "tier3", "tier4")
_FORBIDDEN_VENDOR_TOKENS: tuple[str, ...] = (
    "-".join(("schift", "db")),
    "_".join(("schift", "db")),
    "-".join(("schift", "rag", "ce")),
    "_".join(("schift", "rag", "ce")),
)
_SKIP_CALL_NAMES = frozenset({"skip", "xfail", "importorskip", "SkipTest", "skipTest"})
_SKIP_MARKER_NAMES = frozenset({"skip", "skipif", "xfail"})

_COLLECT_SCRIPT = """
import json, sys, pytest

ini_path, e2e_dir, cache_dir, out_json, tier_arg = sys.argv[1:6]

class MatrixPlugin:
    def __init__(self):
        self.records = []

    def pytest_collection_modifyitems(self, config, items):
        self.records = [
            {
                "nodeid": item.nodeid,
                "markers": sorted({m.name for m in item.iter_markers()}),
            }
            for item in items
        ]

plugin = MatrixPlugin()
args = ["-c", ini_path, e2e_dir, "--collect-only", "-q", "-o", f"cache_dir={cache_dir}"]
if tier_arg:
    args.extend(["--tier", tier_arg])
rc = pytest.main(args, plugins=[plugin])
if rc != 0:
    raise SystemExit(rc)
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(plugin.records, fh)
"""


def _collect_via_pytest(tmp_path: Path, tier: str | None = None) -> list[dict[str, Any]]:
    label = tier or "unfiltered"
    cache_dir = tmp_path / f"pytest_cache_{label}"
    out_json = tmp_path / f"collected_{label}.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _COLLECT_SCRIPT,
            str(_E2E_DIR / "pytest.ini"),
            str(_E2E_DIR),
            str(cache_dir),
            str(out_json),
            tier or "",
        ],
        cwd=str(_E2E_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"pytest collection failed (tier={tier!r}, rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return json.loads(out_json.read_text(encoding="utf-8"))


def _marker_names_from_expr(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            names |= _marker_names_from_expr(elt)
        return names
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            # e.g. pytest.mark.f36(...)
            if (
                isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "pytest"
                and node.func.value.attr == "mark"
            ):
                names.add(node.func.attr)
        return names
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if (
            isinstance(node.value.value, ast.Name)
            and node.value.value.id == "pytest"
            and node.value.attr == "mark"
        ):
            names.add(node.attr)
    return names


def _parametrize_factor(decorators: list[ast.expr]) -> int:
    factor = 1
    for dec in decorators:
        if not isinstance(dec, ast.Call):
            continue
        m_names = _marker_names_from_expr(dec)
        if "parametrize" in m_names and len(dec.args) >= 2:
            argvalues = dec.args[1]
            if isinstance(argvalues, (ast.List, ast.Tuple)):
                factor *= max(1, len(argvalues.elts))
    return factor


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _scan_ast_e2e() -> dict[str, Any]:
    ast_tests: list[dict[str, Any]] = []
    skip_violations: list[str] = []
    crates_imports: list[str] = []
    vendor_violations: list[str] = []

    for tier_dir in _TIER_DIRS:
        tpath = _E2E_DIR / tier_dir
        if not tpath.is_dir():
            continue
        for py_file in sorted(tpath.glob("test_*.py")):
            src = py_file.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py_file))
            rel = str(py_file.relative_to(_E2E_DIR))

            mod_markers: set[str] = set()
            for stmt in tree.body:
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "pytestmark":
                            mod_markers |= _marker_names_from_expr(stmt.value)
                elif isinstance(stmt, ast.AnnAssign):
                    if isinstance(stmt.target, ast.Name) and stmt.target.id == "pytestmark" and stmt.value:
                        mod_markers |= _marker_names_from_expr(stmt.value)

            def _record_fn(
                fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
                extra_markers: set[str],
            ) -> None:
                if not fn_node.name.startswith("test_"):
                    return
                fn_markers = set(mod_markers) | set(extra_markers)
                for dec in fn_node.decorator_list:
                    fn_markers |= _marker_names_from_expr(dec)
                factor = _parametrize_factor(fn_node.decorator_list)
                ast_tests.append(
                    {
                        "file": rel,
                        "tier_dir": tier_dir,
                        "name": fn_node.name,
                        "markers": fn_markers,
                        "count": factor,
                        "lineno": fn_node.lineno,
                        "end_lineno": fn_node.end_lineno or fn_node.lineno,
                    }
                )

            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _record_fn(stmt, set())
                elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith("Test"):
                    cls_markers: set[str] = set()
                    for dec in stmt.decorator_list:
                        cls_markers |= _marker_names_from_expr(dec)
                    for sub in stmt.body:
                        if isinstance(sub, ast.Assign):
                            for tgt in sub.targets:
                                if isinstance(tgt, ast.Name) and tgt.id == "pytestmark":
                                    cls_markers |= _marker_names_from_expr(sub.value)
                    for sub in stmt.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            _record_fn(sub, cls_markers)

            # AST skip / xfail / importorskip scan for tier1..tier4
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    dotted = _dotted_name(node.func)
                    leaf = dotted.rsplit(".", 1)[-1] if dotted else ""
                    if leaf in _SKIP_CALL_NAMES and (
                        dotted.startswith(("pytest.", "unittest."))
                        or leaf in ("importorskip", "SkipTest")
                    ):
                        skip_violations.append(f"{rel}:{node.lineno}: call {dotted}()")
                    for kw in node.keywords:
                        if kw.arg == "marks":
                            bad_marks = _marker_names_from_expr(kw.value) & _SKIP_MARKER_NAMES
                            if bad_marks:
                                skip_violations.append(
                                    f"{rel}:{node.lineno}: param marks={sorted(bad_marks)}"
                                )
                elif isinstance(node, ast.Attribute):
                    if _dotted_name(node.value) == "pytest.mark" and node.attr in _SKIP_MARKER_NAMES:
                        skip_violations.append(f"{rel}:{node.lineno}: pytest.mark.{node.attr}")

    # Import & vendor/crates exclusion scan across ALL *.py under tests/e2e/
    f01_ranges_by_file: dict[str, list[tuple[int, int]]] = {}
    for t in ast_tests:
        if "f01" in t["markers"]:
            f01_ranges_by_file.setdefault(t["file"], []).append((t["lineno"], t["end_lineno"]))

    crates_tok = "cra" + "tes"
    for py_file in sorted(_E2E_DIR.rglob("*.py")):
        rel = str(py_file.relative_to(_E2E_DIR))
        src = py_file.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if crates_tok in alias.name.split("."):
                        crates_imports.append(f"{rel}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if crates_tok in mod.split("."):
                    crates_imports.append(f"{rel}:{node.lineno}: from {mod} import ...")

        f01_ranges = f01_ranges_by_file.get(rel, [])
        for lineno, line in enumerate(src.splitlines(), 1):
            in_f01 = any(start <= lineno <= end for start, end in f01_ranges)
            if in_f01:
                continue
            low = line.lower()
            for vtok in _FORBIDDEN_VENDOR_TOKENS:
                if vtok in low:
                    vendor_violations.append(f"{rel}:{lineno}: forbidden token {vtok!r}")
            # Also forbid bare workspace crate-dir path references outside f01 tests
            if f"{crates_tok}/" in low or f'"{crates_tok}"' in low or f"'{crates_tok}'" in low:
                vendor_violations.append(f"{rel}:{lineno}: {crates_tok} reference outside f01")

    return {
        "ast_tests": ast_tests,
        "skip_violations": skip_violations,
        "crates_imports": crates_imports,
        "vendor_violations": vendor_violations,
    }


@pytest.mark.selftest
def test_coverage_matrix_all_36_features_across_tier1_tier2_tier3(tmp_path: Path) -> None:
    collected = _collect_via_pytest(tmp_path, None)
    ast_info = _scan_ast_e2e()

    ast_total_tiers = sum(t["count"] for t in ast_info["ast_tests"])
    collected_tiers = [c for c in collected if any(c["nodeid"].startswith(f"{td}/") for td in _TIER_DIRS)]
    assert len(collected_tiers) == ast_total_tiers, (
        f"Pytest collected {len(collected_tiers)} tier items != AST count {ast_total_tiers}"
    )

    counts: dict[str, dict[str, int]] = {feat: {f"tier{i}": 0 for i in range(1, 5)} for feat in _ALL_FEATURES}
    for item in collected_tiers:
        nodeid = item["nodeid"]
        markers = set(item["markers"])
        tier_dir = nodeid.split("/", 1)[0]
        assert tier_dir in markers, f"{nodeid} missing @{tier_dir} marker (got {sorted(markers)})"
        feats = markers & _FEATURE_SET
        assert feats, f"{nodeid} missing any fXX feature marker"
        if tier_dir == "tier3":
            assert len(feats) >= 2, f"Tier 3 pairwise test {nodeid} must have >=2 feature markers, got {sorted(feats)}"
        for feat in feats:
            counts[feat][tier_dir] += 1

    for feat in _ALL_FEATURES:
        assert counts[feat]["tier1"] >= 5, f"{feat} has {counts[feat]['tier1']} < 5 tier1 tests"
        assert counts[feat]["tier2"] >= 5, f"{feat} has {counts[feat]['tier2']} < 5 tier2 tests"
        assert counts[feat]["tier3"] >= 1, f"{feat} has {counts[feat]['tier3']} < 1 tier3 pairwise test"
        assert counts[feat]["tier4"] >= 1, f"{feat} has {counts[feat]['tier4']} < 1 tier4 scenario"


@pytest.mark.selftest
def test_coverage_matrix_tier_totals_and_440_threshold(tmp_path: Path) -> None:
    unfiltered = _collect_via_pytest(tmp_path, None)
    tier_all = _collect_via_pytest(tmp_path, "all")

    by_tier = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0, "selftest": 0}
    for item in unfiltered:
        markers = set(item["markers"])
        for key in by_tier:
            if key in markers:
                by_tier[key] += 1

    assert by_tier["tier1"] >= 180, by_tier
    assert by_tier["tier2"] >= 180, by_tier
    assert by_tier["tier3"] >= 36, by_tier
    assert by_tier["tier4"] >= 18, by_tier
    assert by_tier["selftest"] >= 11, by_tier
    assert len(tier_all) >= 440, f"--tier all collected {len(tier_all)} < 440"
    assert len(unfiltered) >= 440, f"Total collected {len(unfiltered)} < 440"


@pytest.mark.selftest
def test_coverage_matrix_cli_tier_filtering_parity(tmp_path: Path) -> None:
    unfiltered = _collect_via_pytest(tmp_path, None)
    by_tier_expected: dict[str, set[str]] = {str(i): set() for i in range(1, 5)}
    for item in unfiltered:
        markers = set(item["markers"])
        for i in range(1, 5):
            if f"tier{i}" in markers:
                by_tier_expected[str(i)].add(item["nodeid"])

    for t in ("1", "2", "3", "4"):
        got = {item["nodeid"] for item in _collect_via_pytest(tmp_path, t)}
        assert got == by_tier_expected[t], f"--tier {t} mismatch"

    got_all = {item["nodeid"] for item in _collect_via_pytest(tmp_path, "all")}
    expected_all = set().union(*by_tier_expected.values())
    assert got_all == expected_all
    assert not any(nid.startswith("selftest/") for nid in got_all)


@pytest.mark.selftest
def test_coverage_matrix_zero_skips_xfails_in_tiers_1_to_4() -> None:
    ast_info = _scan_ast_e2e()
    assert ast_info["skip_violations"] == [], (
        f"Forbidden skip/xfail/importorskip constructs found in Tiers 1-4: {ast_info['skip_violations']}"
    )


@pytest.mark.selftest
def test_coverage_matrix_zero_crates_imports_and_vendor_exclusions() -> None:
    ast_info = _scan_ast_e2e()
    assert ast_info["crates_imports"] == [], (
        f"Forbidden crates imports in tests/e2e/: {ast_info['crates_imports']}"
    )
    assert ast_info["vendor_violations"] == [], (
        f"Forbidden vendor or crates references outside F01 tests: {ast_info['vendor_violations']}"
    )

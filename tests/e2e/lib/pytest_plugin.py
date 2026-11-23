"""Early-loaded pytest plugin registering OpenRAG E2E CLI options and tier filtering."""

from __future__ import annotations

import os

import pytest

DEFAULT_TARGET_ROOT = "/path/to/openrag"
_VALID_MODES = ("auto", "offline", "live")
_VALID_TIERS = ("1", "2", "3", "4", "all")
_TIER_MARKERS = frozenset({"tier1", "tier2", "tier3", "tier4"})


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("openrag-e2e", "OpenRAG E2E test harness options")
    group.addoption(
        "--tier",
        action="store",
        choices=list(_VALID_TIERS),
        default=None,
        help="Filter collected tests by tier (1, 2, 3, 4, or all; excludes selftest).",
    )
    env_mode = os.environ.get("OPENRAG_E2E_MODE", "auto").strip().lower()
    if env_mode not in _VALID_MODES:
        env_mode = "auto"
    group.addoption(
        "--mode",
        action="store",
        choices=list(_VALID_MODES),
        default=env_mode,
        help="Execution mode: auto, offline, or live (env: OPENRAG_E2E_MODE).",
    )
    group.addoption(
        "--target-root",
        action="store",
        default=os.environ.get("OPENRAG_ROOT", DEFAULT_TARGET_ROOT),
        help="OpenRAG repository root under test (env: OPENRAG_ROOT).",
    )
    group.addoption(
        "--api-url",
        action="store",
        default=os.environ.get("OPENRAG_API_URL"),
        help="Base URL of a running OpenRAG HTTP server (env: OPENRAG_API_URL).",
    )
    env_pg_port = os.environ.get("OPENRAG_PG_PORT", "0").strip()
    try:
        default_pg_port = int(env_pg_port) if env_pg_port else 0
    except ValueError:
        default_pg_port = 0
    group.addoption(
        "--pg-port",
        action="store",
        type=int,
        default=default_pg_port,
        help="TCP port for ephemeral PG17 cluster; 0 picks a free port (env: OPENRAG_PG_PORT).",
    )
    group.addoption(
        "--openrag-bin",
        action="store",
        default=os.environ.get("OPENRAG_BIN"),
        help="Explicit path to openrag executable (env: OPENRAG_BIN).",
    )


def filter_items_by_tier(
    items: list[pytest.Item], tier: str | None
) -> tuple[list[pytest.Item], list[pytest.Item]]:
    """Partition pytest items into (kept, deselected) according to `--tier`."""
    if tier is None:
        return list(items), []

    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    if tier in ("1", "2", "3", "4"):
        target_marker = f"tier{tier}"
        for item in items:
            if item.get_closest_marker(target_marker) is not None:
                kept.append(item)
            else:
                deselected.append(item)
        return kept, deselected

    if tier == "all":
        for item in items:
            if any(item.get_closest_marker(m) is not None for m in _TIER_MARKERS):
                kept.append(item)
            else:
                deselected.append(item)
        return kept, deselected

    raise ValueError(f"Unsupported --tier value: {tier!r}")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    tier = config.getoption("--tier")
    if tier is None:
        return
    kept, deselected = filter_items_by_tier(items, tier)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

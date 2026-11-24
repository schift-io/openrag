"""Shared pytest CLI options, tier filtering, and session/function fixtures for OpenRAG E2E."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

import pytest

# Ensure `from lib import ...` resolves regardless of whether pytest is invoked
# from the repository root or from `tests/e2e`.
_E2E_ROOT = Path(__file__).resolve().parent
if str(_E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(_E2E_ROOT))

from lib.cli_client import CliBinaryNotFoundError, OpenRagCli
from lib.fixtures import FixtureFactory
from lib.http_client import EphemeralOpenRagServer, OpenRagHttpClient
from lib.mcp_client import McpStdioClient
from lib.pg_harness import PgEphemeralCluster
from lib.plugin_client import PluginClient
from lib.pytest_plugin import (
    DEFAULT_TARGET_ROOT,
    _VALID_MODES,
    filter_items_by_tier,
)
from lib.web_client import WebClient


@dataclass(frozen=True)
class E2EConfig:
    """Resolved E2E test configuration from CLI flags and environment variables."""

    target_root: Path
    mode: Literal["auto", "offline", "live"]
    api_url: str | None
    pg_port: int
    openrag_bin: Path | None
    tier: str | None


def fail_missing_target(msg: str) -> NoReturn:
    """Non-skipping failure helper required for Tiers 1-4 when product targets are missing."""
    pytest.fail(msg, pytrace=False)


@pytest.fixture(scope="session")
def e2e_config(request: pytest.FixtureRequest) -> E2EConfig:
    raw_mode = str(request.config.getoption("--mode"))
    mode: Literal["auto", "offline", "live"] = (
        raw_mode if raw_mode in _VALID_MODES else "auto"  # type: ignore[assignment]
    )
    raw_bin = request.config.getoption("--openrag-bin")
    raw_api = request.config.getoption("--api-url")
    return E2EConfig(
        target_root=Path(str(request.config.getoption("--target-root"))).resolve(),
        mode=mode,
        api_url=str(raw_api).strip() if raw_api else None,
        pg_port=int(request.config.getoption("--pg-port") or 0),
        openrag_bin=Path(str(raw_bin)).resolve() if raw_bin else None,
        tier=request.config.getoption("--tier"),
    )


@pytest.fixture(scope="session")
def openrag_root(e2e_config: E2EConfig) -> Path:
    return e2e_config.target_root


@pytest.fixture(scope="session")
def openrag_cli(e2e_config: E2EConfig) -> OpenRagCli:
    env: dict[str, str] = {"OPENRAG_MODE": e2e_config.mode}
    return OpenRagCli(
        target_root=e2e_config.target_root,
        explicit_bin=e2e_config.openrag_bin,
        base_env=env,
    )


@pytest.fixture(scope="session")
def pg_ephemeral(e2e_config: E2EConfig) -> Iterator[PgEphemeralCluster]:
    keep_data = os.environ.get("OPENRAG_KEEP_PG") == "1"
    cluster = PgEphemeralCluster(
        port=e2e_config.pg_port,
        keep_data=keep_data,
    )
    try:
        yield cluster
    finally:
        cluster.stop()


@pytest.fixture(scope="session")
def api_client(
    e2e_config: E2EConfig,
    pg_ephemeral: PgEphemeralCluster,
) -> Iterator[OpenRagHttpClient]:
    if e2e_config.api_url:
        yield OpenRagHttpClient(e2e_config.api_url)
        return

    server_holder: list[EphemeralOpenRagServer] = []

    def _start_ephemeral() -> str:
        if server_holder:
            return server_holder[0].base_url
        db_url: str | None = os.environ.get("DATABASE_URL")
        if db_url is None and pg_ephemeral.is_started:
            db_url = pg_ephemeral.dsn()

        srv_mode = "offline" if e2e_config.mode in ("auto", "offline") else "live"
        srv = EphemeralOpenRagServer(
            target_root=e2e_config.target_root,
            database_url=db_url,
            explicit_bin=e2e_config.openrag_bin,
            mode=srv_mode,
        )
        try:
            srv.start()
        except CliBinaryNotFoundError as exc:
            fail_missing_target(str(exc))
        server_holder.append(srv)
        return srv.base_url

    client = OpenRagHttpClient(_base_url_resolver=_start_ephemeral)
    try:
        yield client
    finally:
        for srv in server_holder:
            srv.stop()


@pytest.fixture(scope="function")
def mcp_stdio(
    e2e_config: E2EConfig,
    openrag_cli: OpenRagCli,
) -> Iterator[Callable[..., McpStdioClient]]:
    spawned: list[McpStdioClient] = []

    def _factory(
        command: Sequence[str] | None = None,
        *,
        api_key: str | None = None,
        principal_id: str | None = None,
        prefer_standalone: bool = False,
        framing: Literal["ndjson", "content-length"] = "ndjson",
        env: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> McpStdioClient:
        if command is None:
            try:
                resolved_cmd: Sequence[str] = openrag_cli.mcp_stdio_command(
                    api_key=api_key,
                    principal_id=principal_id,
                    prefer_standalone=prefer_standalone,
                )
            except CliBinaryNotFoundError as exc:
                fail_missing_target(str(exc))
        else:
            resolved_cmd = command

        merged_env = {"OPENRAG_MODE": e2e_config.mode}
        if env:
            merged_env.update(env)

        client = McpStdioClient(
            resolved_cmd,
            framing=framing,
            env=merged_env,
            cwd=e2e_config.target_root if e2e_config.target_root.exists() else None,
            timeout=timeout,
        )
        spawned.append(client)
        return client

    try:
        yield _factory
    finally:
        for client in spawned:
            client.close()


@pytest.fixture(scope="session")
def plugin_client(openrag_root: Path) -> PluginClient:
    return PluginClient(openrag_root / "plugins")


@pytest.fixture(scope="session")
def web_client(openrag_root: Path, e2e_config: E2EConfig) -> WebClient:
    return WebClient(openrag_root, api_url=e2e_config.api_url)


@pytest.fixture(scope="function")
def sample_factory(tmp_path: Path) -> FixtureFactory:
    return FixtureFactory(tmp_path)

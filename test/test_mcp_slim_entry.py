"""The stdio MCP servers start without the gateway-only stack.

``kirocrew mcp-*`` must not import the whole CLI (aiohttp, boto3/botocore, awscrt,
slack_sdk) before it dispatches, and ``mcp-dashboard`` must not reach the same
stack through ``dashboard.chat_folders``. The import checks run in a fresh
interpreter, because ``sys.modules`` in the test process already holds
everything; the start-up steps are checked in process.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kiro_crew.platform as kc_platform
from kiro_crew import (
    _llama_lib_path,
    _ssl_compat,
    cli_log_files,
    mcp_entry,
    platform_compat,
    project_dir,
)
from kiro_crew.platform import tree_liveness
from kiro_crew.platform.context import PlatformCompositionError
from kiro_crew.subprocess_utf8 import UTF8_TEXT

SRC = str(Path(mcp_entry.__file__).resolve().parents[1])
HEAVY = ("aiohttp", "boto3", "botocore", "awscrt", "slack_sdk", "amazon_transcribe")
# Windows needs these in a child's environment: without SYSTEMROOT, asyncio's
# ``_overlapped`` fails to load (WinError 10106).
_HOST_VARS = ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP")


def _child_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in _HOST_VARS if k in os.environ}
    env.setdefault("PATH", "/usr/bin:/bin")
    env.update(
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),  # Path.home() on Windows
        KIROCREW_HOME=str(tmp_path / "home"),
        PYTHONPATH=SRC,
        PYTHONDONTWRITEBYTECODE="1",  # no .pyc left in the checkout
    )
    env.update(extra)
    return env


def _loaded_after(code: str, tmp_path: Path) -> set[str]:
    probe = (
        code
        + "\nimport json, sys\nprint(json.dumps(sorted({n.split('.')[0] for n in sys.modules})))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=_child_env(tmp_path),
        cwd=str(tmp_path),
        capture_output=True,
        **UTF8_TEXT,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("module", sorted({m for m, _ in mcp_entry.SLIM_SERVERS.values()}))
def test_mcp_server_module_skips_gateway_stack(module: str, tmp_path: Path) -> None:
    loaded = _loaded_after(f"import {module}", tmp_path)
    assert not loaded & set(HEAVY), sorted(loaded & set(HEAVY))


def test_slim_entry_module_scope_imports_nothing_from_the_package(tmp_path: Path) -> None:
    # Only the package root (whose __init__ is stdlib-only) may be present.
    probe = _loaded_after(
        "import kiro_crew.mcp_entry, sys\n"
        "assert not [n for n in sys.modules if n.startswith('kiro_crew.') and n != 'kiro_crew.mcp_entry'], "
        "[n for n in sys.modules if n.startswith('kiro_crew.')]",
        tmp_path,
    )
    assert "kiro_crew" in probe
    assert not probe & set(HEAVY)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["gateway"],
        ["mcp-core", "--help"],
        ["-v", "mcp-core"],
        ["mcp-mochi"],
        ["mcp-work", "extra"],
    ],
)
def test_other_argv_shapes_fall_back_to_the_cli(argv: list[str]) -> None:
    assert mcp_entry.slim_server(argv) is None
    assert mcp_entry.run_if_slim(argv) is False


@pytest.mark.parametrize("port", ["abc", "0", "70000"])
def test_bad_port_falls_back_so_the_cli_reports_it(
    monkeypatch: pytest.MonkeyPatch, port: str
) -> None:
    monkeypatch.setenv("KIROCREW_PORT", port)
    assert mcp_entry.slim_server(["mcp-core"]) is None


def test_valid_port_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIROCREW_PORT", "7777")
    assert mcp_entry._port_env_ok() is True
    monkeypatch.delenv("KIROCREW_PORT")
    assert mcp_entry._port_env_ok() is True


# ── start-up steps, in process ──


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list]:
    """Isolate every side effect of ``slim_server`` and record the steps it ran."""
    seen: dict[str, list] = {"boot": [], "log": [], "lock": []}
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KIROCREW_HOME", str(home))
    monkeypatch.setenv("KIROCREW_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("KIROCREW_SANDBOX_ACTIVE", "1")
    monkeypatch.setenv("KIROCREW_SANDBOX_LEVEL", "strict")
    monkeypatch.delenv("KIROCREW_PORT", raising=False)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_ssl_compat, "_ensure_ssl_certs", lambda: None)
    monkeypatch.setattr(_llama_lib_path, "drop_inherited_lib_path", lambda: None)
    monkeypatch.setattr(platform_compat, "ensure_utf8_console", lambda: None)
    monkeypatch.setattr(tree_liveness, "hold_running_tree_lock", lambda: seen["lock"].append(True))
    monkeypatch.setattr(mcp_entry, "_setup_logging", lambda d: seen["log"].append(Path(d)))
    monkeypatch.setattr(kc_platform, "boot_platform", lambda cfg: seen["boot"].append(cfg))
    monkeypatch.setattr(mcp_entry, "_stderr_is", lambda _p: False)
    return seen


def test_slim_server_runs_the_cli_steps_and_returns_the_entry(startup: dict) -> None:
    from kiro_crew import mcp_guide

    server = mcp_entry.slim_server(["mcp-guide"])
    assert server is mcp_guide.run_mcp_server
    assert startup["lock"] == [True] and len(startup["boot"]) == 1
    assert startup["log"] and startup["log"][0].is_dir()
    assert "KIROCREW_SANDBOX_ACTIVE" not in os.environ
    assert "KIROCREW_SANDBOX_LEVEL" not in os.environ
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1" and os.environ["OMP_NUM_THREADS"] == "1"


def test_project_dir_is_detected_when_unset(
    startup: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KIROCREW_PROJECT_DIR")
    monkeypatch.setattr(project_dir, "detect_project_dir", lambda saved: str(tmp_path / "p"))
    assert mcp_entry.slim_server(["mcp-guide"]) is not None
    assert os.environ["KIROCREW_PROJECT_DIR"] == str(tmp_path / "p")


def test_platform_composition_error_fails_closed(
    startup: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(cfg):
        raise PlatformCompositionError("overlay missing")

    monkeypatch.setattr(kc_platform, "boot_platform", _refuse)
    with pytest.raises(PlatformCompositionError):
        mcp_entry.slim_server(["mcp-guide"])


def test_other_platform_boot_errors_are_deferred(
    startup: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _flaky(cfg):
        raise RuntimeError("not yet")

    monkeypatch.setattr(kc_platform, "boot_platform", _flaky)
    assert mcp_entry.slim_server(["mcp-guide"]) is not None


def test_detached_stderr_falls_back_before_logging(
    startup: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_entry, "_stderr_is", lambda _p: True)
    assert mcp_entry.slim_server(["mcp-core"]) is None
    assert startup["log"] == [] and startup["boot"] == []


def test_run_if_slim_calls_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(mcp_entry, "slim_server", lambda argv: lambda: ran.append(argv[0]))
    assert mcp_entry.run_if_slim(["mcp-guide"]) is True
    assert ran == ["mcp-guide"]


def test_stderr_is_compares_the_open_file(tmp_path: Path) -> None:
    other = tmp_path / "other.log"
    other.write_text("", encoding="utf-8")
    assert mcp_entry._stderr_is(other) is False
    assert mcp_entry._stderr_is(tmp_path / "missing.log") is False


@pytest.fixture
def clean_kc_logger():
    logger = logging.getLogger("kiro_crew")
    handlers, level = list(logger.handlers), logger.level
    yield logger
    for h in logger.handlers[:]:
        if h not in handlers:
            logger.removeHandler(h)
            h.close()
    logger.setLevel(level)


def test_setup_logging_appends_to_gateway_log(
    clean_kc_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    before = list(clean_kc_logger.handlers)
    mcp_entry._setup_logging(tmp_path)
    added = [h for h in clean_kc_logger.handlers if h not in before]
    assert len(added) == 1
    assert Path(added[0].baseFilename) == tmp_path / "gateway.log"


def test_setup_logging_keeps_console_only_when_the_file_cannot_open(
    clean_kc_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _no_file(*a, **k):
        raise OSError("read-only")

    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    monkeypatch.setattr(cli_log_files, "_AppendOnlyLogFileHandler", _no_file)
    before = list(clean_kc_logger.handlers)
    mcp_entry._setup_logging(tmp_path)
    assert [h for h in clean_kc_logger.handlers if h not in before] == []


def test_slim_servers_match_the_cli_dispatch() -> None:
    import importlib

    for command, (module, func) in mcp_entry.SLIM_SERVERS.items():
        assert callable(getattr(importlib.import_module(module), func)), command
    cli_src = (Path(SRC) / "kiro_crew" / "cli.py").read_text(encoding="utf-8")
    for command, (module, func) in mcp_entry.SLIM_SERVERS.items():
        assert f'"{command}"' in cli_src
        assert module in cli_src, module


def test_mcp_guide_answers_initialize_without_the_gateway_stack(tmp_path: Path) -> None:
    """End to end: ``python -m kiro_crew mcp-guide`` speaks MCP and maps no aiohttp."""
    (tmp_path / "home").mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-m", "kiro_crew", "mcp-guide"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **UTF8_TEXT,
        env=_child_env(tmp_path, KIROCREW_PROJECT_DIR=str(tmp_path)),
        cwd=str(tmp_path),
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        }
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        resp = None
        for _ in range(50):
            line = proc.stdout.readline()
            if not line:
                break
            try:
                resp = json.loads(line)
            except ValueError:
                continue
            if resp.get("id") == 1:
                break
        assert resp is not None and "result" in resp, (resp, proc.stderr)
        if sys.platform.startswith("linux"):
            maps = Path(f"/proc/{proc.pid}/maps").read_text()
            assert "aiohttp" not in maps
            assert "awscrt" not in maps
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait(timeout=30)

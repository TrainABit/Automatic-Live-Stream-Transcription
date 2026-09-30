"""The suite's own rules (opt-in markers, zero-selection failure, network guard) are tested
by running a small throwaway project that reuses this repository's ``conftest.py``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")

PROJECT_TEST = """
import socket
import pytest

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.network
def test_network():
    pass

def test_plain():
    pass

def test_remote_socket_is_blocked():
    with pytest.raises(OSError, match="network blocked"):
        socket.create_connection(("203.0.113.9", 80), timeout=1)

def test_loopback_is_allowed():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname(), timeout=2)
    client.close()
    server.close()
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "conftest.py").write_text(CONFTEST.read_text())
    (tmp_path / "test_sample.py").write_text(PROJECT_TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    slow: slow\n    network: network\n"
    )
    return tmp_path


def _run(project: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("LST_RUN_")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rs", *args],
        cwd=project,
        capture_output=True,
        text=True,
        env={**clean, **env},
        timeout=120,
    )


def test_slow_and_network_are_skipped_by_default(project: Path):
    result = _run(project)
    assert result.returncode == 0, result.stdout
    assert "3 passed, 2 skipped" in result.stdout
    assert "LST_RUN_SLOW" in result.stdout
    assert "LST_RUN_NETWORK" in result.stdout


def test_environment_switches_opt_in(project: Path):
    result = _run(project, LST_RUN_SLOW="1")
    assert "4 passed, 1 skipped" in result.stdout


def test_marker_selection_runs_only_that_marker(project: Path):
    result = _run(project, "-m", "slow")
    assert result.returncode == 0
    assert "1 passed" in result.stdout


def test_a_marker_selection_that_matches_nothing_fails(project: Path):
    result = _run(project, "-m", "nonexistent_marker")
    assert result.returncode != 0
    assert "selected 0 tests" in result.stdout + result.stderr

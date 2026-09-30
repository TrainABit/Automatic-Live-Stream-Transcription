"""Every module imports on its own, as the first import of a fresh interpreter.

One test process imports modules in a single order and so hides import cycles
that bite any other entry point. Each module is imported first in its own
subprocess so that no earlier import can mask a cycle.
"""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import livestream_transcriber


def _modules() -> list[str]:
    names = [livestream_transcriber.__name__]
    for info in pkgutil.walk_packages(
        livestream_transcriber.__path__, prefix=f"{livestream_transcriber.__name__}."
    ):
        if info.name.rsplit(".", 1)[-1] == "__main__":
            continue  # running it starts the CLI
        names.append(info.name)
    return sorted(names)


def _import_alone(module: str) -> tuple[str, int, str]:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=120,
    )
    lines = proc.stderr.strip().splitlines()
    return module, proc.returncode, lines[-1] if lines else ""


def test_the_module_list_is_complete():
    modules = _modules()
    for expected in (
        "livestream_transcriber.config",
        "livestream_transcriber.models",
        "livestream_transcriber.redact",
        "livestream_transcriber.resilience.storage",
    ):
        assert expected in modules
    assert len(modules) >= 15


def test_every_module_imports_first_in_a_fresh_interpreter():
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        results = list(pool.map(_import_alone, _modules()))
    failures = [f"{name}: {last}" for name, code, last in results if code != 0]
    assert failures == []

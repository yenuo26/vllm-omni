# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Only whole-tree SIGKILL may leave named FIFOs; other resources stay strict."""

from __future__ import annotations

import ast
import stat
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cleanup():
    module = ModuleType("cleanup_under_test")
    module.__dict__.update(Any=Any, Sequence=Sequence, stat_module=stat, pytest=pytest)
    root = Path(__file__).parent
    for path, names in [
        (
            root / "helpers.py",
            {"_remaining_linux_shm_objects", "assert_no_server_tree_process_residual_and_gpu_release"},
        ),
        (root / "scenarios/process_kill.py", {"_assert_cleanup"}),
    ]:
        nodes: list[ast.stmt] = [
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("allow_fifo", [False, True])
@pytest.mark.parametrize("resource", ["fifo", "shm", "fd", "gpu", "process"])
def test_fifo_exception_never_disables_other_resource_checks(cleanup, allow_fifo, resource):
    mode = stat.S_IFIFO if resource == "fifo" else stat.S_IFREG
    cleanup.Path = lambda name: SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=1, st_ino=2, st_mode=mode))
    clock = [0.0]
    cleanup.time = SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    cleanup.logger = Mock()
    cleanup.list_alive_pids = lambda pids: [42] if resource == "process" else []
    cleanup.query_gpu_compute_pid_used_memory_mb = lambda: {42: 100} if resource == "gpu" else {}
    cleanup._linux_fd_counts = lambda pids: {42: 1} if resource == "fd" else {}
    server = SimpleNamespace(
        reliability_fault_snapshot={
            "tree_pids": [42],
            "shm_objects": [("owned", 1, 2)] if resource in {"fifo", "shm"} else [],
        }
    )
    if allow_fifo and resource == "fifo":
        cleanup.assert_no_server_tree_process_residual_and_gpu_release(
            server, scenario="tree", timeout_sec=1, allow_fifo_residual=True
        )
        assert clock[0] == 0.0
        cleanup.logger.warning.assert_called_once()
    else:
        with pytest.raises(AssertionError):
            cleanup.assert_no_server_tree_process_residual_and_gpu_release(
                server, scenario="strict", timeout_sec=1, allow_fifo_residual=allow_fifo
            )


@pytest.mark.parametrize(
    "scenario,allowed",
    [
        ("kill_serve_tree_no_load_sigkill", True),
        ("kill_serve_tree_with_load_sigkill", True),
        ("kill_serve_root_no_load_sigkill", False),
    ],
)
def test_only_tree_crash_selects_fifo_exception(cleanup, scenario, allowed):
    cleanup.worker_residual_timeout_after_kill_signal = lambda signal: 30
    cleanup.assert_no_server_tree_process_residual_and_gpu_release = Mock()
    server = object()
    cleanup._assert_cleanup(server, scenario=scenario, signal_name="SIGKILL")
    cleanup.assert_no_server_tree_process_residual_and_gpu_release.assert_called_once_with(
        server,
        scenario=scenario,
        timeout_sec=30,
        allow_fifo_residual=allowed,
    )

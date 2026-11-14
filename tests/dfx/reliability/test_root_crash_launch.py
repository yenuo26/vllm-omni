# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Root-only startup crashes must not pass via automatic harness rollback."""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def root_launch():
    path = Path(__file__).with_name("scenarios") / "launch.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "ServeRootCrashAtLaunch"
    )
    module = ModuleType("root_launch_under_test")
    module.__dict__.update(
        Any=Any,
        FaultModel=Any,
        FaultScenario=object,
        threading=threading,
        time=time,
        LAUNCH_FAILURE_TIMEOUT_SEC=0.02,
        signal=SimpleNamespace(SIGKILL=9),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize(
    "outcome,expected_error",
    [
        ("clean", None),
        ("no_worker", "no root crash injected"),
        ("early_failure", "no root crash injected"),
        ("ready_before", "no root crash injected"),
        ("ready_after", "did not report failure"),
        ("startup_hang", "post-root-crash deadline"),
        ("resource_leak", "orphan resource residual"),
    ],
)
def test_root_startup_crash_requires_worker_and_checks_resources_before_teardown(root_launch, outcome, expected_error):
    events: list[str] = []
    release = threading.Event()

    class Server:
        proc = object()

        def __enter__(self):
            pytest.fail("__enter__ rollback would hide orphan processes")

        def _start_server(self):
            if outcome == "early_failure":
                raise RuntimeError("startup failed without injection")
            if outcome == "ready_before":
                return
            assert release.wait(3), "startup thread was not released"
            if outcome != "ready_after":
                raise RuntimeError("server exited after root died")

        def _owns_listening_port(self):
            return False

        def __exit__(self, *args):
            events.append("teardown")
            release.set()

    server = Server()
    root_launch.start_server_from_params = Mock(return_value=server)
    root_launch._list_server_process_tree = lambda server: [10, 12] if outcome == "no_worker" else [10, 11, 12]
    root_launch._safe_proc_info = lambda pid: ("stage0_worker" if pid == 11 else "resource_tracker", "")

    def snapshot(server):
        events.append("snapshot")
        return {"root_pids": [10], "tree_pids": [10, 11, 12]}

    root_launch._capture_server_fault_snapshot = snapshot
    root_launch.escalate_live_pids = Mock()

    def kill(pid, sig):
        assert pid == 10 and sig == 9
        assert events == ["snapshot"]
        events.append("root_killed")
        if outcome != "startup_hang":
            release.set()

    root_launch.os = SimpleNamespace(kill=kill)
    root_launch.assert_launch_failure_observable = Mock(side_effect=lambda exc: events.append("failure_seen"))
    root_launch.client_for_server = lambda server: server
    root_launch.assert_post_fault_api_unreachable = Mock(side_effect=lambda *a, **k: events.append("api_down"))

    def cleanup(server, **kwargs):
        assert events[-1] == "api_down" and "teardown" not in events
        assert kwargs == {"scenario": "serve_root_crash_at_launch", "timeout_sec": 30}
        events.append("resources_checked")
        if outcome == "resource_leak":
            raise AssertionError("orphan resource residual")

    root_launch.assert_no_server_tree_process_residual_and_gpu_release = cleanup
    params = SimpleNamespace(startup_timeout=0.08)
    if expected_error:
        with pytest.raises(AssertionError, match=expected_error):
            root_launch.ServeRootCrashAtLaunch().run(SimpleNamespace(worker_pattern="stage0_worker"), params)
    else:
        root_launch.ServeRootCrashAtLaunch().run(SimpleNamespace(worker_pattern="stage0_worker"), params)
        assert events == ["snapshot", "root_killed", "failure_seen", "api_down", "resources_checked", "teardown"]
    assert events[-1] == "teardown"
    if outcome in {"ready_after", "startup_hang", "resource_leak"}:
        root_launch.escalate_live_pids.assert_called_once_with([10, 11, 12], grace_sec=0.0)
    else:
        root_launch.escalate_live_pids.assert_not_called()
    root_launch.start_server_from_params.assert_called_once_with(params)

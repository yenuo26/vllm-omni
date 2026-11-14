# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for pre-readiness stage fault injection."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.dfx.reliability.helpers import FaultModel
from tests.dfx.reliability.scenarios import launch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LaunchModel(FaultModel):
    worker_pattern = "stage0"


@pytest.mark.parametrize("finished", [False, True])
def test_launch_stage_injection_excludes_root_and_checks_readiness(monkeypatch, finished):
    monkeypatch.setattr(launch, "signal", SimpleNamespace(SIGKILL=9))
    events: list[str | tuple[int, int]] = []
    server = object()
    done = threading.Event()
    if finished:
        done.set()
    monkeypatch.setattr(launch, "_list_server_process_tree", lambda _: [10, 11, 12])
    monkeypatch.setattr(launch, "_safe_proc_info", lambda _: ("StageEngineCoreProc_stage0", ""))
    monkeypatch.setattr(launch, "_capture_server_fault_snapshot", lambda _: events.append("snapshot"))
    monkeypatch.setattr(launch.os, "kill", lambda pid, sig: events.append((pid, sig)))
    if finished:
        with pytest.raises(AssertionError, match="startup finished"):
            launch._kill_starting_stage(server, "stage0", done)
        assert events == ["snapshot"]
    else:
        assert launch._kill_starting_stage(server, "stage0", done)
        assert events == ["snapshot", (11, 9)]


def test_launch_stage_injection_never_falls_back_to_unrelated_processes(monkeypatch):
    monkeypatch.setattr(launch, "_list_server_process_tree", lambda _: [10, 11])
    monkeypatch.setattr(launch, "_safe_proc_info", lambda _: ("unrelated", ""))

    def unexpected_kill(*args):
        pytest.fail("no matching descendant must not cause a kill")

    monkeypatch.setattr(launch.os, "kill", unexpected_kill)
    assert not launch._kill_starting_stage(object(), "stage0", threading.Event())


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("failure", None),
        ("early_failure", "no stage fault injected"),
        ("ready_before_kill", "no stage fault injected"),
        ("ready_after_kill", "server became ready"),
        ("missing_stage", "no stage fault injected"),
        ("startup_hang", "startup did not terminate"),
    ],
)
def test_stage_crash_launch_requires_injection_failure_and_cleanup(monkeypatch, outcome, expected_error):
    events = []
    release = threading.Event()

    class Server:
        proc = object()

        def __enter__(self):
            if outcome == "ready_before_kill":
                return self
            if outcome != "early_failure":
                assert release.wait(5), "test startup thread was not released"
            if outcome == "ready_after_kill":
                return self
            # Real OmniServer.__enter__ rolls back before re-raising.
            events.append("rollback")
            raise RuntimeError("server exited during startup")

        def __exit__(self, *args):
            events.append("exit")

    starting = Server()
    monkeypatch.setattr(launch, "LAUNCH_FAILURE_TIMEOUT_SEC", 0.1)
    factory = Mock(return_value=starting)
    monkeypatch.setattr(launch, "start_server_from_params", factory)

    def inject(*args):
        if outcome in {"early_failure", "ready_before_kill"}:
            args[2].wait(1)
            return False
        if outcome == "missing_stage":
            return False
        events.append("stage_killed")
        if outcome != "startup_hang":
            release.set()
        return True

    monkeypatch.setattr(launch, "_kill_starting_stage", inject)
    monkeypatch.setattr(launch, "make_server_tree_kill_fault_injector", lambda **kwargs: lambda _: release.set())
    monkeypatch.setattr(
        launch, "assert_no_server_tree_process_residual_and_gpu_release", lambda *args, **kwargs: events.append("clean")
    )
    if expected_error is not None:
        with pytest.raises(AssertionError, match=expected_error):
            launch.PartialCrashAtLaunch().run(_LaunchModel(), object())
    else:
        launch.PartialCrashAtLaunch().run(_LaunchModel(), object())
        assert events == ["stage_killed", "rollback", "clean"]

    assert factory.call_count == 1


def test_stage_crash_launch_cleanup_failure_propagates(monkeypatch):
    release = threading.Event()

    class Server:
        proc = object()

        def __enter__(self):
            assert release.wait(5)
            raise RuntimeError("server exited")

    monkeypatch.setattr(launch, "start_server_from_params", lambda *args, **kwargs: Server())

    def inject(*args):
        release.set()
        return True

    monkeypatch.setattr(launch, "_kill_starting_stage", inject)

    def residual(*args, **kwargs):
        raise AssertionError("shared memory residual")

    monkeypatch.setattr(launch, "assert_no_server_tree_process_residual_and_gpu_release", residual)
    with pytest.raises(AssertionError, match="shared memory residual"):
        launch.PartialCrashAtLaunch().run(_LaunchModel(), object())

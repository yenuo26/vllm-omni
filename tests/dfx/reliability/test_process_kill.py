# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for shutdown-time stage failure sequencing and deadlines."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.dfx.reliability.helpers import FaultModel
from tests.dfx.reliability.scenarios import process_kill

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("success", None),
        ("no_worker", "no stage matched"),
        ("already_exited", "target stage exited"),
        ("zombie", "target stage exited"),
        ("disappeared", "target stage disappeared"),
        ("residual", "resource residual"),
        ("late_cleanup", "shutdown did not finish"),
        ("slow_injection", "grace period before cleanup"),
    ],
)
def test_stage_exit_requires_shutdown_before_kill_and_no_timeout_escalation(monkeypatch, outcome, expected_error):
    events = []
    clock = [0.0]

    class Model(FaultModel):
        isolation_pattern = "stage1"

        def warmup(self, client, server) -> None:
            events.append("warmup")

    model = Model()
    server = object()
    client = object()
    monkeypatch.setattr(process_kill, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(process_kill, "SIGTERM_CLEANUP_GRACE_SEC", 60.0)
    monkeypatch.setattr(process_kill, "_list_server_process_tree", lambda _: [10, 11])
    # Even if the root matches, it must never be selected as the stage target.
    monkeypatch.setattr(
        process_kill, "_safe_proc_info", lambda pid: ("other" if outcome == "no_worker" and pid == 11 else "stage1", "")
    )

    class Worker:
        def is_running(self):
            return outcome != "already_exited"

        def status(self):
            return process_kill.psutil.STATUS_ZOMBIE if outcome == "zombie" else "running"

        def kill(self):
            assert events[-1] == "SIGTERM"
            if outcome == "disappeared":
                raise process_kill.psutil.NoSuchProcess(11)
            events.append("stage_SIGKILL")
            if outcome == "slow_injection":
                clock[0] = 61.0

    def process(pid):
        assert pid == 11
        return Worker()

    monkeypatch.setattr(process_kill.psutil, "Process", process)

    def root_injector(**kwargs):
        assert kwargs == {"signal_name": "SIGTERM", "post_kill_wait_seconds": 0.0}
        return lambda _: events.append("SIGTERM")

    monkeypatch.setattr(process_kill, "make_server_root_kill_fault_injector", root_injector)

    def cleanup(actual_server, *, scenario, timeout_sec):
        assert actual_server is server
        assert events[-1] == "stage_SIGKILL"
        assert timeout_sec == 60.0
        events.append("cleanup")
        if outcome == "residual":
            raise AssertionError("resource residual")
        clock[0] = 61.0 if outcome == "late_cleanup" else 5.0

    monkeypatch.setattr(process_kill, "assert_no_server_tree_process_residual_and_gpu_release", cleanup)
    monkeypatch.setattr(
        process_kill, "assert_post_fault_api_unreachable", lambda *args, **kwargs: events.append("api_gone")
    )

    def unexpected_signal(*args, **kwargs):
        pytest.fail("must not escalate or kill the tree to pass shutdown")

    monkeypatch.setattr(process_kill, "make_server_tree_kill_fault_injector", unexpected_signal)
    monkeypatch.setattr(process_kill, "make_process_kill_fault_injector", unexpected_signal)
    if expected_error is None:
        process_kill.PartialCrashAtExit().run(model, server, client)
        assert events == ["warmup", "SIGTERM", "stage_SIGKILL", "cleanup", "api_gone"]
    else:
        with pytest.raises((AssertionError, pytest.fail.Exception), match=expected_error):
            process_kill.PartialCrashAtExit().run(model, server, client)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Faults that happen while the server is starting, before the fixture would mark it ready."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

import pytest

from tests.dfx.conftest import resolve_oom_device_spec
from tests.dfx.reliability.helpers import (
    LAUNCH_FAILURE_TIMEOUT_SEC,
    FaultModel,
    FaultScenario,
    _capture_server_fault_snapshot,
    _list_server_process_tree,
    _safe_proc_info,
    assert_launch_failure_observable,
    assert_no_server_tree_process_residual_and_gpu_release,
    client_for_server,
    inject_gpu_oom,
    make_server_tree_kill_fault_injector,
    start_server_from_params,
    stop_gpu_oom_hogs,
)


def _recover(model: FaultModel, params: Any) -> None:
    server = start_server_from_params(params)
    try:
        server.__enter__()
        model.warmup(client_for_server(server), server)
    finally:
        server.__exit__(None, None, None)


class GpuOomAtLaunch(FaultScenario):
    """Fill GPU memory, require startup to fail observably, then start cleanly."""

    server_set = "launch"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        config = model.hard_oom_config
        device = config.get("device") or resolve_oom_device_spec(config, params.stage_config_path)
        kwargs: dict[str, Any] = {
            "device": device,
            "target_mem_ratio": config["target_mem_ratio"],
            "hold_seconds": config["hold_seconds"],
            "startup_timeout_sec": config["startup_timeout_sec"],
            "strict": config["strict"],
        }
        if "max_free_bytes" in config and config["max_free_bytes"] is not None:
            kwargs["max_free_bytes"] = config["max_free_bytes"]
        handle = inject_gpu_oom(**kwargs)
        try:
            poisoned = start_server_from_params(params, startup_timeout=LAUNCH_FAILURE_TIMEOUT_SEC)
            try:
                poisoned.__enter__()
            except Exception as exc:
                assert_launch_failure_observable(exc)
            else:
                poisoned.__exit__(None, None, None)
                pytest.fail("server became ready while GPU memory was exhausted")
        finally:
            stop_gpu_oom_hogs(handle)
        _recover(model, params)


def _kill_starting_stage(server: Any, pattern: str, finished: threading.Event) -> bool:
    """Kill one matching descendant before startup completes; never target the root."""
    tree = _list_server_process_tree(server)
    for pid in tree[1:]:
        name, cmdline = _safe_proc_info(pid)
        if pattern not in f"{name} {cmdline}":
            continue
        _capture_server_fault_snapshot(server)
        assert not finished.is_set(), "startup finished before stage fault injection"
        sigkill = getattr(signal, "SIGKILL", None)
        if sigkill is None:
            raise RuntimeError("SIGKILL is required for stage fault injection")
        try:
            os.kill(pid, sigkill)
        except ProcessLookupError:
            continue
        print(f"[stage-crash-at-launch] killed stage pid={pid} pattern={pattern} name={name}", flush=True)
        return True
    return False


class PartialCrashAtLaunch(FaultScenario):
    """Kill one stage during startup, require bounded failure, cleanup, and restart.

    The injection is scoped to descendants matching the model's worker pattern.
    Startup failure before injection or a ready server is a failure, not coverage.
    The server harness owns rollback on failed entry; recovery runs only after
    that rollback and the resource-residual assertions have completed.
    """

    server_set = "launch"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        starting = start_server_from_params(params, startup_timeout=LAUNCH_FAILURE_TIMEOUT_SEC)
        errors: list[BaseException] = []
        finished = threading.Event()
        ready = threading.Event()

        def _enter() -> None:
            try:
                starting.__enter__()
                ready.set()
            except BaseException as exc:  # noqa: BLE001 - propagate the startup thread's failure below
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=_enter, daemon=True)
        thread.start()
        injected = False
        try:
            deadline = time.monotonic() + LAUNCH_FAILURE_TIMEOUT_SEC
            while time.monotonic() < deadline and not finished.is_set():
                if _kill_starting_stage(starting, model.worker_pattern, finished):
                    injected = True
                    break
                time.sleep(0.05)
            assert injected, (
                f"no stage fault injected before startup completed: pattern={model.worker_pattern!r}, "
                f"ready={ready.is_set()}, errors={errors!r}"
            )
            thread.join(LAUNCH_FAILURE_TIMEOUT_SEC)
            assert not thread.is_alive(), "startup did not terminate within the post-fault deadline"
            assert not ready.is_set(), "server became ready after a stage was killed during startup"
            assert errors, "startup did not report the injected stage failure"
            assert_launch_failure_observable(errors[0])
        finally:
            # Emergency cleanup cannot turn a startup timeout into a passing case.
            if thread.is_alive() and starting.proc is not None:
                make_server_tree_kill_fault_injector(signal_name="SIGKILL")(starting)
                thread.join(30)
            if ready.is_set():
                starting.__exit__(None, None, None)
            assert not thread.is_alive(), "startup thread survived emergency process-tree cleanup"
        assert_no_server_tree_process_residual_and_gpu_release(
            starting, scenario="stage_crash_at_launch", timeout_sec=60.0
        )
        _recover(model, params)


class FullCrashAtLaunch(FaultScenario):
    """Kill the process tree as soon as it exists, then require a clean restart."""

    server_set = "launch"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        starting = start_server_from_params(params)
        errors: list[BaseException] = []

        def _enter() -> None:
            try:
                starting.__enter__()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=_enter)
        thread.start()
        deadline = time.monotonic() + 90.0
        while starting.proc is None and not errors and time.monotonic() < deadline:
            time.sleep(0.1)
        if errors and starting.proc is None:
            thread.join(30)
            assert_launch_failure_observable(errors[0])
        else:
            assert starting.proc is not None, "server process did not appear within 90s"
            make_server_tree_kill_fault_injector(signal_name="SIGKILL", post_kill_wait_seconds=0.0)(starting)
            thread.join(120)
            if errors:
                assert_launch_failure_observable(errors[0])
            else:
                starting.__exit__(None, None, None)
        _recover(model, params)

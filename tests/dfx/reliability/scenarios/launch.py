# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Faults that happen while the server is starting, before the fixture would mark it ready."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tests.dfx.conftest import resolve_oom_device_spec
from tests.dfx.reliability.helpers import (
    LAUNCH_FAILURE_TIMEOUT_SEC,
    FaultModel,
    FaultScenario,
    assert_launch_failure_observable,
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

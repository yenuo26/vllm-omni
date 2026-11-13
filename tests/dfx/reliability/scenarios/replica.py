# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kill one replica and require another replica to keep serving."""

from __future__ import annotations

from typing import Any

import pytest

from tests.dfx.reliability.helpers import (
    INFLIGHT_INJECTION_REQUEST_COUNT,
    INFLIGHT_INJECTION_REQUEST_RATE,
    FaultModel,
    FaultScenario,
    client_for_server,
    kill_bad_replica,
    relaunch_server,
    run_fault_injection_with_rate_load,
    shutdown_serve_root_with_escalation,
)


def _kill(model: FaultModel, server: Any) -> None:
    kill_bad_replica(
        server,
        pattern=model.bad_replica_pattern,
        devices=model.bad_replica_devices,
        signal_name="SIGKILL",
    )


def _assert_survivor_serves(model: FaultModel, client: Any, server: Any) -> None:
    last_error: BaseException | None = None
    for _ in range(6):
        try:
            model.warmup(client, server)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        else:
            return
    pytest.fail(f"no surviving replica accepted a request: {last_error!r}")


def _relaunch_and_probe(model: FaultModel, server: Any) -> None:
    recovered = relaunch_server(server)
    try:
        model.warmup(client_for_server(recovered), recovered)
    finally:
        recovered.__exit__(None, None, None)


class ReplicaCrashIdle(FaultScenario):
    """After one idle replica dies, another replica still accepts a request."""

    server_set = "replica"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        del signal_name
        model.warmup(client, server)
        _kill(model, server)
        _assert_survivor_serves(model, client, server)


class ReplicaCrashWithLoad(FaultScenario):
    """A replica killed under load fails in-flight work, and a survivor still serves."""

    server_set = "replica"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        del signal_name
        model.warmup(client, server)
        load_result = run_fault_injection_with_rate_load(
            submit_request=lambda: model.submit_serve_load(client, server),
            inject_fault=lambda: _kill(model, server),
            num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
            request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
            completion_timeout_sec=model.with_load_timeout_sec,
        )
        assert load_result["failure_observed"], (
            f"[replica with-load] expected at least one in-flight failure; load_result={load_result}"
        )
        _assert_survivor_serves(model, client, server)


class ReplicaCrashAtExit(FaultScenario):
    """After one replica dies, shutdown finishes or escalates, and a new server can start."""

    server_set = "replica"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        del signal_name
        model.warmup(client, server)
        _kill(model, server)
        _assert_survivor_serves(model, client, server)
        shutdown_serve_root_with_escalation(server, scenario="replica_crash_at_exit")
        _relaunch_and_probe(model, server)

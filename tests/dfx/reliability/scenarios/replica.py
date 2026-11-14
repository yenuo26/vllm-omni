# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kill one replica and require another replica to keep serving."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import psutil

from tests.dfx.reliability.helpers import (
    INFLIGHT_INJECTION_REQUEST_COUNT,
    INFLIGHT_INJECTION_REQUEST_RATE,
    LAUNCH_FAILURE_TIMEOUT_SEC,
    FaultModel,
    FaultScenario,
    _capture_server_fault_snapshot,
    _safe_proc_info,
    assert_no_server_tree_process_residual_and_gpu_release,
    client_for_server,
    kill_bad_replica,
    make_server_tree_kill_fault_injector,
    relaunch_server,
    run_fault_injection_with_rate_load,
    shutdown_serve_root_with_escalation,
    start_server_from_params,
)
from tests.dfx.reliability.scenarios.launch import _kill_starting_stage


def _kill(model: FaultModel, server: Any) -> None:
    snapshot = _capture_server_fault_snapshot(server)
    stage_prefix = model.bad_replica_pattern.rsplit("_replica", 1)[0] + "_replica"
    bad_pids = []
    survivors = []
    for pid in snapshot["tree_pids"][1:]:
        name, cmdline = _safe_proc_info(pid)
        identity = f"{name} {cmdline}"
        if model.bad_replica_pattern in identity:
            bad_pids.append(pid)
        elif stage_prefix in identity:
            survivors.append(psutil.Process(pid))
    assert bad_pids, f"no target replica matched {model.bad_replica_pattern!r}"
    assert survivors, "no sibling replica found before injection"
    kill_bad_replica(
        server,
        pattern=model.bad_replica_pattern,
        devices=model.bad_replica_devices,
        signal_name="SIGKILL",
    )
    # Scope live-server cleanup to the killed replica. Shared SHM objects can
    # still be owned by survivors; the full-tree teardown checks those later.
    killed = SimpleNamespace(reliability_fault_snapshot={"tree_pids": bad_pids, "shm_objects": []})
    assert_no_server_tree_process_residual_and_gpu_release(killed, scenario="replica_crash", timeout_sec=30)
    _assert_survivors_alive(survivors)
    server.reliability_surviving_replicas = survivors


def _assert_survivors_alive(survivors: list[Any]) -> None:
    for process in survivors:
        assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE, (
            f"healthy sibling replica exited: pid={process.pid}"
        )


def _assert_survivor_serves(model: FaultModel, client: Any, server: Any) -> None:
    # Allow bounded fault detection, not business-request failures hidden by
    # retry. Health is an external convergence signal, not proof of eviction.
    deadline = time.monotonic() + 30.0
    while True:
        response = client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0]
        assert time.monotonic() < deadline, "replica health did not converge within 30 seconds"
        if response.status_code == 200:
            break
        assert response.status_code == 503, f"unexpected health status: {response.status_code}"
        time.sleep(0.5)
    for index in range(6):
        # Every request must pass: an alternating good/dead routing bug fails.
        model.warmup(client, server)
        _assert_survivors_alive(getattr(server, "reliability_surviving_replicas", []))
        print(f"[replica-crash] survivor request {index + 1}/6 passed", flush=True)


def _relaunch_and_probe(model: FaultModel, server: Any) -> None:
    recovered = relaunch_server(server)
    try:
        model.warmup(client_for_server(recovered), recovered)
    finally:
        recovered.__exit__(None, None, None)


class ReplicaCrashIdle(FaultScenario):
    """An idle replica dies, its PID resources clear, and six later requests all pass.

    Health and business probes verify externally observable isolation, not the
    internal routing table. No replacement replica or long-term leak test runs.
    """

    server_set = "replica"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        del signal_name
        model.warmup(client, server)
        _kill(model, server)
        _assert_survivor_serves(model, client, server)


class ReplicaCrashWithLoad(FaultScenario):
    """Kill a replica under multimodal load, then require stable survivor service.

    At least one in-flight request must fail. After bounded health convergence,
    all six subsequent multimodal requests must succeed without retries.
    """

    server_set = "replica"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        del signal_name
        model.warmup(client, server)
        load_result = run_fault_injection_with_rate_load(
            submit_request=lambda: model.submit_worker_load(client, server),
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


class ReplicaCrashAtLaunch(FaultScenario):
    """A startup replica crash must not prevent the original instance becoming ready.

    Require degraded startup and six successful multimodal requests. Startup
    failure or timeout fails the case; emergency cleanup cannot turn it into a
    passing restart-recovery scenario. No replacement replica is started.
    """

    server_set = "replica"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        starting = start_server_from_params(params)
        finished = threading.Event()
        ready = threading.Event()
        errors: list[BaseException] = []

        def enter() -> None:
            try:
                starting.__enter__()
                ready.set()
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=enter, daemon=True)
        thread.start()
        injected = False
        try:
            deadline = time.monotonic() + params.startup_timeout
            while time.monotonic() < deadline and not finished.is_set():
                if _kill_starting_stage(starting, model.bad_replica_pattern, finished):
                    injected = True
                    break
                time.sleep(0.05)
            assert injected, f"no startup replica fault injected: ready={ready.is_set()}, errors={errors!r}"
            thread.join(params.startup_timeout)
            assert not thread.is_alive(), "replica startup did not finish within deadline after injection"
            assert ready.is_set() and not errors, (
                f"original instance failed degraded startup after replica crash: {errors!r}"
            )
            print("[replica-crash-at-launch] original degraded instance became ready", flush=True)
            _assert_survivor_serves(model, client_for_server(starting), starting)
        finally:
            if thread.is_alive() and starting.proc is not None:
                make_server_tree_kill_fault_injector(signal_name="SIGKILL")(starting)
                thread.join(LAUNCH_FAILURE_TIMEOUT_SEC)
            if ready.is_set():
                starting.__exit__(None, None, None)
            assert not thread.is_alive(), "startup thread survived emergency cleanup"
            # Do not let a no-PIDs cleanup skip mask a missing injection.
            if starting.proc is not None:
                assert_no_server_tree_process_residual_and_gpu_release(
                    starting, scenario="replica_crash_at_launch", timeout_sec=60
                )

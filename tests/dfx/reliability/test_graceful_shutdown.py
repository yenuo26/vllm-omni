# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks separating abrupt crash injection from normal shutdown."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).parent


@pytest.fixture
def shutdown():
    path = ROOT / "scenarios/process_kill.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {"GracefulShutdown", "ReplicaGracefulShutdown", "_require_signal"}
    ]
    module = ModuleType("shutdown_under_test")
    module.__dict__.update(Any=Any, FaultModel=Any, FaultScenario=object, SIGTERM_CLEANUP_GRACE_SEC=60)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("outcome", ["clean", "leak", "slow_signal", "late_cleanup"])
def test_normal_shutdown_has_one_sigterm_and_a_shared_cleanup_deadline(shutdown, multi, outcome):
    events: list[str] = []
    clock = [0.0]
    shutdown.time = SimpleNamespace(monotonic=lambda: clock[0])
    model = SimpleNamespace(warmup=Mock(side_effect=lambda *args: events.append("preflight")))
    server, client = object(), object()

    def terminate(actual):
        assert actual is server and events == ["preflight"]
        events.append("sigterm")
        clock[0] = 61.0 if outcome == "slow_signal" else 2.0

    shutdown.make_server_root_kill_fault_injector = Mock(return_value=terminate)

    def cleanup(actual, *, scenario, timeout_sec):
        assert actual is server and scenario == "graceful_shutdown" and timeout_sec == 58.0
        events.append("cleanup")
        if outcome == "leak":
            raise AssertionError("resource residual")
        clock[0] = 61.0 if outcome == "late_cleanup" else 5.0

    shutdown.assert_no_server_tree_process_residual_and_gpu_release = Mock(side_effect=cleanup)
    shutdown.assert_post_fault_api_unreachable = Mock(side_effect=lambda *a, **k: events.append("api_down"))
    scenario = shutdown.ReplicaGracefulShutdown() if multi else shutdown.GracefulShutdown()
    assert scenario.server_set == ("replica" if multi else "lifecycle")
    if outcome == "clean":
        scenario.run(model, server, client)
        assert events == ["preflight", "sigterm", "cleanup", "api_down"]
    else:
        with pytest.raises(AssertionError, match="resource residual|grace period"):
            scenario.run(model, server, client)
        shutdown.assert_post_fault_api_unreachable.assert_not_called()
    shutdown.make_server_root_kill_fault_injector.assert_called_once_with(
        signal_name="SIGTERM", post_kill_wait_seconds=0.0
    )


@pytest.mark.parametrize("signal", [None, "SIGTERM"])
def test_crash_rejects_non_sigkill(shutdown, signal):
    with pytest.raises((ValueError, RuntimeError)):
        shutdown._require_signal(signal)
    assert shutdown._require_signal("SIGKILL") == "SIGKILL"


def test_all_models_separate_graceful_shutdown_from_crash_signals():
    tree = ast.parse((ROOT / "helpers.py").read_text(encoding="utf-8"))
    signals = next(
        n
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "CRASH_SIGNALS"
    )
    assert signals.value is not None
    assert ast.literal_eval(signals.value) == (("SIGKILL", None),)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FaultModel")
    for statement in model.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id.endswith("_signals")
        ):
            assert isinstance(statement.value, ast.Name) and statement.value.id == "CRASH_SIGNALS"
    for filename in (
        "test_qwen3_omni.py",
        "test_minicpmo_4_5.py",
        "test_wan22.py",
        "test_hunyuan_image.py",
        "test_qwen3_omni_replica.py",
        "test_minicpmo_4_5_replica.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        assert any(n.name == "test_graceful_shutdown" for n in functions), filename
    qwen = ast.parse((ROOT / "test_qwen3_omni.py").read_text(encoding="utf-8"))
    root_signals = next(
        n
        for n in qwen.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "QWEN_SERVE_ROOT_SIGNALS"
    )
    assert root_signals.value is not None
    assert ast.literal_eval(root_signals.value) == (("SIGKILL", "vllm issue#43060"),)

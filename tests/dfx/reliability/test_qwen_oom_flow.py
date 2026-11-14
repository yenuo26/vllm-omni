# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the single-injection Qwen OOM control flow."""

from __future__ import annotations

import ast
import errno
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def qwen_oom():
    # Execute the real free functions without importing GPU/platform modules or
    # evaluating model parametrization during these CPU-only control-flow tests.
    module = ModuleType("qwen_oom_under_test")
    module.__dict__.update(dict(errno=errno, json=json, sys=sys, time=time, Any=Any, pytest=pytest))
    sources = {
        Path(__file__).with_name("test_qwen3_omni.py"): {
            "_assert_multimodal_success",
            "_drain_oom_output",
            "_looks_like_server_unreachable",
            "_run_oom_under_load",
        },
        Path(__file__).with_name("helpers.py"): {
            "extract_openai_error_contract_from_bytes",
            "extract_openai_error_contract_from_payload",
        },
    }
    for path, names in sources.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("recovered", None),
        ("stage_dead", None),
        ("api_dead", None),
        ("driver_oom", None),
        ("accelerator_oom", None),
        ("no_oom_evidence", "without explicit GPU OOM evidence"),
        ("stale_oom_evidence", "without explicit GPU OOM evidence"),
        ("warmup_failed", "expected successful multimodal"),
        ("request_timeout", "unexpected OOM request failure"),
        ("fault_succeeded", "must fail with 500/503"),
        ("invalid_error", "lacks an OpenAI error object"),
        ("recovery_failed", "no audio output"),
    ],
)
def test_qwen_oom_uses_one_injection_and_stops_before_recovery(qwen_oom, capfd, outcome, expected_error):
    events = []
    requests = []
    payload = {"model": "qwen", "modalities": ["text", "audio"], "stream": False}
    success = {"choices": [{"message": {"content": "answer", "audio": {"data": "audio"}}}]}
    qwen_oom.MODEL = SimpleNamespace(
        hard_oom_config={
            "target_mem_ratio": 0.95,
            "hold_seconds": 0,
            "startup_timeout_sec": 120,
            "strict": False,
            "max_free_bytes": 256,
        }
    )
    qwen_oom._mix_probe_payload = lambda _: payload
    qwen_oom._multimodal_request = lambda _: payload
    qwen_oom._deploy_config_path = lambda _: None
    qwen_oom.resolve_oom_device_spec = lambda *args: "0,1"
    server = SimpleNamespace(model="qwen", host="localhost", port=8000, _server_error_lines=[])
    if outcome == "stale_oom_evidence":
        server._server_error_lines.append("torch.OutOfMemoryError: CUDA out of memory from an earlier request")

    def inject(**kwargs):
        events.append("inject")
        return "handle"

    def stop(handle):
        assert handle == "handle"
        events.append("stop")

    def post(host, port, path, body, *, timeout_sec):
        assert body["modalities"] == ["text", "audio"] and body["stream"] is False
        requests.append(body)
        if len(requests) == 1:
            assert events == []
            events.append("warmup")
            return (500 if outcome == "warmup_failed" else 200), json.dumps(success).encode()
        if len(requests) == 2:
            assert events == ["warmup", "inject"]
            events.append("fault")
            if outcome not in {"no_oom_evidence", "stale_oom_evidence"}:
                print(
                    {
                        "driver_oom": "Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain",
                        "accelerator_oom": "torch.AcceleratorError: CUDA error: out of memory",
                    }.get(outcome, "torch.OutOfMemoryError: CUDA out of memory")
                )
            if outcome == "request_timeout":
                raise TimeoutError("request timed out")
            if outcome == "fault_succeeded":
                return 200, json.dumps(success).encode()
            return 500, b"{}" if outcome == "invalid_error" else b'{"error":{"message":"engine dead"}}'
        assert len(requests) == 3 and events[-1] == "health"
        assert "stop" in events
        events.append("recovery")
        if outcome == "recovery_failed":
            return 200, b'{"choices":[{"message":{"content":"answer"}}]}'
        return 200, json.dumps(success).encode()

    def health(*args, **kwargs):
        assert events[-1] == "stop"
        events.append("health")
        if outcome == "api_dead":
            raise ConnectionRefusedError("connection refused")
        return [SimpleNamespace(status_code=200 if outcome in {"recovered", "recovery_failed"} else 503)]

    qwen_oom.inject_gpu_oom = inject
    qwen_oom.stop_gpu_oom_hogs = stop
    qwen_oom.post_json_raw = post
    client = SimpleNamespace(send_health_http_request=health)
    if expected_error is not None:
        with pytest.raises(AssertionError, match=expected_error) as error:
            qwen_oom._run_oom_under_load(server, client, capfd)
        if outcome in {"no_oom_evidence", "stale_oom_evidence"}:
            assert "status=500" in str(error.value)
            assert "engine dead" in str(error.value)
            assert "stderr_tail=" in str(error.value)
    else:
        qwen_oom._run_oom_under_load(server, client, capfd)
    if outcome == "warmup_failed":
        assert events == ["warmup"]
    else:
        assert events.count("inject") == events.count("stop") == 1
    if outcome in {"stage_dead", "api_dead"}:
        assert len(requests) == 2
    if outcome == "recovered":
        assert events == ["warmup", "inject", "fault", "stop", "health", "recovery"]


def test_qwen_oom_capture_starts_before_server_fixture():
    tree = ast.parse(Path(__file__).with_name("test_qwen3_omni.py").read_text(encoding="utf-8"))
    case = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_gpu_oom_under_load"
    )
    assert [arg.arg for arg in case.args.args][:2] == ["capfd", "omni_server_function"]

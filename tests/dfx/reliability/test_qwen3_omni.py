# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni reliability scenarios."""

from __future__ import annotations

import errno
import json
import time
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import resolve_oom_device_spec
from tests.dfx.reliability.helpers import (
    PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC,
    FaultModel,
    SignalSpec,
    extract_openai_error_contract_from_bytes,
    inject_gpu_oom,
    launch_scenario_params,
    post_json_raw,
    scenario_params,
    stop_gpu_oom_hogs,
)
from tests.dfx.reliability.scenarios import (
    FullCrashAtLaunch,
    GpuOomAtLaunch,
    GpuOomUnderLoad,
    GracefulShutdown,
    PartialCrashAtExit,
    PartialCrashAtLaunch,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
)
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

QWEN_SERVE_ROOT_SIGNALS: tuple[SignalSpec, ...] = (("SIGKILL", "vllm issue#43060"),)

RUNTIME_WORKER_PATTERN = "StageEngineCoreProc_stage0"
_PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC = 30
# Includes a fresh interpreter's torch import, CUDA initialization and allocation.
_OOM_SIDECAR_STARTUP_TIMEOUT_SEC = 120


def _default_oom_device_spec() -> str:
    import torch

    count = torch.accelerator.device_count() or 0
    if count <= 0:
        return "0"
    return ",".join(str(index) for index in range(count))


def _hard_oom_config() -> dict[str, Any]:
    import torch

    return {
        "device": _default_oom_device_spec(),
        "target_mem_ratio": 0.95,
        "max_free_bytes": 32 * 1024 * 1024 if getattr(torch.version, "hip", None) else 256 * 1024 * 1024,
        "hold_seconds": 0,
        "startup_timeout_sec": _OOM_SIDECAR_STARTUP_TIMEOUT_SEC,
        "strict": False,
    }


FAULT_ERROR_KEYWORDS = (
    "the request failed",
    "oom",
    "out of memory",
    "cuda",
    "orchestrator",
    "timeout",
    "connection",
    "500",
    "503",
)
PROCESS_KILL_ERROR_KEYWORDS = (
    "timeout",
    "did not complete within",
    "connection",
    "engine",
    "orchestrator",
    "dead",
    "internal",
    "500",
    "503",
)


def _system_prompt() -> dict[str, Any]:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                ),
            }
        ],
    }


def _text_request(model: str) -> dict[str, Any]:
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )
    return {"model": model, "messages": messages, "stream": False, "modalities": ["text"]}


_MULTIMODAL_REQUESTS: dict[str, dict[str, Any]] = {}


def _multimodal_request(model: str) -> dict[str, Any]:
    """Return one shared multimodal body per model.

    Process-kill submits this from a thread pool. Building it again would call
    pyttsx3 from those workers. espeak is process-global, and overlapping
    engines segfault the pytest process.
    """
    cached = _MULTIMODAL_REQUESTS.get(model)
    if cached is None:
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_system_prompt(),
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text="Describe the image and the audio in one short sentence.",
        )
        cached = {"model": model, "messages": messages, "stream": False, "modalities": ["text", "audio"]}
        _MULTIMODAL_REQUESTS[model] = cached
    return {
        "model": cached["model"],
        "messages": cached["messages"],
        "stream": False,
        "modalities": list(cached["modalities"]),
    }


def _heavy_messages() -> list[dict[str, Any]]:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
    return dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text="What is recited in the audio? What is in this image? What is in this video? " * 200,
    )


def _mix_probe_payload(server: Any) -> dict[str, Any]:
    return {
        "model": server.model,
        "messages": _heavy_messages(),
        "stream": False,
        "modalities": ["text", "audio"],
    }


def _assert_text_200(client: Any, model: str) -> None:
    client.send_chat_completions_http_request({"json": _text_request(model), "timeout": 120})


def _deploy_config_path(server: Any) -> str | None:
    args: list[str] = server.serve_args
    for index, arg in enumerate(args):
        if arg == "--deploy-config" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--deploy-config="):
            return arg.split("=", 1)[1]
    return None


def _looks_like_server_unreachable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionRefusedError, BrokenPipeError, ConnectionResetError)):
        return True
    errno_val = getattr(exc, "errno", None)
    if isinstance(exc, OSError) and errno_val is not None:
        return errno_val in (errno.ECONNREFUSED, errno.ECONNRESET, errno.EPIPE)
    message = str(exc).lower()
    return "connection refused" in message or "actively refused" in message


class Qwen3Omni(FaultModel):
    model_id = "qwen3_omni"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 2
    scenarios = [
        {
            "test_name": "qwen3_omni_reliability_async_chunk",
            "server_params": {
                "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "stage_config_name": "qwen3_omni_moe.yaml",
                "server_args": ["--async-chunk"],
            },
        },
    ]
    deploy_dir = Path(get_deploy_config_path("qwen3_omni_moe.yaml")).parent
    lifecycle_scenario = "qwen3_omni_reliability_async_chunk"
    eager_oom_stage_ids = (0, 1)
    worker_pattern = RUNTIME_WORKER_PATTERN
    isolation_pattern = "StageEngineCoreProc_stage1"
    hard_oom_config = _hard_oom_config()
    fault_error_keywords = FAULT_ERROR_KEYWORDS
    process_kill_error_keywords = PROCESS_KILL_ERROR_KEYWORDS
    with_load_timeout_sec = 300.0
    concurrent_workers = 4
    concurrent_timeout_sec = 30.0
    serve_root_signals = QWEN_SERVE_ROOT_SIGNALS
    serve_root_with_load_signals = QWEN_SERVE_ROOT_SIGNALS
    oom_large_fail_message = "expected large chat payload request failure during GPU OOM injection"

    def warmup(self, client: Any, server: Any) -> None:
        client.send_omni_request(_multimodal_request(server.model), request_num=1)

    def submit_worker_load(self, client: Any, server: Any) -> None:
        client.send_omni_request(_multimodal_request(server.model), request_num=1)

    def submit_serve_load(self, client: Any, server: Any) -> None:
        client.send_omni_request(_text_request(server.model), request_num=1)

    def submit_concurrent(self, client: Any, server: Any) -> Any:
        return client.send_chat_completions_http_request(
            {"json": _multimodal_request(server.model), "timeout": 20},
            err_code=(500, 503),
        )

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        client.send_chat_completions_http_request(
            {
                "json": _multimodal_request(server.model),
                "timeout": _PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC,
            },
            err_code=500,
        )

    def isolation_warmup(self, client: Any, server: Any) -> None:
        self.warmup(client, server)

    def isolation_after(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)
        client.send_chat_completions_http_request(
            {"json": _multimodal_request(server.model), "timeout": PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC},
            err_code=500,
        )


def _assert_multimodal_success(status: int, body: bytes) -> None:
    assert status == 200, f"expected successful multimodal request, got {status}: {body[:500]!r}"
    payload = json.loads(body)
    assert isinstance(payload, dict) and not payload.get("error"), f"unexpected response: {payload!r}"
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "multimodal response has no choices"
    message = choices[0].get("message", {})
    assert message.get("content"), "multimodal response has no text output"
    assert message.get("audio", {}).get("data"), "multimodal response has no audio output"


def _run_oom_under_load(server: Any, client: Any) -> None:
    # Generate media before pressure: allocation readiness should be followed
    # immediately by a request, not lengthy synthetic-media preparation.
    payload = _mix_probe_payload(server)
    warmup = _multimodal_request(server.model)
    _assert_multimodal_success(
        *post_json_raw(server.host, server.port, "/v1/chat/completions", warmup, timeout_sec=180)
    )
    print("[oom-under-load] multimodal preflight passed; starting memory pressure", flush=True)
    stderr_before = set(getattr(server, "_server_error_lines", ()))
    config = MODEL.hard_oom_config
    handle = inject_gpu_oom(
        device=resolve_oom_device_spec(config, _deploy_config_path(server)),
        target_mem_ratio=config["target_mem_ratio"],
        hold_seconds=config["hold_seconds"],
        startup_timeout_sec=config["startup_timeout_sec"],
        strict=config["strict"],
        max_free_bytes=config.get("max_free_bytes"),
    )
    fault_body = b""
    status: int | None = None
    try:
        started = time.monotonic()
        print("[oom-under-load] pressure ready; sending one heavy multimodal request (timeout=180s)", flush=True)
        try:
            status, fault_body = post_json_raw(
                server.host, server.port, "/v1/chat/completions", payload, timeout_sec=180
            )
        except Exception as exc:
            print(f"[oom-under-load] request exception after {time.monotonic() - started:.3f}s: {exc!r}", flush=True)
            # A dead API can close/refuse the connection; a client timeout or
            # unrelated exception is not the expected request-failure contract.
            assert _looks_like_server_unreachable(exc), f"unexpected OOM request failure: {exc!r}"
        else:
            # Bound diagnostics; never print the request's base64 media.
            print(
                f"[oom-under-load] request status={status}, elapsed={time.monotonic() - started:.3f}s, "
                f"response_bytes={len(fault_body)}, response_head={fault_body[:4096]!r}",
                flush=True,
            )
            assert status in (500, 503), f"OOM request must fail with 500/503, got {status}: {fault_body[:500]!r}"
            assert extract_openai_error_contract_from_bytes(fault_body) is not None, (
                f"OOM response lacks an OpenAI error object: {fault_body[:500]!r}"
            )
        assert time.monotonic() - started < 180, "OOM request did not fail within its deadline"
        # The API may return generic EngineDeadError; the harness's stderr
        # buffer can identify whether GPU OOM actually caused it.
        stderr_delta = "".join(line for line in getattr(server, "_server_error_lines", ()) if line not in stderr_before)
        evidence = fault_body.decode("utf-8", errors="replace") + stderr_delta
        oom_markers = (
            "cuda out of memory",
            "hip out of memory",
            "torch.outofmemoryerror",
            "torch.cuda.outofmemoryerror",
            "cuda error: out of memory",
            "hip error: out of memory",
            "cuda_error_out_of_memory",
            "hiperroroutofmemory",
            "memory allocation failed with oom on device",
        )
        assert any(marker in evidence.lower() for marker in oom_markers), (
            "request failed without explicit GPU OOM evidence in this injection's response or logs; "
            f"status={status}, response_head={fault_body[:4096]!r}, "
            f"stderr_tail={stderr_delta[-4096:]!r}"
        )
    finally:
        stop_gpu_oom_hogs(handle)
        print("[oom-under-load] memory pressure removed", flush=True)

    try:
        health = client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
    except Exception as exc:
        assert _looks_like_server_unreachable(exc), f"unexpected post-OOM health failure: {exc!r}"
        print("[oom-under-load] API unreachable after OOM; whole-instance restart required")
        return
    if health == 503:
        # Subsequent serial cases start a fresh function-scoped instance and
        # check their workloads. That is suite-level startup coverage, not an
        # in-case recovery assertion or individual-stage replacement check.
        print("[oom-under-load] health=503 after OOM removal; whole-instance restart required")
        return
    assert health == 200, f"unexpected post-OOM health status: {health}"
    _assert_multimodal_success(
        *post_json_raw(server.host, server.port, "/v1/chat/completions", payload, timeout_sec=180)
    )
    print("[oom-under-load] original instance served text and audio after OOM removal")


MODEL = Qwen3Omni()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, GpuOomUnderLoad)
def test_gpu_oom_at_load(omni_server_function, online_client_function) -> None:
    """One async-chunk OOM injection, followed by recovery or a restart-required state.

    A healthy instance must serve the same multimodal request again. A 503 or
    unreachable API requires a full restart. Serial suite cases provide fresh
    startup coverage; this case alone does not assert restart recovery.
    """
    _run_oom_under_load(omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad)
def test_stage_crash_at_idle_001(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """After a quiet worker kill, the next request returns HTTP 500, then concurrent requests fail."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillWorkerWithLoad, skip_reason="issue#8912")
def test_stage_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillWorkerWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootNoLoad)
def test_serve_root_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootWithLoad)
def test_serve_root_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeNoLoad)
def test_process_tree_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeWithLoad)
def test_process_tree_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@launch_scenario_params(MODEL, GpuOomAtLaunch)
def test_gpu_oom_at_launch(launch_params) -> None:
    """Exhaust GPU memory before startup. The failure is observable, and a later start serves."""
    GpuOomAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, PartialCrashAtLaunch)
def test_stage_crash_at_launch(launch_params) -> None:
    """Kill one stage before readiness, require startup failure, cleanup, and a successful restart."""
    PartialCrashAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, FullCrashAtLaunch)
def test_process_tree_crash_at_launch(launch_params) -> None:
    """Kill the startup process tree; later serial cases cover fresh startup and service.

    This case alone does not verify restart recovery.
    """
    FullCrashAtLaunch().run(MODEL, launch_params)


@scenario_params(MODEL, PartialCrashAtExit)
def test_stage_crash_at_exit(omni_server_function, online_client_function) -> None:
    """Begin graceful shutdown, kill one stage, and require timely cleanup before restarting."""
    PartialCrashAtExit().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, PartialStageIsolation)
def test_stage_crash_at_idle_002(omni_server_function, online_client_function) -> None:
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""
    PartialStageIsolation().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, GracefulShutdown)
def test_graceful_shutdown(omni_server_function, online_client_function) -> None:
    """SIGTERM the healthy root; all children and resources must clear within 60s."""
    GracefulShutdown().run(MODEL, omni_server_function, online_client_function)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""HunyuanImage reliability scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.dfx.reliability.helpers import (
    FaultModel,
    client_for_server,
    ensure_shared_oom_server,
    launch_scenario_params,
    scenario_params,
    shared_oom_params,
)
from tests.dfx.reliability.scenarios import (
    FullCrashAtLaunch,
    GpuOomAtLaunch,
    GpuOomLarge,
    GpuOomRecovery,
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
from vllm_omni.platforms import current_omni_platform

_IMAGE_FAST_FAIL_HTTP_TIMEOUT_SEC = 30
_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy"


def _image_request(server: Any) -> dict[str, Any]:
    return {
        "json": {
            "model": server.model,
            "prompt": "A simple red apple on a white background.",
            "size": "512x512",
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": 4,
            "guidance_scale": 7.5,
            "seed": 42,
            "bot_task": "none",
            "use_system_prompt": "en_unified",
        },
        "timeout": 600,
    }


def _fast_fail_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": "post-fault fast-fail check",
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": 4,
        "bot_task": "none",
        "use_system_prompt": "en_unified",
    }


def _reliability_server_params() -> dict[str, Any]:
    """Disable expert parallel only for ROCm fault-injection coverage."""
    params: dict[str, Any] = {
        "model": "tencent/HunyuanImage-3.0-Instruct",
        "stage_config_name": "hunyuan_image3_dit.yaml",
    }
    if current_omni_platform.is_rocm():
        params["update"] = {
            "stages": {
                0: {"parallel_config.enable_expert_parallel": False},
            }
        }
        # A cold or contended MI325 restart can still be loading the 32-shard
        # checkpoint after the 900-second fixture default. Keep the bounded
        # startup deadline, but leave enough headroom for observed progress.
        params["stage_init_timeout"] = 1800
        params["init_timeout"] = 2100
    return params


FAULT_ERROR_KEYWORDS = (
    "oom",
    "out of memory",
    "cuda",
    "connection",
    "job failed",
    "unknown error",
    "internal server error",
    "500 server error",
)


class HunyuanImage(FaultModel):
    model_id = "hunyuan_image"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 4
    extra_marks = (pytest.mark.diffusion,)
    scenarios = [
        {
            "test_name": "hunyuan_image_dit_reliability_default",
            "server_params": _reliability_server_params(),
        }
    ]
    deploy_dir = _DEPLOY_DIR
    worker_pattern = "vLLM-Omni::"
    isolation_pattern = "vLLM-Omni::"
    hard_oom_config = {
        "target_mem_ratio": 0.95,
        "hold_seconds": 0,
        "startup_timeout_sec": 20,
        "strict": True,
    }
    fault_error_keywords = FAULT_ERROR_KEYWORDS
    process_kill_error_keywords = (
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
    oom_large_fail_message = "expected large /v1/images/generations request failure during GPU OOM injection"

    def warmup(self, client: Any, server: Any) -> None:
        client.send_images_generations_http_request(_image_request(server))

    def submit_concurrent(self, client: Any, server: Any) -> Any:
        request = _image_request(server)
        request["timeout"] = 30
        return client.send_images_generations_http_request(request, err_code=(500, 503))

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        client.send_images_generations_http_request(
            {"json": _fast_fail_body(server.model), "timeout": _IMAGE_FAST_FAIL_HTTP_TIMEOUT_SEC},
            err_code=500,
        )

    def oom_large_probe(self, client: Any, server: Any) -> None:
        client.send_images_generations_http_request(_image_request(server))

    def run_oom_recovery(self, server: Any, client: Any) -> None:
        from tests.dfx.reliability.scenarios.oom import _expect_probe_failure

        _expect_probe_failure(self, self.oom_large_probe, self.oom_large_fail_message, client, server)
        client.send_images_generations_http_request(_image_request(server))


MODEL = HunyuanImage()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@shared_oom_params(MODEL)
class TestHunyuanGpuOomShared:
    """OOM cases that release the memory hog and leave the API process up share one server."""

    def test_gpu_oom_large_request_failure(self, omni_server_shared) -> None:
        server = ensure_shared_oom_server(MODEL, omni_server_shared)
        GpuOomLarge().run(MODEL, server, client_for_server(server))

    def test_gpu_oom_recovers_after_fault_removed(self, omni_server_shared) -> None:
        """After OOM pressure stops, a new image request completes."""
        server = ensure_shared_oom_server(MODEL, omni_server_shared)
        GpuOomRecovery().run(MODEL, server, client_for_server(server))


@launch_scenario_params(MODEL, GpuOomAtLaunch)
def test_gpu_oom_at_launch(launch_params) -> None:
    """Exhaust GPU memory before startup. The failure is observable, and a later start serves."""
    GpuOomAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, PartialCrashAtLaunch)
def test_stage_crash_at_launch(launch_params) -> None:
    """Kill one stage before readiness, require startup failure, cleanup, and a successful restart."""
    PartialCrashAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, FullCrashAtLaunch)
def test_full_crash_tree_at_launch(launch_params) -> None:
    """Kill the process tree during startup, then require a clean restart."""
    FullCrashAtLaunch().run(MODEL, launch_params)


@scenario_params(MODEL, PartialCrashAtExit)
def test_stage_crash_at_exit(omni_server_function, online_client_function) -> None:
    """Begin graceful shutdown, kill one stage, and require timely cleanup before restarting."""
    PartialCrashAtExit().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad)
def test_stage_crash_at_idle_001(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """After a quiet worker kill, the next request returns HTTP 500, then concurrent requests fail."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillWorkerWithLoad)
def test_worker_crash_at_load(
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


@scenario_params(MODEL, PartialStageIsolation)
def test_stage_crash_at_idle_002(omni_server_function, online_client_function) -> None:
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""
    PartialStageIsolation().run(MODEL, omni_server_function, online_client_function)

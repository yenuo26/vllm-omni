# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 reliability scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.dfx.reliability.helpers import (
    PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC,
    FaultModel,
    SignalSpec,
    launch_scenario_params,
    scenario_params,
)
from tests.dfx.reliability.scenarios import (
    FullCrashAtExit,
    FullCrashAtLaunch,
    GpuOomAtLaunch,
    GpuOomUnderLoad,
    PartialCrashAtExit,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerWithLoad,
)
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

_MODEL = "openbmb/MiniCPM-o-4_5"
_SERVE_ROOT_SIGTERM: tuple[SignalSpec, ...] = (("SIGTERM", None),)
_NATIVE_CHAT_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja"
)


def _system_prompt() -> dict[str, Any]:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are MiniCPM-o 4.5, a virtual human capable of perceiving "
                    "auditory and visual inputs, as well as generating text and speech."
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


def _audio_request(model: str) -> dict[str, Any]:
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": True, "enable_thinking": False},
    }


def _assert_text_200(client: Any, model: str) -> None:
    client.send_chat_completions_http_request({"json": _text_request(model), "timeout": 120})


class MiniCPMO45(FaultModel):
    model_id = "minicpmo_4_5"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 1
    extra_marks = (pytest.mark.omni,)
    scenarios = [
        {
            "test_name": "minicpmo_4_5_reliability_default",
            "server_params": {
                "model": _MODEL,
                "stage_config_name": "minicpmo_4_5.yaml",
                "server_args": [
                    "--trust-remote-code",
                    "--async-chunk",
                    "--chat-template",
                    str(_NATIVE_CHAT_TEMPLATE),
                    "--chat-template-content-format",
                    "openai",
                ],
            },
        }
    ]
    deploy_dir = Path(get_deploy_config_path("minicpmo_4_5.yaml")).parent
    worker_pattern = "StageEngineCoreProc_stage0"
    isolation_pattern = "StageEngineCoreProc_stage1"
    with_load_timeout_sec = 300.0
    serve_root_signals = _SERVE_ROOT_SIGTERM
    serve_root_with_load_signals = _SERVE_ROOT_SIGTERM
    hard_oom_config = {
        "target_mem_ratio": 0.95,
        "max_free_bytes": 256 * 1024 * 1024,
        "hold_seconds": 0,
        "startup_timeout_sec": 20,
        "strict": False,
    }
    fault_error_keywords = (
        "oom",
        "out of memory",
        "cuda",
        "connection",
        "timeout",
        "engine",
        "internal",
        "500",
        "503",
    )
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
    oom_large_fail_message = "expected audio chat failure during GPU OOM injection"

    def warmup(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)

    def submit_worker_load(self, client: Any, server: Any) -> None:
        client.send_chat_completions_http_request({"json": _audio_request(server.model), "timeout": 120})

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        client.send_chat_completions_http_request(
            {"json": _audio_request(server.model), "timeout": PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC},
            err_code=500,
        )

    def oom_large_probe(self, client: Any, server: Any) -> None:
        client.send_chat_completions_http_request({"json": _audio_request(server.model), "timeout": 60})

    def isolation_warmup(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)

    def isolation_after(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)
        client.send_chat_completions_http_request(
            {"json": _audio_request(server.model), "timeout": PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC},
            err_code=500,
        )


MODEL = MiniCPMO45()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@launch_scenario_params(MODEL, GpuOomAtLaunch)
def test_gpu_oom_at_launch(launch_params) -> None:
    """Exhaust GPU memory before startup. The failure is observable, and a later start serves."""
    GpuOomAtLaunch().run(MODEL, launch_params)


@scenario_params(MODEL, GpuOomUnderLoad)
def test_gpu_oom_under_load(omni_server_function, online_client_function) -> None:
    """An audio request fails while GPU memory is exhausted, then a later text request succeeds."""
    GpuOomUnderLoad().run(MODEL, omni_server_function, online_client_function)


@launch_scenario_params(MODEL, FullCrashAtLaunch)
def test_full_crash_tree_at_launch(launch_params) -> None:
    """Kill the process tree during startup, then require a clean restart."""
    FullCrashAtLaunch().run(MODEL, launch_params)


@scenario_params(MODEL, FullCrashAtExit)
def test_full_crash_serve_root_at_exit(omni_server_function, online_client_function) -> None:
    """A stuck shutdown escalates to SIGKILL, and a new server can serve."""
    FullCrashAtExit().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, PartialCrashAtExit)
def test_partial_crash_at_exit(omni_server_function, online_client_function) -> None:
    """Kill one stage, then require shutdown to finish or escalate and a restart to serve."""
    PartialCrashAtExit().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerWithLoad)
def test_process_kill_worker_with_load_request_failure(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillWorkerWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeNoLoad)
def test_full_crash_tree_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeWithLoad)
def test_full_crash_tree_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootNoLoad)
def test_full_crash_serve_root_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootWithLoad)
def test_full_crash_serve_root_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, PartialStageIsolation)
def test_partial_stage_isolation(omni_server_function, online_client_function) -> None:
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""
    PartialStageIsolation().run(MODEL, omni_server_function, online_client_function)

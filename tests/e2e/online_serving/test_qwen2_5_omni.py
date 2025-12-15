# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os

from pathlib import Path

import openai
import pytest
import time
from vllm.assets.video import VideoAsset
from tests.conftest import OmniServer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen2.5-Omni-7B"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml")]
# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.fixture
def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.mark.parametrize("test_param", test_params)
def test_video_to_audio(
    test_param,
) -> None:
    """Test processing video, generating audio output via OpenAI API."""
    # Create data URL for the base64 encoded video
    model, stage_config_path = test_param
    with OmniServer(model, []) as server:
        time.sleep(1000000)
        pass

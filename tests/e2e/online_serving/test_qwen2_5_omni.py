# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os

from pathlib import Path

import openai
import pytest
from vllm.assets.video import VideoAsset
from tests.conftest import OmniServer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen2.5-Omni-3B"]

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


@pytest.fixture(scope="session")
def base64_encoded_video() -> str:
    """Base64 encoded video for testing."""
    import base64

    video = VideoAsset(name="baby_reading", num_frames=4)
    with open(video.video_path, "rb") as f:
        content = f.read()
        return base64.b64encode(content).decode("utf-8")


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def dummy_messages_from_video_data(
    video_data_url: str,
    content_text: str = "Describe the video briefly.",
):
    """Create messages with video data URL for OpenAI API."""
    return [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_data_url}},
                {"type": "text", "text": content_text},
            ],
        },
    ]


@pytest.mark.parametrize("test_param", test_params)
def test_video_to_audio(
    test_param,
) -> None:
    """Test processing video, generating audio output via OpenAI API."""
    # Create data URL for the base64 encoded video
    model, stage_config_path = test_param
    with OmniServer(model, ["--stage-configs-path", stage_config_path]) as server:
        pass

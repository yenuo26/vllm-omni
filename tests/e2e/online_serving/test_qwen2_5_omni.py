# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os
from pathlib import Path

import openai
import pytest

from tests.conftest import OmniServer, dummy_messages_from_mix_data, prepare_multimodal_base64_data, modify_stage_config
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen2.5-Omni-7B"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen2_5_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

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


def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.mark.parametrize("test_config", test_params)
def test_mixed_modalities_to_text_audio(
    test_config: tuple[str, str]
) -> None:
    model, stage_config_path = test_config
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--init-sleep-seconds", "90"]) as server:
        """Test processing video, generating audio output via OpenAI API."""
        # Create data URL for the base64 encoded video
        video_data_url = f"data:video/mp4;base64,{prepare_multimodal_base64_data('baby_reading', 'video')}"

        # Create data URL for the base64 encoded audio
        audio_data_url = f"data:audio/ogg;base64,{prepare_multimodal_base64_data('mary_had_lamb', 'audio')}"

        # Create data URL for the base64 encoded image
        image_data_url = f"data:image/jpeg;base64,{prepare_multimodal_base64_data('cherry_blossom', 'image')}"

        messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), video_data_url=video_data_url,
                                                audio_data_url=audio_data_url,
                                                image_data_url=image_data_url)

        # Test single completion
        api_client = client(server)
        chat_completion = api_client.chat.completions.create(
            model=server.model,
            messages=messages,
        )

        # Verify text output
        text_choice = chat_completion.choices[0]
        assert text_choice.finish_reason == "length"

        # Verify we got a response
        text_message = text_choice.message
        assert text_message.content is not None and len(text_message.content) >= 10
        assert text_message.role == "assistant"

        # Verify audio output
        audio_choice = chat_completion.choices[1]
        assert audio_choice.finish_reason == "stop"
        audio_message = audio_choice.message

        # Check if audio was generated
        if hasattr(audio_message, "audio") and audio_message.audio:
            assert audio_message.audio.data is not None
            assert len(audio_message.audio.data) > 0


@pytest.mark.parametrize("test_config", test_params)
def test_video_to_text_audio(
        test_config: tuple[str, str]
) -> None:
    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(stage_config_path, 0, {"runtime.max_batch_size": 5})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--init-sleep-seconds", "90"]) as server:
        """Test processing video, generating audio output via OpenAI API."""
        # Create data URL for the base64 encoded video
        video_data_url = f"data:video/mp4;base64,{prepare_multimodal_base64_data('baby_reading', 'video')}"

        # Create data URL for the base64 encoded audio
        audio_data_url = f"data:audio/ogg;base64,{prepare_multimodal_base64_data('mary_had_lamb', 'audio')}"

        # Create data URL for the base64 encoded image
        image_data_url = f"data:image/jpeg;base64,{prepare_multimodal_base64_data('cherry_blossom', 'image')}"

        messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), video_data_url=video_data_url,
                                                audio_data_url=audio_data_url,
                                                image_data_url=image_data_url)

        # Test single completion
        api_client = client(server)
        chat_completion = api_client.chat.completions.create(
            model=server.model,
            messages=messages,
        )

        # Verify text output
        text_choice = chat_completion.choices[0]
        assert text_choice.finish_reason == "length"

        # Verify we got a response
        text_message = text_choice.message
        assert text_message.content is not None and len(text_message.content) >= 10
        assert text_message.role == "assistant"

        # Verify audio output
        audio_choice = chat_completion.choices[1]
        assert audio_choice.finish_reason == "stop"
        audio_message = audio_choice.message

        # Check if audio was generated
        if hasattr(audio_message, "audio") and audio_message.audio:
            assert audio_message.audio.data is not None
            assert len(audio_message.audio.data) > 0

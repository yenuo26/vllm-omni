# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import concurrent.futures
import os
import time
from pathlib import Path

import openai
import pytest

from tests.conftest import (
    OmniServer,
    convert_audio_to_text,
    cosine_similarity_text,
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
    run_benchmark,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


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


@pytest.mark.parametrize("test_config", test_params)
def test_text_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(), content_text="What is the capital of China?"
        )

        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=20, modalities=["text"]
        )
        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.

        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens <= 20, "The output length more than the requested max_tokens."
        assert "beijing" in text_choice.message.content.lower(), "The output do not contain keywords."


@pytest.mark.parametrize("test_config", test_params)
def test_text_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text and audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(), content_text="What is the capital of China?"
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(api_client.chat.completions.create, model=server.model, messages=messages)
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert "beijing" in text_choice.message.content.lower(), "The output do not contain keywords."

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content.lower(), text_content.lower()) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_to_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {
                "engine_args.gpu_memory_utilization": 0.95,
                "engine_args.tensor_parallel_size": 1,
                "runtime.devices": "0",
            },
            2: {"runtime.devices": "1"},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=10, stop=None, modalities=["audio"]
        )

        # Verify only output audio
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify audio output success
        audio_message = chat_completion.choices[1].message
        assert audio_message.audio.data is not None, "No audio output is generated"
        assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_audio_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {
                "engine_args.gpu_memory_utilization": 0.95,
                "engine_args.tensor_parallel_size": 1,
                "runtime.devices": "2",
                "default_sampling_params.ignore_eos": True,
            },
            2: {"runtime.devices": "1"},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(1, 1)}"
        messages = dummy_messages_from_mix_data(audio_data_url=audio_data_url)
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=1000, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 1000, (
            "The output length differs from the requested max_tokens."
        )

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_audio_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests, "default_sampling_params.ignore_eos": True},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        audio_data_url = []
        for _ in range(5):
            audio_data_url.append(f"data:audio/wav;base64,{generate_synthetic_audio(1, 5)}")

        messages = dummy_messages_from_mix_data(audio_data_url=audio_data_url)

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_image_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {
                "engine_args.gpu_memory_utilization": 0.95,
                "engine_args.tensor_parallel_size": 1,
                "runtime.devices": "2",
                "default_sampling_params.ignore_eos": True,
            },
            2: {"runtime.devices": "1"},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}"
        messages = dummy_messages_from_mix_data(image_data_url=image_data_url)
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=1000, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 1000, (
            "The output length differs from the requested max_tokens."
        )

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_image_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests, "default_sampling_params.ignore_eos": True},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        image_data_url = []
        for _ in range(4):
            image_data_url.append(f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)}")

        messages = dummy_messages_from_mix_data(image_data_url=image_data_url)

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_audio_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(1, 1)}"
        messages = dummy_messages_from_mix_data(
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
            audio_data_url=audio_data_url,
        )
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=10, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_audio_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {
                "engine_args.gpu_memory_utilization": 0.95,
                "engine_args.tensor_parallel_size": 1,
                "runtime.devices": "2",
                "runtime.max_batch_size": num_concurrent_requests,
            },
            1: {"runtime.max_batch_size": num_concurrent_requests, "runtime.devices": "1"},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        audio_data_url = []
        for _ in range(4):
            audio_data_url.append(f"data:audio/wav;base64,{generate_synthetic_audio(10, 5)}")

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            audio_data_url=audio_data_url,
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_image_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {
                "engine_args.gpu_memory_utilization": 0.95,
                "engine_args.tensor_parallel_size": 1,
                "runtime.devices": "0",
                "default_sampling_params.ignore_eos": True,
            },
            2: {"runtime.devices": "1"},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}"
        messages = dummy_messages_from_mix_data(
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
            image_data_url=image_data_url,
        )
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=1000, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 1000, (
            "The output length differs from the requested max_tokens."
        )

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_image_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        image_data_url = []
        for _ in range(4):
            image_data_url.append(f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)}")

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            image_data_url=image_data_url,
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_video_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {
                "engine_args.gpu_memory_utilization": 0.95,
                "engine_args.tensor_parallel_size": 1,
                "runtime.devices": "0",
            },
            2: {"runtime.devices": "1"},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(16, 16, 24)}"
        messages = dummy_messages_from_mix_data(
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
            video_data_url=video_data_url,
        )
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=10, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_video_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = []
        for _ in range(4):
            video_data_url.append(f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 300)}")

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_mix_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 24)}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(1, 1)}"

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                    modalities=["text"],
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify only output text
            assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify E2E
            print(f"the request e2e is: {time.perf_counter() - start_time}")
            # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_mix_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = list()
        image_data_url = list()
        audio_data_url = list()

        for _ in range(2):
            video_data_url.append(f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 300)}")
            image_data_url.append(f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}")
            audio_data_url.append(f"data:audio/wav;base64,{generate_synthetic_audio(10, 2)}")

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_mix_to_text_audio_002(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 256
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        request_rates = [0.5, 0.8, 1]
        for request_rate in request_rates:
            args = [
                "--model",
                server.model,
                "--host",
                server.host,
                "--port",
                str(server.port),
                "--dataset-name",
                "random-mm",
                "--request_rate",
                str(request_rate),
                "--random-input-len",
                "100",
                "--random-range-ratio",
                "0.0",
                "--random-mm-base-items-per-request",
                "3",
                "--random-mm-num-mm-items-range-ratio",
                "0",
                "--random-mm-limit-mm-per-prompt",
                '{"image":1, "video": 1, "audio": 1}',
                "--random-mm-bucket-config",
                '{"(16,16,1)":0.33, "(0,1,1)": 0.33, "(16, 16, 24)": 0.33}',
                "--ignore-eos",
                "--random-output-len",
                "10",
                "--num-prompts",
                "100",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--endpoint",
                "/v1/chat/completions",
                "--backend",
                "openai-chat",
            ]
            result = run_benchmark(args)
            assert result.get("completed") == 100, "The request success rate did not reach 100%."


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_mix_to_text_audio_003(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 256
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        request_rates = [0.5, 0.8, 1]
        for request_rate in request_rates:
            args = [
                "--model",
                server.model,
                "--host",
                server.host,
                "--port",
                str(server.port),
                "--dataset-name",
                "random-mm",
                "--request_rate",
                str(request_rate),
                "--random-input-len",
                "1000",
                "--random-range-ratio",
                "0.0",
                "--random-mm-base-items-per-request",
                "6",
                "--random-mm-num-mm-items-range-ratio",
                "0",
                "--random-mm-limit-mm-per-prompt",
                '{"image":2, "video": 2, "audio": 2}',
                "--random-mm-bucket-config",
                '{"(1280,720,1)":0.33, "(0,10,2)": 0.33, "(1280, 720, 300)": 0.33}',
                "--ignore-eos",
                "--random-output-len",
                "1000",
                "--num-prompts",
                "100",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--endpoint",
                "/v1/chat/completions",
                "--backend",
                "openai-chat",
            ]
            result = run_benchmark(args)
            assert result.get("completed") == 100, "The request success rate did not reach 100%."


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_mix_to_text_audio_004(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 256
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        request_rates = [0.5, 0.8, 1]
        for request_rate in request_rates:
            args = [
                "--model",
                server.model,
                "--host",
                server.host,
                "--port",
                str(server.port),
                "--dataset-name",
                "random-mm",
                "--request_rate",
                str(request_rate),
                "--random-input-len",
                "1000",
                "--random-range-ratio",
                "0.0",
                "--random-mm-base-items-per-request",
                "10",
                "--random-mm-num-mm-items-range-ratio",
                "0",
                "--random-mm-limit-mm-per-prompt",
                '{"image":4, "video": 4, "audio": 2}',
                "--random-mm-bucket-config",
                '{"(1280,720,1)":0.33, "(0,10,5)": 0.33, "(1280, 720, 300)": 0.33}',
                "--ignore-eos",
                "--random-output-len",
                "1000",
                "--num-prompts",
                "100",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--endpoint",
                "/v1/chat/completions",
                "--backend",
                "openai-chat",
            ]
            result = run_benchmark(args)
            assert result.get("completed") == 1000, "The request success rate did not reach 100%."


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_chunked_prefill_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(stage_config_path, {0: {"engine_args.max_num_batched_tokens": 32}})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(16, 16, 300)}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(1, 5)}"

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text="What is recited in the audio? What is in this image? Please describe the video briefly.",
        )
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model,
            messages=messages,
            modalities=["text"],
            max_token=10,
        )
        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_chunked_prefill_002(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests, "engine_args.max_num_batched_tokens": 32},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(16, 16, 300)}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(1, 5)}"

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text="What is recited in the audio? What is in this image? Please describe the video briefly.",
        )
        # Test single completion
        api_client = client(server)
        e2e_list = list()
        sampling_params_list = [{"max_tokens": 10}, {"max_tokens": 20}, {"max_tokens": 20}]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    lambda: api_client.chat.completions.create(
                        model=server.model, messages=messages, extra_body={"sampling_params_list": sampling_params_list}
                    )
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_chunked_prefill_003(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 100
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests, "engine_args.max_num_batched_tokens": 128},
            1: {"runtime.max_batch_size": num_concurrent_requests, "engine_args.max_num_batched_tokens": 128},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        request_rates = [0.5, 0.8, 1]
        for request_rate in request_rates:
            args = [
                "--model",
                server.model,
                "--host",
                server.host,
                "--port",
                str(server.port),
                "--dataset-name",
                "random-mm",
                "--request_rate",
                str(request_rate),
                "--random-input-len",
                "200",
                "--random-range-ratio",
                "0.0",
                "--random-mm-base-items-per-request",
                "3",
                "--random-mm-num-mm-items-range-ratio",
                "0",
                "--random-mm-limit-mm-per-prompt",
                '{"image":1, "video": 1, "audio": 1}',
                "--random-mm-bucket-config",
                '{"(16,16,1)":0.33, "(0,1,1)": 0.33, "(16, 16, 24)": 0.33}',
                "--ignore-eos",
                "--random-output-len",
                "10",
                "--num-prompts",
                "100",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--endpoint",
                "/v1/chat/completions",
                "--backend",
                "openai-chat",
            ]
            result = run_benchmark(args)
            assert result.get("completed") == 100, "The request success rate did not reach 100%."


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_use_audio_in_video_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = []
        for _ in range(4):
            video_data_url.append(f"data:video/mp4;base64,{generate_synthetic_video(16, 16, 300)}")

        messages = dummy_messages_from_mix_data(
            video_data_url=video_data_url
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                    extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}},
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )

@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_use_audio_in_video_002(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = []
        for _ in range(4):
            video_data_url.append(f"data:video/mp4;base64,{generate_synthetic_video(16, 16, 300)}")

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                    extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}},
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )

@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_use_audio_in_video_003(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        video_data_url = list()
        image_data_url = list()
        audio_data_url = list()

        for _ in range(4):
            video_data_url.append(f"data:video/mp4;base64,{generate_synthetic_video(16, 16, 300)}")
            image_data_url.append(f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}")
            audio_data_url.append(f"data:audio/wav;base64,{generate_synthetic_audio(10, 2)}")

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                    extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}},
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, (
                "The output length differs from the requested max_tokens."
            )

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            assert cosine_similarity_text(audio_content, text_content) > 0.9, (
                "The audio content is not same as the text"
            )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena
"""
import base64
import io
import logging
import os
import tempfile
from collections.abc import Iterator, Mapping
from typing import Any, cast, Dict

import cv2
import numpy as np
import torch
import torchaudio
from vllm.benchmarks.datasets import RandomMultiModalDataset

logger = logging.getLogger(__name__)


def process_video(video: Any) -> Mapping[str, Any]:
    """
    Process a single video input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw video bytes: - Expects a dict with a 'bytes' key
       containing raw video data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(video, dict) and 'bytes' in video:
        video_bytes = video['bytes']
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "type": "video_url",
            "video_url": {
                "url": f"data:video/mp4;base64,{video_base64}"
            },
        }

    if isinstance(video, str):
        video_url = (video if video.startswith(
            ("http://", "https://", "file://")) else f"file://{video}")
        return {"type": "video_url", "video_url": {"url": video_url}}

    raise ValueError(
        f"Invalid video input {video}. Must be a string of local path/remote url, or a dictionary with raw video bytes in the form of `{{'bytes': raw_video_bytes}}`."  # noqa: E501
    )


def process_audio(audio: Any) -> Mapping[str, Any]:
    """
    Process a single audio input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw audio bytes: - Expects a dict with a 'bytes' key
       containing raw audio data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the audio URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(audio, dict) and 'bytes' in audio:
        audio_bytes = audio['bytes']
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/mpeg;base64,{audio_base64}"
            },
        }
    if isinstance(audio, str):
        audio_url = (audio if audio.startswith(
            ("http://", "https://", "file://")) else f"file://{audio}")
        return {"type": "audio_url", "audio_url": {"url": audio_url}}

    raise ValueError(f"Invalid audio input {audio}. Must be a string of local path/remote url, or a dictionary with raw audio bytes in the form of `{{'bytes': raw_audio_bytes}}`."
   )



# -----------------------------------------------------------------------------
# MultiModalDataset Implementation
# -----------------------------------------------------------------------------
class OmniRandomMultiModalDataset(RandomMultiModalDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def generate_synthetic_audio(
        self,
        duration: int,  # seconds
        num_channels: int #1：Mono，2：Stereo 5：5.1 surround sound
    ) -> Dict[str, Any]:
        """Generate synthetic audio with random values.
           Default use 48000Hz.
        """
        sample_rate = 48000
        num_samples = int(sample_rate * duration)
        audio_data = self._rng.uniform(
            -0.5, 0.5,
            (num_samples, num_channels)
        )
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_tensor = torch.FloatTensor(audio_data.T)
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate,
            format="mp3"
        )
        buffer.seek(0)
        audio_bytes = buffer.read()
        return {
            'bytes': audio_bytes,
        }


    def generate_mm_item(self,
                         mm_item_config: tuple[int, int, int],
                         ) -> Mapping[str, Any]:
        """
        Create synthetic images and videos and
        apply process_image/process_video respectively.
        This follows the OpenAI API chat completions
        https://github.com/openai/openai-python
        """

        if self.map_config_to_modality(mm_item_config) == "image":
            return process_image(self.generate_synthetic_image(
                                                            mm_item_config[1],
                                                            mm_item_config[0]))
        elif self.map_config_to_modality(mm_item_config) == "video":
            return process_video(self.generate_synthetic_video(
                                                            mm_item_config[1],
                                                            mm_item_config[0],
                                                            mm_item_config[2]))
        elif self.map_config_to_modality(mm_item_config) == "audio":
            return process_audio(self.generate_synthetic_audio(
                                                            mm_item_config[1],
                                                            mm_item_config[2]))
        else:
            raise ValueError(f"Invalid multimodal item configuration: "
                             f"{mm_item_config}")


    def generate_synthetic_video(self, width: int,
                                    height: int,
                                    num_frames: int) -> Any:
        """Generate synthetic video with random values.
        """
        video_data = self._rng.integers(
            0, 256,
            (num_frames, height, width, 3),
            dtype=np.uint8,
        )
        video_tensor = torch.from_numpy(video_data)
        with tempfile.NamedTemporaryFile(suffix=f".mp4", delete=False) as tmp:
            temp_path = tmp.name
        frames, height, width, channels = video_tensor.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 30, (width, height))

        for i in range(frames):
            frame = video_tensor[i].numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()

        with open(temp_path, 'rb') as f:
            video_bytes = f.read()

        os.unlink(temp_path)

        return {
            'bytes': video_bytes,
        }

    def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
        """Map the configuration to the modality."""
        if config[0] == 0:
            return "audio"
        elif config[-1] == 1:
            return "image"
        elif config[-1] > 1:
            return "video"
        else:
            raise ValueError(f"Invalid multimodal item configuration: {config}")

import base64
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import pytest
import soundfile as sf
import torch
import whisper
import yaml
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils import get_open_port

from vllm_omni.entrypoints.omni import Omni

logger = init_logger(__name__)


@pytest.fixture(autouse=True)
def clean_gpu_memory_between_tests():
    if os.getenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0") != "1":
        yield
        return

    # Wait for GPU memory to be cleared before starting the test
    import gc

    from tests.utils import wait_for_gpu_memory_to_clear

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        try:
            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)),
                threshold_ratio=0.1,
            )
        except ValueError as e:
            logger.info("Failed to clean GPU memory: %s", e)

    yield

    # Clean up GPU memory after the test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def dummy_messages_from_mix_data(
    system_prompt: dict[str, Any] = None,
    video_data_url: Any = None,
    audio_data_url: Any = None,
    image_data_url: Any = None,
    content_text: str = None,
):
    """Create messages with video、image、audio data URL for OpenAI API."""

    if content_text is not None:
        content = [{"type": "text", "text": content_text}]
    else:
        content = []

    media_items = []
    if isinstance(video_data_url, list):
        for video_url in video_data_url:
            media_items.append((video_url, "video"))
    else:
        media_items.append((video_data_url, "video"))

    if isinstance(image_data_url, list):
        for url in image_data_url:
            media_items.append((url, "image"))
    else:
        media_items.append((image_data_url, "image"))

    if isinstance(audio_data_url, list):
        for url in audio_data_url:
            media_items.append((url, "audio"))
    else:
        media_items.append((audio_data_url, "audio"))

    content.extend(
        {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}
        for url, media_type in media_items
        if url is not None
    )
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages = [system_prompt] + messages
    return messages


def generate_synthetic_audio(
    duration: int,  # seconds
    num_channels: int,  # 1：Mono，2：Stereo 5：5.1 surround sound
) -> dict[str, Any]:
    """Generate synthetic audio with random values.
    Default use 48000Hz.
    """
    sample_rate = 48000
    num_samples = int(sample_rate * duration)
    audio_data = np.random.uniform(-0.5, 0.5, (num_samples, num_channels))
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_tensor = torch.FloatTensor(audio_data.T)
    audio_np = audio_tensor.numpy()

    buffer = io.BytesIO()

    sf.write(buffer, audio_np.T, sample_rate, format="wav")

    buffer.seek(0)
    audio_bytes = buffer.read()
    buffer.close()
    return base64.b64encode(audio_bytes).decode("utf-8")


def generate_synthetic_video(width: int, height: int, num_frames: int) -> Any:
    """Generate synthetic video with random values."""
    video_data = np.random.randint(
        0,
        256,
        (num_frames, height, width, 3),
        dtype=np.uint8,
    )
    video_tensor = torch.from_numpy(video_data)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name
    frames, height, width, channels = video_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, 30, (width, height))

    for i in range(frames):
        frame = video_tensor[i].numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    with open(temp_path, "rb") as f:
        content = f.read()
    os.unlink(temp_path)

    return base64.b64encode(content).decode("utf-8")


def generate_synthetic_image(width: int, height: int) -> Any:
    """Generate synthetic image with random values."""
    from PIL import Image

    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85, optimize=True)
    buffer.seek(0)
    image_bytes = buffer.read()

    return base64.b64encode(image_bytes).decode("utf-8")


def cosine_similarity_text(s1, s2):
    """
        Calculate cosine similarity between two text strings.
        Notes:
    ------
    - Higher score means more similar texts
    - Score of 1.0 means identical word composition (bag-of-words)
    - Score of 0.0 means completely different vocabulary
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = CountVectorizer().fit_transform([s1, s2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def convert_audio_to_text(audio_data):
    """
    Convert base64 encoded audio data to text using speech recognition.
    """

    audio_data = base64.b64decode(audio_data)
    output_path = f"./test_{int(time.time())}"
    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_data)

    print(f"audio data is saved: {output_path}")
    model = whisper.load_model("base")
    text = model.transcribe(output_path)["text"]
    if text:
        return text
    else:
        return ""


def modify_stage_config(
    yaml_path: str,
    stage_updates: dict[int, dict[str, Any]],
) -> str:
    """
    Batch modify configurations for multiple stages in a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.
        stage_updates: Dictionary where keys are stage IDs and values are dictionaries of
                      modifications for that stage. Each modification dictionary uses
                      dot-separated paths as keys and new configuration values as values.
                      Example: {
                          0: {'engine_args.max_model_len': 5800},
                          1: {'runtime.max_batch_size': 2}
                      }

    Returns:
        str: Path to the newly created modified YAML file with timestamp suffix.

    Example:
        >>> output_file = modify_stage_config(
        ...     'config.yaml',
        ...     {
        ...         0: {'engine_args.max_model_len': 5800},
        ...         1: {'runtime.max_batch_size': 2}
        ...     }
        ... )
        >>> print(f"Modified configuration saved to: {output_file}")
        Modified configuration saved to: config_1698765432.yaml
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"yaml does not exist: {path}")
    try:
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    stage_args = config.get("stage_args", [])
    if not stage_args:
        raise ValueError("the stage_args does not exist")

    for stage_id, config_dict in stage_updates.items():
        target_stage = None
        for stage in stage_args:
            if stage.get("stage_id") == stage_id:
                target_stage = stage
                break

        if target_stage is None:
            available_ids = [s.get("stage_id") for s in stage_args if "stage_id" in s]
            raise KeyError(f"Stage ID {stage_id} is not exist, available IDs: {available_ids}")

        for key_path, value in config_dict.items():
            current = target_stage
            keys = key_path.split(".")
            for i in range(len(keys) - 1):
                key = keys[i]
                if key not in current:
                    raise KeyError(f"the {'.'.join(keys[: i + 1])} does not exist")

                elif not isinstance(current[key], dict) and i < len(keys) - 2:
                    raise ValueError(f"{'.'.join(keys[: i + 1])}' cannot continue deeper because it's not a dict")
                current = current[key]
            current[keys[-1]] = value

    output_path = f"{yaml_path.split('.')[0]}_{int(time.time())}.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)

    return output_path


def run_benchmark(args: list) -> Any:
    """Generate synthetic image with random values."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"result_{current_dt}.json"
    if "--result-filename" in args:
        print(f"The result file will be overwritten by {result_filename}")
    command = ["vllm-omni", "bench", "serve", "--omni"] + args + ["--save-result", "--result-filename", result_filename]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end=" ")

    if "--result-dir" in args:
        index = args.index("--result-dir")
        result_dir = args[index + 1]
    else:
        result_dir = "./"

    with open(os.path.join(result_dir, result_filename), encoding="utf-8") as f:
        result = json.load(f)
    return result


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Set working directory to vllm-omni root
        )

        # Wait for server to be ready
        max_wait = 600  # 10 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, self.port))
                    if result == 0:
                        print(f"Server ready on {self.host}:{self.port}")
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def _kill_process_tree(self, pid):
        """kill process and its children"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            gone, still_alive = psutil.wait_procs(children, timeout=10)

            for child in still_alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            try:
                parent.terminate()
                parent.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            try:
                parent = psutil.Process(self.proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass

                gone, still_alive = psutil.wait_procs(children, timeout=10)

                for child in still_alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

                try:
                    parent.terminate()
                    parent.wait(timeout=10)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass

            except psutil.NoSuchProcess:
                pass


PromptAudioInput = list[tuple[Any, int]] | tuple[Any, int] | None
PromptImageInput = list[Any] | Any | None
PromptVideoInput = list[Any] | Any | None


class OmniRunner:
    """
    Test runner for Omni models.
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        stage_init_timeout: int = 300,
        batch_timeout: int = 10,
        init_timeout: int = 300,
        shm_threshold_bytes: int = 65536,
        log_stats: bool = False,
        stage_configs_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize an OmniRunner for testing.

        Args:
            model_name: The model name or path
            seed: Random seed for reproducibility
            stage_init_timeout: Timeout for initializing a single stage in seconds
            batch_timeout: Timeout for batching in seconds
            init_timeout: Timeout for initializing stages in seconds
            shm_threshold_bytes: Threshold for using shared memory
            log_stats: Enable detailed statistics logging
            stage_configs_path: Optional path to YAML stage config file
            **kwargs: Additional arguments passed to Omni
        """
        self.model_name = model_name
        self.seed = seed

        self.omni = Omni(
            model=model_name,
            log_stats=log_stats,
            stage_init_timeout=stage_init_timeout,
            batch_timeout=batch_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            stage_configs_path=stage_configs_path,
            **kwargs,
        )

    def get_default_sampling_params_list(self) -> list[SamplingParams]:
        """
        Get a list of default sampling parameters for all stages.

        Returns:
            List of SamplingParams with default decoding for each stage
        """
        return [st.default_sampling_params for st in self.omni.stage_list]

    def get_omni_inputs(
        self,
        prompts: list[str] | str,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Construct Omni input format from prompts and multimodal data.

        Args:
            prompts: Text prompt(s) - either a single string or list of strings
            system_prompt: Optional system prompt (defaults to Qwen system prompt)
            audios: Audio input(s) - tuple of (audio_array, sample_rate) or list of tuples
            images: Image input(s) - PIL Image or list of PIL Images
            videos: Video input(s) - numpy array or list of numpy arrays
            mm_processor_kwargs: Optional processor kwargs (e.g., use_audio_in_video)

        Returns:
            List of prompt dictionaries suitable for Omni.generate()
        """
        if system_prompt is None:
            system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                "Group, capable of perceiving auditory and visual inputs, as well as "
                "generating text and speech."
            )

        video_padding_token = "<|VIDEO|>"
        image_padding_token = "<|IMAGE|>"
        audio_padding_token = "<|AUDIO|>"

        if self.model_name == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
            video_padding_token = "<|video_pad|>"
            image_padding_token = "<|image_pad|>"
            audio_padding_token = "<|audio_pad|>"

        if isinstance(prompts, str):
            prompts = [prompts]

        def _normalize_mm_input(mm_input, num_prompts):
            if mm_input is None:
                return [None] * num_prompts
            if isinstance(mm_input, list):
                if len(mm_input) != num_prompts:
                    raise ValueError(
                        f"Multimodal input list length ({len(mm_input)}) must match prompts length ({num_prompts})"
                    )
                return mm_input
            return [mm_input] * num_prompts

        num_prompts = len(prompts)
        audios_list = _normalize_mm_input(audios, num_prompts)
        images_list = _normalize_mm_input(images, num_prompts)
        videos_list = _normalize_mm_input(videos, num_prompts)

        omni_inputs = []
        for i, prompt_text in enumerate(prompts):
            user_content = ""
            multi_modal_data = {}

            audio = audios_list[i]
            if audio is not None:
                if isinstance(audio, list):
                    for _ in audio:
                        user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio
                else:
                    user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio

            image = images_list[i]
            if image is not None:
                if isinstance(image, list):
                    for _ in image:
                        user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image
                else:
                    user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image

            video = videos_list[i]
            if video is not None:
                if isinstance(video, list):
                    for _ in video:
                        user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video
                else:
                    user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video

            user_content += prompt_text

            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            input_dict: dict[str, Any] = {"prompt": full_prompt}
            if multi_modal_data:
                input_dict["multi_modal_data"] = multi_modal_data
            if modalities:
                input_dict["modalities"] = modalities
            if mm_processor_kwargs:
                input_dict["mm_processor_kwargs"] = mm_processor_kwargs

            omni_inputs.append(input_dict)

        return omni_inputs

    def generate(
        self,
        prompts: list[dict[str, Any]],
        sampling_params_list: list[SamplingParams] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Generate outputs for the given prompts.

        Args:
            prompts: List of prompt dictionaries with 'prompt' and optionally
                    'multi_modal_data' keys
            sampling_params_list: List of sampling parameters for each stage.
                                 If None, uses default parameters.

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        if sampling_params_list is None:
            sampling_params_list = self.get_default_sampling_params_list()

        yield from self.omni.generate(prompts, sampling_params_list)

    def generate_multimodal(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Convenience method to generate with multimodal inputs.

        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            audios: Audio input(s)
            images: Image input(s)
            videos: Video input(s)
            mm_processor_kwargs: Optional processor kwargs

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            images=images,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
            modalities=modalities,
        )
        yield from self.generate(omni_inputs, sampling_params_list)

    def generate_audio(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Convenience method to generate with multimodal inputs.
        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            audios: Audio input(s)
            mm_processor_kwargs: Optional processor kwargs
        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        yield from self.generate(omni_inputs, sampling_params_list)

    def generate_video(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Convenience method to generate with multimodal inputs.
        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            videos: Video input(s)
            mm_processor_kwargs: Optional processor kwargs
        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        yield from self.generate(omni_inputs, sampling_params_list)

    def generate_image(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        images: PromptImageInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Convenience method to generate with multimodal inputs.
        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            images: Image input(s)
            mm_processor_kwargs: Optional processor kwargs
        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            images=images,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        yield from self.generate(omni_inputs, sampling_params_list)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        del self.omni
        cleanup_dist_env_and_memory()

    def close(self):
        """Close and cleanup the Omni instance."""
        if hasattr(self.omni, "close"):
            self.omni.close()

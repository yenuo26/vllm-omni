import os

import pytest
import torch
import socket
import subprocess
import sys
import time
import base64
from vllm.logger import init_logger
from vllm.utils import get_open_port
from vllm.assets.video import VideoAsset



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
    system_prompt: str,
    video_data_url: str,
    audio_data_url: str,
    image_data_url: str,
    content_text: str = "What is recited in the audio? What is in this image? Describe the video briefly.",
):
    """Create messages with video、image、audio data URL for OpenAI API."""
    content = [{"type": "text", "text": content_text}]
    if video_data_url is not None:
        content.append({"type": "video_url", "video_url": {"url": video_data_url}})
    if image_data_url is not None:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    if audio_data_url is not None:
        content.append({"type": "audio_url", "audio_url": {"url": audio_data_url}})
    return [
        system_prompt,
        {
            "role": "user",
            "content": content
        },
    ]


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

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Literal

import aiohttp
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.benchmarks import datasets
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    StreamedResponseHandler,
    _get_chat_content,
    _update_headers_common,
    _update_payload_common,
    _validate_api_url,
)

from vllm_omni.benchmarks.data_modules.random_multi_modal_dataset import OmniRandomMultiModalDataset

get_samples_old = datasets.get_samples


def get_samples(args, tokenizer):
    if args.dataset_name == "random-mm":
        if args.backend not in ["openai-chat"]:
            raise ValueError("Multi-modal content (images) is only supported on 'openai-chat' backend.")
        dataset = OmniRandomMultiModalDataset(random_seed=args.seed, dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            prefix_len=args.random_prefix_len,
            range_ratio=args.random_range_ratio,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            base_items_per_request=args.random_mm_base_items_per_request,
            limit_mm_per_prompt=args.random_mm_limit_mm_per_prompt,
            num_mm_items_range_ratio=args.random_mm_num_mm_items_range_ratio,
            bucket_config=args.random_mm_bucket_config,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )
        return input_requests
    else:
        return get_samples_old(args, tokenizer)


datasets.get_samples = get_samples

# ruff: noqa: E402
# Prevent import order from causing patch failures
from vllm.benchmarks.lib import endpoint_request_func

RequestFuncOutput_old = endpoint_request_func.RequestFuncOutput


@dataclass
class RequestFuncOutput(RequestFuncOutput_old):
    audio_ttft: float = 0.0


endpoint_request_func.RequestFuncOutput = RequestFuncOutput


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    content = _get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": request_func_input.model_name if request_func_input.model_name else request_func_input.model,
        "messages": [
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_completion_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    modality = data.get("modality")
                                    if modality == "text":
                                        output.itl.append(timestamp - most_recent_timestamp)
                                    elif modality == "audio":
                                        output.audio_ttft = timestamp - most_recent_timestamp

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS["openai-chat"] = async_request_openai_chat_completions

# ruff: noqa: E402
# Prevent import order from causing patch failures
from vllm.benchmarks import serve

BenchmarkMetrics_old = serve.BenchmarkMetrics


@dataclass
class BenchmarkMetrics(BenchmarkMetrics_old):
    mean_audio_ttft_ms: float = 0.0
    median_audio_ttft_ms: float = 0.0
    std_audio_ttft_ms: float = 0.0
    percentiles_audio_ttft_ms: list[tuple[float, float]] = None


serve.BenchmarkMetrics = BenchmarkMetrics


calculate_metrics_old = serve.calculate_metrics


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
):
    from vllm_omni.benchmarks.metrics.metrics import calculate_metrics

    metrics, actual_output_lens = calculate_metrics_old(
        input_requests, outputs, dur_s, tokenizer, selected_percentiles, goodput_config_dict
    )
    metrics = calculate_metrics(outputs, selected_percentiles, metrics)
    return metrics, actual_output_lens


serve.calculate_metrics = calculate_metrics

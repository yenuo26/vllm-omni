import warnings
from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import MILLISECONDS_TO_SECONDS_CONVERSION, TERM_PLOTLIB_AVAILABLE, BenchmarkMetrics, TaskType


@dataclass
class MultiModalsBenchmarkMetrics(BenchmarkMetrics):
    mean_audio_ttfp_ms: float = 0.0
    median_audio_ttfp_ms: float = 0.0
    std_audio_ttfp_ms: float = 0.0
    percentiles_audio_ttfp_ms: list[tuple[float, float]] = None
    total_audio_duration_ms: float = 0.0
    total_audio_frames: int = 0
    audio_throughput: float = 0.0
    mean_audio_rtf_ms: float = 0.0
    median_audio_rtf_ms: float = 0.0
    std_audio_rtf_ms: float = 0.0
    percentiles_audio_rtf_ms: list[tuple[float, float]] = None


def print_metrics(
    task_type,
    selected_percentile_metrics,
    max_concurrency,
    request_rate,
    benchmark_duration,
    goodput_config_dict,
    metrics: MultiModalsBenchmarkMetrics,
):
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        print("{:<40} {:<10.2f}".format("Peak concurrent requests:", metrics.max_concurrent_requests))
    print_text_metrics(task_type, selected_percentile_metrics, metrics)
    if task_type == TaskType.GENERATION:
        print_audio_metrics(selected_percentile_metrics, metrics)
    print("=" * 50)


def print_text_metrics(task_type, selected_percentile_metrics, metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Text Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
        print("{:<40} {:<10.2f}".format("Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s))
        print("{:<40} {:<10.2f}".format("Peak concurrent requests:", metrics.max_concurrent_requests))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    if task_type == TaskType.GENERATION:
        for metric in selected_percentile_metrics:
            if not metric.startswith("audio"):
                process_one_metric(metric, metrics)
    else:
        process_one_metric("e2el", metrics)


def print_audio_metrics(selected_percentile_metrics, metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Audio Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total audio duration generated(s):", metrics.total_audio_duration_ms))
    print("{:<40} {:<10}".format("Total audio frames generated:", metrics.total_audio_frames))
    print("{:<40} {:<10}".format("Audio throughput(audio duration/s):", metrics.audio_throughput))
    for metric in selected_percentile_metrics:
        if metric.startswith("audio"):
            process_one_metric(metric, metrics)


def process_one_metric(
    # E.g., "ttft"
    metric_attribute_name: str,
    metrics: MultiModalsBenchmarkMetrics,
):
    # This function prints and adds statistics of the specified
    # metric.
    metric_header_map = {
        "ttft": "Time to First Token",
        "tpot": "Time per Output Token (excl. 1st token)",
        "itl": "Inter-token Latency",
        "e2el": "End-to-end Latency",
        "audio_ttfp": "Time to First Packet",
        "audio_rtf": "Real Time Factor",
    }
    print("{s:{c}^{n}}".format(s=metric_header_map[metric_attribute_name], n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format(
            f"Mean {metric_attribute_name.upper()} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms"),
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            f"Median {metric_attribute_name.upper()} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms"),
        )
    )
    for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_attribute_name.upper()} (ms):", value))


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
    task_type,
    selected_percentile_metrics,
    max_concurrency,
    request_rate,
    benchmark_duration,
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark.

    Args:
        input_requests: The input requests.
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        tokenizer: The tokenizer to use.
        selected_percentiles: The percentiles to select.
        goodput_config_dict: The goodput configuration.

    Returns:
        A tuple of the benchmark metrics and the actual output lengths.
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    audio_ttfps: list[float] = []
    audio_rtfs: list[float] = []
    audio_duration: list[float] = []
    audio_frames: list[int] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            audio_ttfps.append(outputs[i].audio_ttfp)
            audio_rtfs.append(outputs[i].audio_rtf)
            audio_duration.append(outputs[i].audio_duration)
            audio_frames.append(outputs[i].audio_frames)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
            valid_metrics.append(audio_ttfps)
            slo_values.append(goodput_config_dict["audio_ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration on the benchmark arguments.",
            stacklevel=2,
        )

    # Calculate max output tokens per second metric
    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    # Find the time range across all successful requests
    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]
    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(output.start_time + output.latency for output in successful_outputs)

        # Create second buckets (ceiling to ensure we capture all time)
        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for i, output in enumerate(successful_outputs):
            # Calculate token generation timestamp using
            # start_time, ttft, and itl
            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)

            # Add tokens to second buckets
            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            # Track concurrent requests for each second this request was active
            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int((output.start_time + output.latency) - min_start_time)

            for second in range(request_start_second, request_end_second + 1):
                concurrent_requests_per_second[second] += 1

        # Find the maximum tokens per second and corresponding
        # concurrent requests
        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

        if TERM_PLOTLIB_AVAILABLE:
            import termplotlib as tpl

            fig = tpl.figure()
            fig.plot(
                np.arange(len(tokens_per_second)),
                tokens_per_second,
                title="Output tokens per second",
            )
            fig.plot(
                np.arange(len(concurrent_requests_per_second)),
                concurrent_requests_per_second,
                title="Concurrent requests per second",
            )
            fig.show()
        else:
            print("tip: install termplotlib and gnuplot to plot the metrics")

    metrics = MultiModalsBenchmarkMetrics(
        completed=completed,
        failed=len(failed_outputs),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by the endpoint
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles],
        mean_audio_ttfp_ms=np.mean(audio_ttfps or 0) * 1000,
        std_audio_ttfp_ms=np.std(audio_ttfps or 0) * 1000,
        median_audio_ttfp_ms=np.median(audio_ttfps or 0) * 1000,
        percentiles_audio_ttfp_ms=[(p, np.percentile(audio_ttfps or 0, p) * 1000) for p in selected_percentiles],
        total_audio_duration_ms=sum(audio_duration),
        total_audio_frames=sum(audio_frames),
        audio_throughput=sum(audio_duration) / dur_s,
        mean_audio_rtf_ms=np.mean(audio_rtfs or 0) * 1000,
        std_audio_rtf_ms=np.std(audio_rtfs or 0) * 1000,
        median_audio_rtf_ms=np.median(audio_rtfs or 0) * 1000,
        percentiles_audio_rtf_ms=[(p, np.percentile(audio_rtfs or 0, p) * 1000) for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles],
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )
    print_metrics(
        task_type,
        selected_percentile_metrics,
        max_concurrency,
        request_rate,
        benchmark_duration,
        goodput_config_dict,
        metrics,
    )
    return metrics, actual_output_lens

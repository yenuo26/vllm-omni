import numpy as np
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import BenchmarkMetrics


def calculate_metrics(outputs: list[RequestFuncOutput], selected_percentiles: list[float], metrics: BenchmarkMetrics):
    audio_ttfts = []
    for i in range(len(outputs)):
        if outputs[i] is not None and outputs[i].success:
            audio_ttfts.append(outputs[i].audio_ttft)
    mean_ttft_ms = np.mean(audio_ttfts or 0) * 1000
    std_ttft_ms = np.std(audio_ttfts or 0) * 1000
    median_ttft_ms = np.median(audio_ttfts or 0) * 1000
    percentiles_ttft_ms = [(p, np.percentile(audio_ttfts or 0, p) * 1000) for p in selected_percentiles]

    print("{s:{c}^{n}}".format(s=" Supplemental result ", n=50, c="="))
    print("{s:{c}^{n}}".format(s="Time to audio First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean Audio TTFT  (ms):", mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median Audio TTFT (ms):", median_ttft_ms))
    metrics.mean_audio_ttft_ms = mean_ttft_ms
    metrics.median_audio_ttft_ms = median_ttft_ms
    metrics.std_audio_ttft_ms = std_ttft_ms
    metrics.percentiles_audio_ttft_ms = percentiles_ttft_ms
    for p, value in percentiles_ttft_ms:
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.2f}".format(f"P{p_word} Audio TTFT (ms):", value))
    print("=" * 50)
    return metrics

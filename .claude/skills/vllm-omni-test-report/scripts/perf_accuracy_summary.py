#!/usr/bin/env python3
"""Buildkite + local performance and accuracy chapters for the release report.

Renders two H2 chapters — Performance Summary and Accuracy Summary — as
metric-value tables (one numeric column per metric). Each chapter is one
table per metric schema so scenarios that share scores stay together.
Filenames are
not table content. Values are quoted only from result JSON files (typically
the latest omni-reviewbot vllm-omni Buildkite archives by default (Local /
--log-dir-h* only when callers pass them) trees). Missing sources get an
honest empty note. Buildkite *API job* payloads do not carry metric scores,
so those are mentioned only as a note when no archived Buildkite JSON was
supplied.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from local_perf_results import local_perf_result_files
from md_table import render_markdown_table

_ACC_JOB_RE = re.compile(r"accurac|\bacc(?:uracy)?\b|precision|result_acc", re.I)
_PERF_JOB_RE = re.compile(r"perf|bench|throughput|latency|\bqps\b", re.I)
_DATE_IN_TEXT_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")

# Identity columns, always shown when any row has them.
_PERF_ID_COLS = ("Host", "Test", "Model", "Hardware", "Date", "Completed", "Failed")
_ACC_ID_COLS = (
    "Host",
    "Test",
    "Subset",
    "Dataset",
    "Hardware",
    "Date",
    "Status",
    "Case",
    "Task",
    "Category",
    "Duration",
)

# A zero modality throughput/latency means "this run did not measure that
# modality", not a real score. Those zeros are not copied onto a row.
_ZERO_SKIP_PERF_COLS = frozenset(
    {
        "image_throughput",
        "video_throughput",
        "audio_throughput",
        "mean_denoise_step_latency_ms",
        "mean_video_rtf",
        "mean_video_generation_ms",
        "mean_peak_memory_mb",
    }
)
# Shared percentile shape: mean, median, p95, p99. p50 is omitted because
# these archives store it as a duplicate of median.
_TEXT_PERF_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "duration",
)
_AUDIO_PERF_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "audio_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "mean_audio_ttfp_ms",
    "median_audio_ttfp_ms",
    "p95_audio_ttfp_ms",
    "p99_audio_ttfp_ms",
    "mean_audio_rtf",
    "median_audio_rtf",
    "p95_audio_rtf",
    "p99_audio_rtf",
    "mean_audio_duration_s",
    "median_audio_duration_s",
    "p99_audio_duration_s",
    "duration",
)
_OMNI_TEXT_AUDIO_METRICS = tuple(dict.fromkeys((*_TEXT_PERF_METRICS, *_AUDIO_PERF_METRICS)))
_IMAGE_PERF_METRICS = (
    "request_throughput",
    "image_throughput",
    "mean_denoise_step_latency_ms",
    "mean_peak_memory_mb",
    "median_peak_memory_mb",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "duration",
)
_VIDEO_PERF_METRICS = (
    "request_throughput",
    "video_throughput",
    "mean_video_rtf",
    "median_video_rtf",
    "p95_video_rtf",
    "p99_video_rtf",
    "mean_video_generation_ms",
    "median_video_generation_ms",
    "p95_video_generation_ms",
    "p99_video_generation_ms",
    "mean_denoise_step_latency_ms",
    "mean_peak_memory_mb",
    "median_peak_memory_mb",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "duration",
)
_REALTIME_PERF_METRICS = (
    "request_throughput",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "mean_video_ttfc_ms",
    "median_video_ttfc_ms",
    "p95_video_ttfc_ms",
    "p99_video_ttfc_ms",
    "mean_video_fps",
    "median_video_fps",
    "p95_video_fps",
    "p99_video_fps",
    "min_video_fps",
    "mean_video_rtf",
    "median_video_rtf",
    "p95_video_rtf",
    "p99_video_rtf",
    "mean_video_steady_interval_ms",
    "median_video_steady_interval_ms",
    "p95_video_steady_interval_ms",
    "p99_video_steady_interval_ms",
    "duration",
)
_DUPLEX_PERF_METRICS = (
    "request_throughput",
    "output_throughput",
    "audio_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "mean_audio_ttfp_ms",
    "median_audio_ttfp_ms",
    "p95_audio_ttfp_ms",
    "p99_audio_ttfp_ms",
    "mean_audio_rtf",
    "median_audio_rtf",
    "p95_audio_rtf",
    "p99_audio_rtf",
    "mean_duplex_response_ttft_ms",
    "median_duplex_response_ttft_ms",
    "p99_duplex_response_ttft_ms",
    "mean_duplex_response_ttfp_ms",
    "p99_duplex_response_ttfp_ms",
    "mean_duplex_client_ttft_ms",
    "p99_duplex_client_ttft_ms",
    "mean_duplex_client_ttfp_ms",
    "p99_duplex_client_ttfp_ms",
    "mean_duplex_client_rtf",
    "p99_duplex_client_rtf",
    "mean_duplex_busy_rtf",
    "p99_duplex_busy_rtf",
    "mean_duplex_stall_ratio",
    "p99_duplex_stall_ratio",
    "duration",
)
_DIFFUSION_PERF_METRICS = (
    "throughput_qps",
    "latency_mean",
    "latency_median",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "peak_memory_mb_max",
    "peak_memory_mb_mean",
    "peak_memory_mb_median",
    "duration",
)
_PERF_SCHEMA_ORDER = (
    "text_serving",
    "omni_text_audio",
    "audio_tts",
    "image_generation",
    "video_generation",
    "lingbot_realtime",
    "omni_duplex_perf",
    "diffusion",
)
_AUDIO_PERF_SIG = (
    "audio_throughput",
    "mean_audio_ttfp_ms",
    "mean_audio_rtf",
    "median_audio_rtf",
)
_TOKEN_PERF_SIG = ("mean_tpot_ms", "median_tpot_ms", "mean_itl_ms", "median_itl_ms")
_REALTIME_PERF_SIG = (
    "mean_video_ttfc_ms",
    "mean_video_fps",
    "min_video_fps",
    "mean_video_steady_interval_ms",
)
_DUPLEX_PERF_SIG = (
    "mean_duplex_response_ttft_ms",
    "mean_duplex_response_ttfp_ms",
    "mean_duplex_client_ttfp_ms",
    "mean_duplex_client_rtf",
    "mean_duplex_busy_rtf",
    "mean_duplex_stall_ratio",
)
_PERF_KEY_RE = re.compile(
    r"(throughput|_latency|latency_|_ms$|rtf|fps|qps|ttft|tpot|itl|e2el|peak_memory|ttfc|ttfp)",
    re.I,
)
_PERF_SKIP_KEYS = frozenset(
    {
        "num_prompts",
        "max_concurrency",
        "burstiness",
        "request_rate",
        "random_input_len",
        "random_output_len",
        "random_prefix_len",
        "total_input_tokens",
        "total_output_tokens",
        "total_audio_frames",
        "total_video_frames",
        "total_images",
        "total_audio_duration_s",
        "total_video_duration_s",
        "average_pixels_per_image",
        "max_output_tokens_per_s",
        "max_concurrent_requests",
        "completed",
        "failed",
        "completed_requests",
        "failed_requests",
    }
)
_ACC_METRIC_COLS = (
    "IA_QTF1",
    "Precision",
    "Recall",
    "Global_TP",
    "Global_FP",
    "Global_FN",
    "evaluated",
    "total",
)
# OmniInteract is identified by these scores. ``evaluated`` / ``total`` also
# appear on duplex eval, so they do not select the schema by themselves.
_OMNIINTERACT_SCORE_KEYS = (
    "IA_QTF1",
    "Precision",
    "Recall",
    "Global_TP",
    "Global_FP",
    "Global_FN",
)
_SIMILARITY_METRIC_COLS = ("ssim", "psnr", "alpha_mae")
_DAILY_OMNI_METRIC_COLS = (
    "daily_omni_accuracy",
    "daily_omni_accuracy_incl_http_fail",
    "daily_omni_correct",
    "daily_omni_evaluated",
    "daily_omni_evaluated_ok",
    "daily_omni_no_gold",
    "daily_omni_request_failed",
    "daily_omni_parse_failed",
)
_SEED_TTS_METRIC_COLS = (
    "seed_tts_content_metric",
    "seed_tts_content_error_mean",
    "seed_tts_content_error_median",
    "seed_tts_content_evaluated",
    "seed_tts_sim_mean",
    "seed_tts_sim_median",
    "seed_tts_sim_evaluated",
    "seed_tts_utmos_mean",
    "seed_tts_utmos_median",
    "seed_tts_utmos_evaluated",
)
_SEED_TTS_MEAN_KEYS = (
    "seed_tts_content_error_mean",
    "seed_tts_sim_mean",
    "seed_tts_utmos_mean",
)
_VBENCH_METRICS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
)
_WBENCH_METRICS = (
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "dynamic_degree",
    "motion_smoothness",
    "hpsv3_quality",
)
_QUALITY_METRICS = tuple(dict.fromkeys((*_VBENCH_METRICS, *_WBENCH_METRICS)))
_ACC_SCHEMA_ORDER = (
    "image_video_similarity",
    "daily_omni",
    "daily_omni_by_task",
    "daily_omni_by_category",
    "daily_omni_by_duration",
    "seed_tts",
    "omniinteract",
    "omni_duplex",
    "omni_duplex_rtd_by_task",
    "omni_duplex_pr_by_task",
    "lingbot_vbench",
    "lingbot_wbench",
    "lingbot_precision",
)
_ACC_SKIP_KEYS = frozenset(
    {
        "status",
        "errors",
        "error",
        "protocol_pin",
        "protocol_source",
        "judge_model",
        "judge_base_url",
        "score_root",
        "timing_precision",
        "summary",
        "items",
        "samples",
        "rtd",
        "pr",
    }
)
_SAMPLE_DUMP_RE = re.compile(r"(?:^|_)sample_\d+$|output_decoded_sample_|input_latent_sample_")
_INF_TEXT = frozenset({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})
_MODALITY_ZERO_KEYS = _ZERO_SKIP_PERF_COLS


def _fmt_metric(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def _short_model(model_id: Any) -> str:
    text = str(model_id or "").strip()
    if not text:
        return "—"
    if "models--" in text:
        try:
            mid = text.split("models--", 1)[1]
            mid = mid.split("/snapshots", 1)[0]
            return mid.replace("--", "/")
        except Exception:
            pass
    return text.rstrip("/").split("/")[-1] or text


def _iso_date_from_text(text: Any) -> str | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    m = _DATE_IN_TEXT_RE.search(raw.replace("T", " "))
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _iter_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _lookup(record: dict[str, Any], *keys: str) -> Any:
    """Return the first present value, checking ``record`` then ``record['result']``."""
    nested = record.get("result") if isinstance(record.get("result"), dict) else {}
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
        if key in nested and nested[key] not in (None, ""):
            return nested[key]
    return None


def local_acc_result_files(result_dir: Path) -> list[Path]:
    if not result_dir.is_dir():
        return []
    return sorted(path for path in result_dir.rglob("result_acc_*.json") if path.is_file())


def _perf_number(value: Any) -> int | float | None:
    """A measured performance figure. Zero means the modality was not run."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value == 0:
        return None
    return value


def _metric_map(record: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        number = _perf_number(_lookup(record, key))
        if number is not None:
            out[key] = number
    return out


def _baseline_profile(record: dict[str, Any]) -> dict[str, Any]:
    baseline = record.get("baseline")
    if not isinstance(baseline, dict) or not baseline:
        nested = record.get("result")
        if isinstance(nested, dict) and isinstance(nested.get("baseline"), dict):
            baseline = nested["baseline"]
        else:
            return {}
    hardware = str(_lookup(record, "Hardware", "hardware") or "").strip()
    chosen: dict[str, Any] | None = None
    if hardware:
        for key, value in baseline.items():
            if isinstance(value, dict) and str(key).lower() == hardware.lower():
                chosen = value
                break
    if chosen is None and not hardware:
        dict_values = [value for value in baseline.values() if isinstance(value, dict)]
        if len(dict_values) == 1:
            chosen = dict_values[0]
    if chosen is not None:
        return chosen
    if any(isinstance(value, dict) for value in baseline.values()):
        return {}
    return baseline


def _attach_perf_baseline(
    record: dict[str, Any], metrics: dict[str, Any], order: list[str]
) -> None:
    profile = _baseline_profile(record)
    for key in list(order):
        if key.startswith("baseline_"):
            continue
        number = _perf_number(profile.get(key))
        if number is None:
            continue
        baseline_key = f"baseline_{key}"
        if baseline_key in metrics:
            continue
        metrics[baseline_key] = number
        order.append(baseline_key)


def _classify_perf_record(
    record: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]] | None:
    """Return ``(schema, title, allowlist)`` for one performance result."""
    if _metric_map(record, _REALTIME_PERF_SIG):
        return "lingbot_realtime", "Lingbot-World realtime", _REALTIME_PERF_METRICS
    if _metric_map(record, _DUPLEX_PERF_SIG):
        return "omni_duplex_perf", "Omni duplex eval", _DUPLEX_PERF_METRICS
    if _metric_map(record, ("video_throughput", "mean_video_generation_ms")):
        return "video_generation", "Video generation", _VIDEO_PERF_METRICS
    if _metric_map(record, ("image_throughput", "mean_denoise_step_latency_ms")):
        return "image_generation", "Image generation", _IMAGE_PERF_METRICS
    audio = bool(_metric_map(record, _AUDIO_PERF_SIG))
    tokens = bool(_metric_map(record, _TOKEN_PERF_SIG))
    if audio and tokens:
        return "omni_text_audio", "Omni text and audio", _OMNI_TEXT_AUDIO_METRICS
    if audio:
        return "audio_tts", "Audio / TTS", _AUDIO_PERF_METRICS
    if _metric_map(
        record,
        (
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "mean_e2el_ms",
        ),
    ):
        return "text_serving", "Text serving", _TEXT_PERF_METRICS
    if _metric_map(record, ("throughput_qps", "latency_mean", "latency_median", "latency_p99")):
        return "diffusion", "Diffusion", _DIFFUSION_PERF_METRICS
    return None


def _fallback_perf_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Nonzero perf-like scalars that did not match a known family."""
    sources = [record]
    nested = record.get("result")
    if isinstance(nested, dict):
        sources.append(nested)
    found: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            if not isinstance(key, str) or key in _PERF_SKIP_KEYS or key.startswith("num_"):
                continue
            if key.startswith(("baseline", "stage_durations", "percentiles_", "random_")):
                continue
            if not _PERF_KEY_RE.search(key):
                continue
            number = _perf_number(value)
            if number is not None:
                found.setdefault(key, number)
    return found


def _perf_row_from_record(record: dict[str, Any], *, host: str) -> dict[str, Any] | None:
    classified = _classify_perf_record(record)
    if classified is None:
        metrics = _fallback_perf_metrics(record)
        if not metrics:
            return None
        order = sorted(metrics)
        schema = "other:" + "|".join(order)
        title = "Performance (" + ", ".join(order) + ")"
    else:
        schema, title, allowlist = classified
        metrics = _metric_map(record, allowlist)
        if not metrics:
            return None
        order = [key for key in allowlist if key in metrics]
    _attach_perf_baseline(record, metrics, order)
    model = _lookup(record, "model_id", "model")
    if model is None and isinstance(record.get("server_params"), dict):
        model = record["server_params"].get("model")
    when = _lookup(record, "date", "timestamp", "benchmark_end_time") or "—"
    test_name = _lookup(record, "test_name", "name")
    row: dict[str, Any] = {
        "Host": host or "—",
        "Test": str(test_name or "—"),
        "Model": _short_model(model) if model else "—",
        "Hardware": str(_lookup(record, "Hardware", "hardware") or "—"),
        "Date": str(when),
        "Completed": _lookup(record, "completed", "completed_requests"),
        "Failed": _lookup(record, "failed", "failed_requests"),
    }
    row.update(metrics)
    row["_schema"] = schema
    row["_title"] = title
    row["_metric_order"] = tuple(order)
    return row


def _scan_perf_dir(log_dir: Path, *, host: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    files = local_perf_result_files(log_dir)
    if not files:
        return rows, notes
    for path in files:
        payload = _load_json(path)
        if payload is None:
            notes.append(f"*Could not read `{path.name}` — performance rows skipped.*")
            continue
        file_rows: list[dict[str, Any]] = []
        for rec in _iter_records(payload):
            row = _perf_row_from_record(rec, host=host)
            if row is not None:
                file_rows.append(row)
        if file_rows:
            rows.extend(file_rows)
        else:
            notes.append(f"*No performance metrics in `{path.name}`.*")
    return rows, notes


def collect_local_perf_rows(log_dir: Path, *, host: str = "") -> list[dict[str, Any]]:
    rows, _notes = _scan_perf_dir(log_dir, host=host)
    return rows


def _is_metric_value(value: Any) -> bool:
    """True for a recorded score, count, bool, or an explicit infinity string."""
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return True
    if isinstance(value, str):
        return value.strip().lower() in _INF_TEXT
    return False


def _accuracy_object(record: dict[str, Any]) -> dict[str, Any]:
    acc = record.get("accuracy")
    return acc if isinstance(acc, dict) else {}


def _acc_identity(
    record: dict[str, Any],
    *,
    host: str,
    test: str | None = None,
    subset: str | None = None,
    case: str | None = None,
    breakdown_key: str | None = None,
    breakdown_value: str | None = None,
) -> dict[str, Any]:
    """Identity columns already used by the accuracy chapter.

    ``Test`` prefers the record's own test name, scenario label, or model.
    The result filename is not an identity value.
    """
    test_name = str(record.get("test_name") or "").strip()
    name = str(record.get("name") or "").strip()
    label = str(record.get("label") or "").strip()
    case_field = str(record.get("case") or "").strip()
    dataset = str(record.get("dataset") or "").strip()
    model = _short_model(record.get("model_id") or record.get("model"))
    if model == "—":
        model = ""
    if test is None:
        test = test_name or label or model or name or dataset or "—"
    if subset is None:
        if test_name and name:
            subset = name
        elif case_field and case_field != test:
            subset = case_field
        elif label and label != test:
            subset = label
        else:
            subset = "—"
    acc = _accuracy_object(record)
    if acc.get("status") not in (None, ""):
        status = str(acc.get("status"))
    elif record.get("benchmark_exit_code") not in (None, ""):
        status = str(record.get("benchmark_exit_code"))
    else:
        status = "—"
    row: dict[str, Any] = {
        "Host": host or "—",
        "Test": test or "—",
        "Subset": subset or "—",
        "Dataset": dataset or "—",
        "Hardware": str(record.get("Hardware") or record.get("hardware") or "—"),
        "Date": str(record.get("date") or "—"),
        "Status": status,
    }
    if case:
        row["Case"] = case
    if breakdown_key and breakdown_value:
        row[breakdown_key] = breakdown_value
    return row


def _pack_acc_row(
    record: dict[str, Any],
    *,
    host: str,
    schema: str,
    title: str,
    metric_order: tuple[str, ...],
    metrics: dict[str, Any],
    test: str | None = None,
    subset: str | None = None,
    case: str | None = None,
    breakdown_key: str | None = None,
    breakdown_value: str | None = None,
) -> dict[str, Any]:
    row = _acc_identity(
        record,
        host=host,
        test=test,
        subset=subset,
        case=case,
        breakdown_key=breakdown_key,
        breakdown_value=breakdown_value,
    )
    row.update(metrics)
    row["_schema"] = schema
    row["_title"] = title
    row["_metric_order"] = metric_order
    return row


def _baseline_threshold_scalars(record: dict[str, Any]) -> dict[str, Any]:
    """Scalar accuracy thresholds from ``baseline``.

    Hardware-keyed profiles contribute their scalars. Tensor sample dumps
    (lingbot precision baselines) are not thresholds and are dropped.
    """
    baseline = record.get("baseline")
    if not isinstance(baseline, dict) or not baseline:
        return {}
    scalars: dict[str, Any] = {}
    nested: list[tuple[str, dict[str, Any]]] = []
    for key, value in baseline.items():
        if isinstance(value, dict):
            nested.append((str(key), value))
        elif _is_metric_value(value) and not _SAMPLE_DUMP_RE.search(str(key)):
            scalars[str(key)] = value
    if nested:
        hardware = str(record.get("Hardware") or record.get("hardware") or "").strip().lower()
        chosen: dict[str, Any] | None = None
        if hardware:
            for key, value in nested:
                if key.lower() == hardware:
                    chosen = value
                    break
        if chosen is None and len(nested) == 1:
            chosen = nested[0][1]
        if chosen:
            for key, value in chosen.items():
                if _is_metric_value(value) and not _SAMPLE_DUMP_RE.search(str(key)):
                    scalars.setdefault(str(key), value)
    if len(scalars) > 12 and all(_SAMPLE_DUMP_RE.search(key) for key in scalars):
        return {}
    return scalars


def _merge_baseline(
    metrics: dict[str, Any], order: list[str], record: dict[str, Any]
) -> None:
    for key, value in _baseline_threshold_scalars(record).items():
        if key in metrics:
            continue
        metrics[key] = value
        order.append(key)


def _ordered_present(keys: tuple[str, ...], source: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    order: list[str] = []
    metrics: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if _is_metric_value(value):
            metrics[key] = value
            order.append(key)
    return order, metrics


def _similarity_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    order, metrics = _ordered_present(_SIMILARITY_METRIC_COLS, record)
    if "ssim" not in metrics and "psnr" not in metrics:
        return []
    # Latency, thresholds, and calibration are not the similarity scores.
    return [
        _pack_acc_row(
            record,
            host=host,
            schema="image_video_similarity",
            title="Image and video similarity",
            metric_order=tuple(order),
            metrics=metrics,
        )
    ]


def _daily_breakdown_rows(
    record: dict[str, Any],
    *,
    host: str,
    schema: str,
    title: str,
    accuracy_key: str,
    counts_key: str,
    github_key: str,
    label_column: str,
) -> list[dict[str, Any]]:
    accuracies = record.get(accuracy_key)
    if not isinstance(accuracies, dict):
        return []
    counts = record.get(counts_key) if isinstance(record.get(counts_key), dict) else {}
    github = record.get(github_key) if isinstance(record.get(github_key), dict) else {}
    parsed: list[tuple[str, Any, dict[str, Any], Any]] = []
    show_github = False
    for task, accuracy in accuracies.items():
        if not _is_metric_value(accuracy):
            continue
        count = counts.get(task) if isinstance(counts.get(task), dict) else {}
        gh = github.get(task)
        if _is_metric_value(gh) and _is_metric_value(accuracy):
            try:
                if abs(float(gh) - float(accuracy)) > 1e-12:
                    show_github = True
            except (TypeError, ValueError):
                show_github = True
        parsed.append((str(task), accuracy, count, gh))
    rows: list[dict[str, Any]] = []
    for task, accuracy, count, gh in parsed:
        metrics: dict[str, Any] = {"Accuracy": accuracy}
        order = ["Accuracy"]
        if _is_metric_value(count.get("correct")):
            metrics["Correct"] = count["correct"]
            order.append("Correct")
        if _is_metric_value(count.get("total")):
            metrics["Total"] = count["total"]
            order.append("Total")
        if show_github and _is_metric_value(gh):
            metrics["Accuracy (github style)"] = gh
            order.append("Accuracy (github style)")
        rows.append(
            _pack_acc_row(
                record,
                host=host,
                schema=schema,
                title=title,
                metric_order=tuple(order),
                metrics=metrics,
                breakdown_key=label_column,
                breakdown_value=task,
            )
        )
    return rows


def _daily_omni_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    if not _is_metric_value(record.get("daily_omni_accuracy")):
        return []
    order, metrics = _ordered_present(_DAILY_OMNI_METRIC_COLS, record)
    _merge_baseline(metrics, order, record)
    rows = [
        _pack_acc_row(
            record,
            host=host,
            schema="daily_omni",
            title="Daily Omni",
            metric_order=tuple(order),
            metrics=metrics,
        )
    ]
    rows.extend(
        _daily_breakdown_rows(
            record,
            host=host,
            schema="daily_omni_by_task",
            title="Daily Omni by task",
            accuracy_key="daily_omni_per_task_accuracy",
            counts_key="daily_omni_per_task",
            github_key="daily_omni_per_task_accuracy_github_style",
            label_column="Task",
        )
    )
    rows.extend(
        _daily_breakdown_rows(
            record,
            host=host,
            schema="daily_omni_by_category",
            title="Daily Omni by category",
            accuracy_key="daily_omni_per_category_accuracy",
            counts_key="daily_omni_per_category",
            github_key="daily_omni_per_category_accuracy_github_style",
            label_column="Category",
        )
    )
    rows.extend(
        _daily_breakdown_rows(
            record,
            host=host,
            schema="daily_omni_by_duration",
            title="Daily Omni by duration",
            accuracy_key="daily_omni_per_duration_accuracy",
            counts_key="daily_omni_per_duration",
            github_key="daily_omni_per_duration_accuracy_github_style",
            label_column="Duration",
        )
    )
    return rows


def _seed_tts_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    if not any(_is_metric_value(record.get(key)) for key in _SEED_TTS_MEAN_KEYS):
        return []
    order, metrics = _ordered_present(_SEED_TTS_METRIC_COLS, record)
    # UTMOS columns only when a mean was recorded. A null mean is not a score.
    if "seed_tts_utmos_mean" not in metrics:
        for key in (
            "seed_tts_utmos_median",
            "seed_tts_utmos_evaluated",
        ):
            metrics.pop(key, None)
            if key in order:
                order.remove(key)
    content_metric = record.get("seed_tts_content_metric")
    if isinstance(content_metric, str) and content_metric.strip():
        if "seed_tts_content_metric" not in metrics:
            metrics["seed_tts_content_metric"] = content_metric.strip()
            order.insert(0, "seed_tts_content_metric")
    _merge_baseline(metrics, order, record)
    return [
        _pack_acc_row(
            record,
            host=host,
            schema="seed_tts",
            title="Seed-TTS",
            metric_order=tuple(order),
            metrics=metrics,
        )
    ]


def _omniinteract_metrics(acc: dict[str, Any]) -> dict[str, Any]:
    summary = acc.get("summary") if isinstance(acc.get("summary"), dict) else {}
    metrics: dict[str, Any] = {}
    for key in _ACC_METRIC_COLS:
        if _is_metric_value(acc.get(key)):
            metrics[key] = acc[key]
        elif _is_metric_value(summary.get(key)):
            metrics[key] = summary[key]
    return metrics


def _omniinteract_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    acc = _accuracy_object(record)
    metrics = _omniinteract_metrics(acc)
    if not any(key in metrics for key in _OMNIINTERACT_SCORE_KEYS):
        return []
    order = [key for key in _ACC_METRIC_COLS if key in metrics]
    _merge_baseline(metrics, order, record)
    return [
        _pack_acc_row(
            record,
            host=host,
            schema="omniinteract",
            title="OmniInteract",
            metric_order=tuple(order),
            metrics=metrics,
        )
    ]


def _duplex_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    acc = _accuracy_object(record)
    rtd = acc.get("rtd") if isinstance(acc.get("rtd"), dict) else {}
    pr = acc.get("pr") if isinstance(acc.get("pr"), dict) else {}
    metrics: dict[str, Any] = {}
    order: list[str] = []
    for key, source in (
        ("mean_content_score", rtd),
        ("mean_avg_temporal_score", rtd),
        ("mean_all_success", pr),
    ):
        if _is_metric_value(source.get(key)):
            metrics[key] = source[key]
            order.append(key)
    if not metrics:
        return []
    for key in ("evaluated", "total"):
        if _is_metric_value(acc.get(key)):
            metrics[key] = acc[key]
            order.append(key)
    _merge_baseline(metrics, order, record)
    rows = [
        _pack_acc_row(
            record,
            host=host,
            schema="omni_duplex",
            title="Omni duplex eval",
            metric_order=tuple(order),
            metrics=metrics,
        )
    ]
    by_task = rtd.get("by_task") if isinstance(rtd.get("by_task"), dict) else {}
    for task, scores in by_task.items():
        if not isinstance(scores, dict):
            continue
        task_metrics: dict[str, Any] = {}
        task_order: list[str] = []
        for key in ("mean_content_score", "mean_avg_temporal_score"):
            if _is_metric_value(scores.get(key)):
                task_metrics[key] = scores[key]
                task_order.append(key)
        if not task_metrics:
            continue
        rows.append(
            _pack_acc_row(
                record,
                host=host,
                schema="omni_duplex_rtd_by_task",
                title="Omni duplex RTD by task",
                metric_order=tuple(task_order),
                metrics=task_metrics,
                breakdown_key="Task",
                breakdown_value=str(task),
            )
        )
    pr_by_task = pr.get("by_task") if isinstance(pr.get("by_task"), dict) else {}
    for task, value in pr_by_task.items():
        if not _is_metric_value(value):
            continue
        rows.append(
            _pack_acc_row(
                record,
                host=host,
                schema="omni_duplex_pr_by_task",
                title="Omni duplex PR by task",
                metric_order=("all_success",),
                metrics={"all_success": value},
                breakdown_key="Task",
                breakdown_value=str(task),
            )
        )
    return rows


def _case_quality_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    suffixes = sorted(_QUALITY_METRICS, key=len, reverse=True)
    found: dict[str, dict[str, Any]] = {}
    for key, value in record.items():
        if not isinstance(key, str) or not _is_metric_value(value):
            continue
        for metric in suffixes:
            suffix = "_" + metric
            if key.endswith(suffix) and len(key) > len(suffix):
                case = key[: -len(suffix)]
                found.setdefault(case, {})[metric] = value
                break
    return found


def _quality_family(record: dict[str, Any], cases: dict[str, dict[str, Any]]) -> str | None:
    metrics = {metric for scores in cases.values() for metric in scores}
    dataset = str(record.get("dataset") or "").lower()
    if "hpsv3_quality" in metrics or "wbench" in dataset:
        if "subject_consistency" not in metrics and "background_consistency" not in metrics:
            return "wbench"
    if "subject_consistency" in metrics or "background_consistency" in metrics or "vbench" in dataset:
        if "hpsv3_quality" not in metrics:
            return "vbench"
    return None


def _lingbot_quality_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    cases = _case_quality_map(record)
    family = _quality_family(record, cases)
    if family is None or not cases:
        return []
    if family == "wbench":
        schema = "lingbot_wbench"
        title = "Lingbot-World WBench"
        columns = _WBENCH_METRICS
    else:
        schema = "lingbot_vbench"
        title = "Lingbot-World VBench"
        columns = _VBENCH_METRICS
    rows: list[dict[str, Any]] = []
    for case in sorted(cases):
        order, metrics = _ordered_present(columns, cases[case])
        if not metrics:
            continue
        rows.append(
            _pack_acc_row(
                record,
                host=host,
                schema=schema,
                title=title,
                metric_order=tuple(order),
                metrics=metrics,
                subset="—",
                case=case,
            )
        )
    return rows


def _precision_column(tensor: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", tensor).strip("_")
    return f"max_abs_{slug or 'error'}"


def _lingbot_precision_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    report = record.get("measurement_report")
    if not isinstance(report, dict):
        return []
    suite = str(report.get("suite") or "")
    dataset = str(record.get("dataset") or "")
    if "precision" not in suite.lower() and "precision" not in dataset.lower():
        return []
    metrics: dict[str, Any] = {}
    order: list[str] = []
    if isinstance(report.get("passed"), bool):
        metrics["passed"] = report["passed"]
        order.append("passed")
    comparison = report.get("sample_comparison")
    if isinstance(comparison, dict):
        max_abs = comparison.get("max_abs")
        if isinstance(max_abs, dict):
            for tensor, value in max_abs.items():
                if not _is_metric_value(value):
                    continue
                column = _precision_column(str(tensor))
                metrics[column] = value
                order.append(column)
        if _is_metric_value(comparison.get("sample_count")):
            metrics["sample_count"] = comparison["sample_count"]
            order.append("sample_count")
    tolerance = report.get("tolerance")
    if isinstance(tolerance, dict):
        for key in ("atol", "rtol"):
            if _is_metric_value(tolerance.get(key)):
                metrics[key] = tolerance[key]
                order.append(key)
    if not metrics:
        return []
    return [
        _pack_acc_row(
            record,
            host=host,
            schema="lingbot_precision",
            title="Lingbot-World precision",
            metric_order=tuple(order),
            metrics=metrics,
        )
    ]


def _fallback_accuracy_rows(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    """Scalar scores on an ``accuracy`` object that is not one of the known schemas."""
    acc = _accuracy_object(record)
    metrics: dict[str, Any] = {}
    for key, value in acc.items():
        if key in _ACC_SKIP_KEYS:
            continue
        if _is_metric_value(value):
            metrics[str(key)] = value
        elif isinstance(value, dict):
            for sub, sub_val in value.items():
                if sub in _ACC_SKIP_KEYS or isinstance(sub_val, (dict, list)):
                    continue
                if _is_metric_value(sub_val):
                    metrics[f"{key}_{sub}"] = sub_val
    if not metrics:
        return []
    order = tuple(sorted(metrics))
    schema = "other:" + "|".join(order)
    title = "Accuracy (" + ", ".join(order) + ")"
    _merge_baseline(metrics, list(order), record)
    order = tuple(key for key in (*order, *metrics) if key in metrics)
    # de-dupe while keeping sort, then any baseline keys appended by merge
    seen: list[str] = []
    for key in order:
        if key not in seen:
            seen.append(key)
    return [
        _pack_acc_row(
            record,
            host=host,
            schema=schema,
            title=title,
            metric_order=tuple(seen),
            metrics=metrics,
        )
    ]


def _acc_rows_from_record(record: dict[str, Any], *, host: str) -> list[dict[str, Any]]:
    """Map one ``result_acc_*.json`` object onto accuracy-table rows.

    Known schemas are exclusive: a similarity file is not also dumped as a
    performance row, and lingbot quality scores are not also a precision row.
    """
    if _is_metric_value(record.get("ssim")) or _is_metric_value(record.get("psnr")):
        return _similarity_rows(record, host=host)
    if _is_metric_value(record.get("daily_omni_accuracy")):
        return _daily_omni_rows(record, host=host)
    if any(_is_metric_value(record.get(key)) for key in _SEED_TTS_MEAN_KEYS):
        return _seed_tts_rows(record, host=host)
    interact = _omniinteract_rows(record, host=host)
    if interact:
        return interact
    duplex = _duplex_rows(record, host=host)
    if duplex:
        return duplex
    quality = _lingbot_quality_rows(record, host=host)
    if quality:
        return quality
    precision = _lingbot_precision_rows(record, host=host)
    if precision:
        return precision
    return _fallback_accuracy_rows(record, host=host)


def _scan_acc_dir(log_dir: Path, *, host: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    files = local_acc_result_files(log_dir)
    if not files:
        return rows, notes
    for path in files:
        payload = _load_json(path)
        if payload is None:
            notes.append(f"*Could not read `{path.name}` — accuracy rows skipped.*")
            continue
        records = _iter_records(payload)
        file_rows: list[dict[str, Any]] = []
        for rec in records:
            file_rows.extend(_acc_rows_from_record(rec, host=host))
        if file_rows:
            rows.extend(file_rows)
        else:
            notes.append(f"*No accuracy metrics in `{path.name}`.*")
    return rows, notes


def collect_local_acc_rows(log_dir: Path, *, host: str = "") -> list[dict[str, Any]]:
    rows, _notes = _scan_acc_dir(log_dir, host=host)
    return rows


def _partition_acc_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]], tuple[str, ...]]]:
    """Group accuracy rows that share a metric schema."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    titles: dict[str, str] = {}
    metric_orders: dict[str, list[str]] = {}
    seen: list[str] = []
    for row in rows:
        schema = str(row.get("_schema") or "other")
        if schema not in buckets:
            buckets[schema] = []
            seen.append(schema)
            titles[schema] = str(row.get("_title") or schema)
            metric_orders[schema] = []
        buckets[schema].append(row)
        for key in row.get("_metric_order") or ():
            if key not in metric_orders[schema]:
                metric_orders[schema].append(key)
    rank = {schema: index for index, schema in enumerate(_ACC_SCHEMA_ORDER)}
    seen.sort(key=lambda schema: (rank.get(schema, len(_ACC_SCHEMA_ORDER)), schema))
    return [
        (titles[schema], buckets[schema], tuple(metric_orders[schema])) for schema in seen
    ]


def classify_buildkite_job_name(name: str) -> str | None:
    """Return ``accuracy``, ``performance``, or ``None`` for a Buildkite job name."""
    if _ACC_JOB_RE.search(name or ""):
        return "accuracy"
    if _PERF_JOB_RE.search(name or ""):
        return "performance"
    return None


def _column_has_value(rows: list[dict[str, Any]], key: str) -> bool:
    if key in _MODALITY_ZERO_KEYS:
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
                return True
            if value not in (None, "", "—", 0, 0.0):
                return True
        return False
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "—"):
            return True
    return False


def _render_metric_table(
    rows: list[dict[str, Any]],
    *,
    id_cols: tuple[str, ...],
    metric_cols: tuple[str, ...],
    extra_metric_keys: tuple[str, ...] = (),
) -> str:
    headers = [col for col in id_cols if _column_has_value(rows, col)]
    seen = set(headers)
    for col in (*metric_cols, *extra_metric_keys):
        if col in seen:
            continue
        if _column_has_value(rows, col):
            headers.append(col)
            seen.add(col)
    body: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for col in headers:
            value = row.get(col)
            if col in id_cols and isinstance(value, str):
                cells.append(value if value else "—")
            else:
                cells.append(_fmt_metric(value))
        body.append(cells)
    return render_markdown_table(headers, body) + "\n"


def _buildkite_metric_note(*, kind: str, buildkite_sources: list[tuple[str, list[dict[str, str]]]]) -> str:
    if not buildkite_sources:
        return ""
    jobs = [job for _, group in buildkite_sources for job in group]
    matched = [job for job in jobs if classify_buildkite_job_name(job.get("name") or "") == kind]
    if not jobs:
        return (
            "*Buildkite builds were queried, but no reportable jobs were available. "
            "Job payloads do not include metric values, so they are not tabulated.*\n"
        )
    if not matched:
        return (
            f"*Buildkite builds were fetched, but no job names matching {kind} "
            "were present, and job payloads do not include metric values.*\n"
        )
    return (
        f"*Buildkite {kind} jobs were found in the fetched builds, but the API "
        "payload does not include metric values, so they are not shown in the "
        "table below.*\n"
    )


def _collect_all(
    log_dirs: list[tuple[str, Path | None]],
    *,
    kind: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for label, log_dir in log_dirs:
        flag = label.lower()
        if log_dir is None:
            notes.append(f"*No `--log-dir-{flag}` supplied — {kind} rows for {label} skipped.*")
            continue
        path = Path(log_dir)
        if not path.is_dir():
            notes.append(f"*Log dir `{path}` does not exist — {kind} rows for {label} skipped.*")
            continue
        if kind == "performance":
            collected, unmapped = _scan_perf_dir(path, host=label)
            notes.extend(unmapped)
        else:
            collected, unmapped = _scan_acc_dir(path, host=label)
            notes.extend(unmapped)
        if not collected:
            # Files that exist but have no scores already added a note.
            if unmapped:
                continue
            if kind == "performance":
                notes.append(
                    f"*No local `result_test_*` / `diffusion_result_*` / "
                    f"`benchmark_results_*` files found under {label}.*"
                )
            else:
                notes.append(f"*No local `result_acc_*.json` files found under {label}.*")
            continue
        rows.extend(collected)
    return rows, notes



def _partition_perf_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]], tuple[str, ...]]]:
    """Group performance rows that share a metric schema."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    titles: dict[str, str] = {}
    metric_orders: dict[str, list[str]] = {}
    seen: list[str] = []
    for row in rows:
        schema = str(row.get("_schema") or "other")
        if schema not in buckets:
            buckets[schema] = []
            seen.append(schema)
            titles[schema] = str(row.get("_title") or schema)
            metric_orders[schema] = []
        buckets[schema].append(row)
        for key in row.get("_metric_order") or ():
            if key not in metric_orders[schema]:
                metric_orders[schema].append(key)
    rank = {schema: index for index, schema in enumerate(_PERF_SCHEMA_ORDER)}
    seen.sort(key=lambda schema: (rank.get(schema, len(_PERF_SCHEMA_ORDER)), schema))
    return [
        (titles[schema], buckets[schema], tuple(metric_orders[schema])) for schema in seen
    ]


def render_perf_accuracy_summary_section(
    *,
    log_dirs: list[tuple[str, Path | None]],
    report_date: str,
    buildkite_sources: list[tuple[str, list[dict[str, str]]]],
    source_notes: list[str] | None = None,
) -> str:
    """Markdown for ``## Performance Summary`` and ``## Accuracy Summary``.

    ``log_dirs`` is ``[(label, path_or_None), ...]`` (H200 / H800 / A100 / A3,
    plus Buildkite / Local archive roots from omni-reviewbot).
    ``buildkite_sources`` is ``[(build label, jobs), ...]``. API job lists are
    not tabulated: those payloads have no metric values.
    ``source_notes`` is an optional list of short source labels shown in the
    chapter intro (for example the omni-reviewbot archive paths used).
    """
    intro_tail = (
        f"(report date `{report_date}`). Each table lists the metric values "
        "found in the result JSON files; missing sources are called out "
        "rather than filled in."
    )
    if source_notes:
        intro_tail += " Sources: " + "; ".join(source_notes) + "."

    perf_rows, perf_notes = _collect_all(log_dirs, kind="performance")
    acc_rows, acc_notes = _collect_all(log_dirs, kind="accuracy")

    def _chapter(
        title: str,
        kind: str,
        rows: list[dict[str, Any]],
        notes: list[str],
        id_cols: tuple[str, ...],
        metric_cols: tuple[str, ...],
        empty: str,
    ) -> list[str]:
        host_labels = {label for label, _ in log_dirs if label}
        if host_labels <= {"Buildkite"}:
            archive_phrase = "**Buildkite archives**"
        elif host_labels <= {"Local"}:
            archive_phrase = "**Local archives**"
        else:
            archive_phrase = "**Buildkite and Local archives**"
        lines = [
            title,
            "",
            f"Same-day {kind} results from {archive_phrase} {intro_tail}",
            "",
        ]
        bk_note = _buildkite_metric_note(kind=kind, buildkite_sources=buildkite_sources)
        if bk_note:
            lines.append(bk_note)
            lines.append("")
        if rows and kind == "performance":
            for label, group, group_metrics in _partition_perf_rows(rows):
                lines.append(f"### {label}")
                lines.append("")
                lines.append(
                    _render_metric_table(
                        group,
                        id_cols=id_cols,
                        metric_cols=group_metrics,
                    )
                )
        elif rows and kind == "accuracy":
            for label, group, group_metrics in _partition_acc_rows(rows):
                lines.append(f"### {label}")
                lines.append("")
                lines.append(
                    _render_metric_table(
                        group,
                        id_cols=id_cols,
                        metric_cols=group_metrics,
                    )
                )
        elif rows:
            lines.append(
                _render_metric_table(
                    rows,
                    id_cols=id_cols,
                    metric_cols=metric_cols,
                )
            )
        else:
            lines.append(empty)
            lines.append("")
        for note in notes:
            # When a table is present, skip per-host "no files" notes for hosts
            # that simply had no matching files — the table already shows what
            # was found. Keep notes about missing or invalid log dirs.
            if rows and "files found under" in note:
                continue
            lines.append(note)
            lines.append("")
        return lines

    lines: list[str] = []
    lines.extend(
        _chapter(
            "## Performance Summary",
            "performance",
            perf_rows,
            perf_notes,
            _PERF_ID_COLS,
            (),
            "*No local performance result files were found.*",
        )
    )
    lines.extend(
        _chapter(
            "## Accuracy Summary",
            "accuracy",
            acc_rows,
            acc_notes,
            _ACC_ID_COLS,
            _ACC_METRIC_COLS,
            "*No local `result_acc_*.json` files were found.*",
        )
    )
    return "\n".join(lines).rstrip() + "\n"

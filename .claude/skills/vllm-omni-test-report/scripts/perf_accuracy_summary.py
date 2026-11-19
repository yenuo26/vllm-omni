#!/usr/bin/env python3
"""Same-day Buildkite + local performance and accuracy summary for the release report.

Values are quoted only from the supplied local result files and Buildkite job
metadata. Missing sources render an honest empty note — nothing is invented.
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

# Compact metric keys, in display order. Only keys actually present are shown.
_PERF_THROUGHPUT_KEYS = (
    "throughput_qps",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "audio_throughput",
    "image_throughput",
    "video_throughput",
)
_PERF_LATENCY_KEYS = (
    "latency_mean",
    "latency_p99",
    "mean_ttft_ms",
    "mean_e2el_ms",
    "mean_audio_ttfp_ms",
)
_ACC_METRIC_KEYS = ("IA_QTF1", "Precision", "Recall", "Global_TP", "Global_FP", "Global_FN")


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
        # Hugging Face cache path: models--org--name/snapshots/<sha>
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


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | None:
    for key in keys:
        if key in record and record[key] not in (None, "", []):
            return key, record[key]
    return None


def local_acc_result_files(result_dir: Path) -> list[Path]:
    if not result_dir.is_dir():
        return []
    return sorted(path for path in result_dir.rglob("result_acc_*.json") if path.is_file())


def collect_local_perf_rows(log_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for path in local_perf_result_files(log_dir):
        payload = _load_json(path)
        if payload is None:
            rows.append([path.name, "—", "—", "—", "—", "—", f"*unreadable JSON: `{path.name}`*"])
            continue
        records = _iter_records(payload)
        if not records:
            rows.append([path.name, "—", "—", "—", "—", "—", f"`{path.name}`"])
            continue
        for rec in records:
            test = str(rec.get("test_name") or rec.get("name") or path.stem)
            hardware = str(rec.get("Hardware") or rec.get("hardware") or "—")
            when = str(
                rec.get("date")
                or rec.get("timestamp")
                or rec.get("benchmark_end_time")
                or "—"
            )
            completed = rec.get("completed")
            failed = rec.get("failed")
            if completed is None and isinstance(rec.get("result"), dict):
                completed = rec["result"].get("completed_requests")
                failed = rec["result"].get("failed_requests")
            cf = f"{_fmt_metric(completed)} / {_fmt_metric(failed)}"
            tput = _first_present(rec, _PERF_THROUGHPUT_KEYS)
            if tput is None and isinstance(rec.get("result"), dict):
                tput = _first_present(rec["result"], _PERF_THROUGHPUT_KEYS)
            lat = _first_present(rec, _PERF_LATENCY_KEYS)
            if lat is None and isinstance(rec.get("result"), dict):
                lat = _first_present(rec["result"], _PERF_LATENCY_KEYS)
            tput_s = f"{tput[0]}={_fmt_metric(tput[1])}" if tput else "—"
            lat_s = f"{lat[0]}={_fmt_metric(lat[1])}" if lat else "—"
            rows.append([test, hardware, when, cf, tput_s, lat_s, f"`{path.name}`"])
    return rows


def collect_local_acc_rows(log_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for path in local_acc_result_files(log_dir):
        payload = _load_json(path)
        if payload is None:
            rows.append([path.name, "—", "—", "—", "—", "—", "—", f"*unreadable JSON: `{path.name}`*"])
            continue
        for rec in _iter_records(payload):
            test = str(rec.get("test_name") or rec.get("name") or path.stem)
            subset = str(rec.get("name") or rec.get("dataset") or "—")
            hardware = str(rec.get("Hardware") or rec.get("hardware") or "—")
            when = str(rec.get("date") or "—")
            acc = rec.get("accuracy") if isinstance(rec.get("accuracy"), dict) else {}
            status = str(acc.get("status") or rec.get("benchmark_exit_code") or "—")
            metrics: list[str] = []
            for key in _ACC_METRIC_KEYS:
                if key in acc and acc[key] not in (None, ""):
                    metrics.append(f"{key}={_fmt_metric(acc[key])}")
            metric_s = ", ".join(metrics) if metrics else "—"
            baseline = rec.get("baseline") if isinstance(rec.get("baseline"), dict) else {}
            base_bits = [f"{k}={_fmt_metric(v)}" for k, v in baseline.items()] if baseline else []
            gen = rec.get("generation") if isinstance(rec.get("generation"), dict) else {}
            gen_s = "—"
            if gen:
                gen_s = (
                    f"{_fmt_metric(gen.get('completed'))} completed / "
                    f"{_fmt_metric(gen.get('failed'))} failed"
                )
            rows.append(
                [
                    test,
                    subset,
                    hardware,
                    when,
                    status,
                    metric_s,
                    ", ".join(base_bits) if base_bits else "—",
                    gen_s,
                    f"`{path.name}`",
                ]
            )
    return rows


def classify_buildkite_job_name(name: str) -> str | None:
    """Return ``accuracy``, ``performance``, or ``None`` for a Buildkite job name."""
    if _ACC_JOB_RE.search(name or ""):
        return "accuracy"
    if _PERF_JOB_RE.search(name or ""):
        return "performance"
    return None


def _buildkite_job_table(jobs: list[dict[str, str]], kind: str) -> str:
    rows: list[list[str]] = []
    for job in jobs:
        name = job.get("name") or ""
        if classify_buildkite_job_name(name) != kind:
            continue
        state = job.get("state") or "—"
        url = job.get("url") or ""
        link = f"[open]({url})" if url else "—"
        build = job.get("build") or "—"
        rows.append([name, state, build, link])
    if not rows:
        return (
            f"*No Buildkite job names matching {kind} were present in the "
            "fetched builds. Job states only — no metric values are inferred.*\n"
        )
    return render_markdown_table(["Job", "State", "Build", "Link"], rows) + "\n"


def _local_block(*, label: str, log_dir: Path | None, kind: str) -> str:
    heading = f"#### Local — {label}"
    if log_dir is None:
        return f"{heading}\n\n*No `--log-dir-{label.lower()}` supplied — {kind} summary skipped.*\n"
    if not Path(log_dir).is_dir():
        return f"{heading}\n\n*Log dir `{log_dir}` does not exist — {kind} summary skipped.*\n"
    if kind == "performance":
        rows = collect_local_perf_rows(Path(log_dir))
        headers = [
            "Test",
            "Hardware",
            "Date",
            "Completed / Failed",
            "Throughput",
            "Latency",
            "Source",
        ]
        empty = "*No local `result_test_*` / `diffusion_result_*` / `benchmark_results_*` files found.*\n"
    else:
        rows = collect_local_acc_rows(Path(log_dir))
        headers = [
            "Test",
            "Subset",
            "Hardware",
            "Date",
            "Status",
            "Metrics",
            "Baseline",
            "Generation",
            "Source",
        ]
        empty = "*No local `result_acc_*.json` files found.*\n"
    if not rows:
        return f"{heading}\n\n{empty}"
    return f"{heading}\n\n" + render_markdown_table(headers, rows) + "\n"


def render_perf_accuracy_summary_section(
    *,
    log_dirs: list[tuple[str, Path | None]],
    report_date: str,
    buildkite_sources: list[tuple[str, list[dict[str, str]]]],
) -> str:
    """Markdown for ``## Performance and Accuracy Summary``.

    ``log_dirs`` is ``[(label, path_or_None), ...]`` (H200 / H800 / A100 / A3).
    ``buildkite_sources`` is ``[(build label, jobs), ...]`` where each job is a
    dict with ``name``, ``state``, ``url``, and optional ``build``.
    """
    lines: list[str] = [
        "## Performance and Accuracy Summary",
        "",
        "Same-day performance and accuracy results from **Buildkite** and **local "
        f"sync** (report date `{report_date}`). Values are quoted only from the "
        "result files and job metadata that were actually found; missing sources "
        "are called out rather than filled in.",
        "",
        "### Performance",
        "",
    ]
    bk_jobs_all: list[dict[str, str]] = []
    for label, jobs in buildkite_sources:
        tagged = []
        for job in jobs:
            item = dict(job)
            item.setdefault("build", label)
            tagged.append(item)
        bk_jobs_all.extend(tagged)
    if buildkite_sources:
        lines.append("#### Buildkite")
        lines.append("")
        if not bk_jobs_all:
            lines.append(
                "*Buildkite builds were queried, but no reportable jobs were available.*\n"
            )
        else:
            lines.append(_buildkite_job_table(bk_jobs_all, "performance"))
    else:
        lines.append("#### Buildkite")
        lines.append("")
        lines.append("*No Buildkite builds were fetched for this chapter.*\n")
    lines.append("")
    for label, log_dir in log_dirs:
        lines.append(_local_block(label=label, log_dir=log_dir, kind="performance"))
        lines.append("")

    lines.extend(["### Accuracy", ""])
    if buildkite_sources:
        lines.append("#### Buildkite")
        lines.append("")
        if not bk_jobs_all:
            lines.append(
                "*Buildkite builds were queried, but no reportable jobs were available.*\n"
            )
        else:
            lines.append(_buildkite_job_table(bk_jobs_all, "accuracy"))
    else:
        lines.append("#### Buildkite")
        lines.append("")
        lines.append("*No Buildkite builds were fetched for this chapter.*\n")
    lines.append("")
    for label, log_dir in log_dirs:
        lines.append(_local_block(label=label, log_dir=log_dir, kind="accuracy"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

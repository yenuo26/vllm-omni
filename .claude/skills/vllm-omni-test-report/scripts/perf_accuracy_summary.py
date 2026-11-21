#!/usr/bin/env python3
"""Buildkite + local performance and accuracy chapters for the release report.

Renders two H2 chapters — Performance Summary and Accuracy Summary — as
metric-value tables (one numeric column per metric). Filenames are not table
content. Values are quoted only from result JSON files (typically the latest
omni-reviewbot vllm-omni Buildkite archives by default (Local / --log-dir-h* only when callers pass them)
trees). Missing sources get an honest empty note. Buildkite *API job*
payloads do not carry metric scores, so those are mentioned only as a note
when no archived Buildkite JSON was supplied.
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
_ACC_ID_COLS = ("Host", "Test", "Subset", "Dataset", "Hardware", "Date", "Status")

# Metric columns in display order. A column is emitted only when at least one
# row has a real value. Modality-specific throughputs that are all zero are
# dropped so an audio run does not sprout empty image/video columns.
_PERF_METRIC_COLS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "audio_throughput",
    "image_throughput",
    "video_throughput",
    "throughput_qps",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "mean_audio_ttfp_ms",
    "mean_audio_rtf",
    "latency_mean",
    "latency_median",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "duration",
    "baseline_throughput_qps",
    "baseline_latency_mean",
    "baseline_latency_p99",
)
_ZERO_SKIP_PERF_COLS = frozenset(
    {"image_throughput", "video_throughput", "audio_throughput"}
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


def _baseline_perf_metrics(record: dict[str, Any]) -> dict[str, Any]:
    baseline = record.get("baseline")
    if not isinstance(baseline, dict) or not baseline:
        return {}
    hardware = str(record.get("Hardware") or record.get("hardware") or "").strip()
    # diffusion: {"H200": {"throughput_qps": 0.39, ...}}
    nested: dict[str, Any] | None = None
    if hardware:
        for key, value in baseline.items():
            if isinstance(value, dict) and key.lower() == hardware.lower():
                nested = value
                break
    if nested is None:
        dict_values = [v for v in baseline.values() if isinstance(v, dict)]
        if len(dict_values) == 1:
            nested = dict_values[0]
    src = nested if nested is not None else baseline
    out: dict[str, Any] = {}
    for key in ("throughput_qps", "latency_mean", "latency_p99"):
        if key in src and src[key] not in (None, ""):
            out[f"baseline_{key}"] = src[key]
    return out


def collect_local_perf_rows(log_dir: Path, *, host: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in local_perf_result_files(log_dir):
        payload = _load_json(path)
        if payload is None:
            continue
        for rec in _iter_records(payload):
            model = _lookup(rec, "model_id", "model")
            if model is None and isinstance(rec.get("server_params"), dict):
                model = rec["server_params"].get("model")
            when = _lookup(rec, "date", "timestamp", "benchmark_end_time") or "—"
            completed = _lookup(rec, "completed", "completed_requests")
            failed = _lookup(rec, "failed", "failed_requests")
            row: dict[str, Any] = {
                "Host": host or "—",
                "Test": str(_lookup(rec, "test_name", "name") or path.stem),
                "Model": _short_model(model) if model else "—",
                "Hardware": str(_lookup(rec, "Hardware", "hardware") or "—"),
                "Date": str(when),
                "Completed": completed,
                "Failed": failed,
            }
            for key in _PERF_METRIC_COLS:
                if key.startswith("baseline_"):
                    continue
                value = _lookup(rec, key)
                if value is not None:
                    row[key] = value
            row.update(_baseline_perf_metrics(rec))
            rows.append(row)
    return rows


def collect_local_acc_rows(log_dir: Path, *, host: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in local_acc_result_files(log_dir):
        payload = _load_json(path)
        if payload is None:
            continue
        for rec in _iter_records(payload):
            acc = rec.get("accuracy") if isinstance(rec.get("accuracy"), dict) else {}
            row: dict[str, Any] = {
                "Host": host or "—",
                "Test": str(rec.get("test_name") or rec.get("name") or path.stem),
                "Subset": str(rec.get("name") or "—"),
                "Dataset": str(rec.get("dataset") or "—"),
                "Hardware": str(rec.get("Hardware") or rec.get("hardware") or "—"),
                "Date": str(rec.get("date") or "—"),
                "Status": str(acc.get("status") or rec.get("benchmark_exit_code") or "—"),
            }
            for key in _ACC_METRIC_COLS:
                if key in acc and acc[key] not in (None, ""):
                    row[key] = acc[key]
            baseline = rec.get("baseline") if isinstance(rec.get("baseline"), dict) else {}
            for key, value in baseline.items():
                if value not in (None, "") and not isinstance(value, (dict, list)):
                    row[key] = value
            rows.append(row)
    return rows


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
        collected = (
            collect_local_perf_rows(path, host=label)
            if kind == "performance"
            else collect_local_acc_rows(path, host=label)
        )
        if not collected:
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



_DIFFUSION_METRIC_HINTS = ("throughput_qps", "latency_mean", "latency_p99")
_SERVING_METRIC_HINTS = ("request_throughput", "output_throughput", "mean_ttft_ms")


def _row_has_any(row: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(row.get(key) not in (None, "", "—") for key in keys)


def _partition_perf_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Split serving vs diffusion rows so each table only shows relevant metrics."""
    serving: list[dict[str, Any]] = []
    diffusion: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for row in rows:
        is_diff = _row_has_any(row, _DIFFUSION_METRIC_HINTS)
        is_svc = _row_has_any(row, _SERVING_METRIC_HINTS)
        if is_diff and not is_svc:
            diffusion.append(row)
        elif is_svc and not is_diff:
            serving.append(row)
        else:
            other.append(row)
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    if serving:
        groups.append(("Serving / omni", serving))
    if diffusion:
        groups.append(("Diffusion", diffusion))
    if other:
        groups.append(("Other", other))
    return groups


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
        extra: tuple[str, ...] = ()
        if kind == "accuracy" and rows:
            known = set(id_cols) | set(metric_cols)
            extra = tuple(
                key
                for row in rows
                for key in row
                if key not in known and _column_has_value(rows, key)
            )
            # unique, stable
            extra = tuple(dict.fromkeys(extra))
        if rows and kind == "performance":
            for label, group in _partition_perf_rows(rows):
                lines.append(f"### {label}")
                lines.append("")
                lines.append(
                    _render_metric_table(
                        group,
                        id_cols=id_cols,
                        metric_cols=metric_cols,
                    )
                )
        elif rows:
            lines.append(
                _render_metric_table(
                    rows,
                    id_cols=id_cols,
                    metric_cols=metric_cols,
                    extra_metric_keys=extra,
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
            _PERF_METRIC_COLS,
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

#!/usr/bin/env python3
"""Open ``ci-failure`` issues vs the CUDA / NPU nightlies already loaded.

Ported from the unmerged Issue close confirmation work (PR #13) and used by
the nightly report's Issue section. An issue is marked closable only when
there is positive evidence that the previously failing job is now fully
green on every relevant pipeline. Missing data, a skipped or not-run job,
a weak name match, or a CUDA vs NPU mismatch is not closable.

This module does not render a chapter and never closes or comments on issues.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pytest_log_parse import extract_pytest_counts  # noqa: E402

ISSUE_REPO = "vllm-project/vllm-omni"
ISSUE_REPO_API = f"https://api.github.com/repos/{ISSUE_REPO}/issues"

HISTORY_LIMIT = 14
_SCHEDULED_NIGHTLY_RE = re.compile(r"scheduled\s+nightly", re.I)
_WEEKLY_RE = re.compile(r"\bweekly\b", re.I)

_NODE_RE = re.compile(
    r"(?:[\w./-]+/)?[\w.-]+\.py::[\w]+(?:\[[^\]]+\])?"
)
_EMOJI_ALIAS_RE = re.compile(r":[a-z0-9_+\-]+:", re.I)
_NPU_RE = re.compile(r"\bnpu\b", re.I)
_CUDA_RE = re.compile(r"\b(cuda|h100|h200|h800|\bl4\b)\b", re.I)
# Title/label tags already used when filing issues. [B200] is a CUDA-family tag.
_NPU_TAG_RE = re.compile(r"\[\s*NPU\s*\]", re.I)
_CUDA_TAG_RE = re.compile(r"\[\s*(CUDA|B200|H100|H200|H800|L4)\s*\]", re.I)
# Buildkite pipeline slugs only. Bare "vllm-omni" (docs, collect_env, the
# GitHub repo) must not count as the CUDA pipeline, and vllm-omni-npu-ci
# must not count as vllm-omni.
_NPU_PIPELINE_RE = re.compile(
    r"buildkite\.com/[^/\s]+/vllm-omni-npu-ci(?![A-Za-z0-9-])"
    r"|(?<![A-Za-z0-9-])vllm/vllm-omni-npu-ci(?![A-Za-z0-9-])",
    re.I,
)
_CUDA_PIPELINE_RE = re.compile(
    r"buildkite\.com/[^/\s]+/vllm-omni(?!-npu-ci|-amd-ci)(?![A-Za-z0-9-])"
    r"|(?<![A-Za-z0-9-])vllm/vllm-omni(?!-npu-ci|-amd-ci)(?![A-Za-z0-9-])",
    re.I,
)
_PYTEST_COUNTS_RE = re.compile(r"\d+\s+(?:passed|failed|skipped|errors?)\b", re.I)

# A job-name hint must be at least this distinctive to count as a match.
_MIN_JOB_HINT_LEN = 18
_MIN_JOB_HINT_TOKENS = 3

STATUS_CLOSABLE = "closable"
STATUS_FAILING = "still_failing"
STATUS_INSUFFICIENT = "insufficient"


@dataclass
class PipelineSnapshot:
    """Latest scheduled nightly for one Buildkite pipeline."""

    label: str
    build_number: int | None = None
    build_url: str | None = None
    jobs: list[dict[str, Any]] = field(default_factory=list)
    available: bool = False
    note: str | None = None
    org: str = "vllm"
    pipeline: str = ""


@dataclass
class IssueConfirmation:
    issue: dict[str, Any]
    sides: tuple[str, ...]
    tests: list[str]
    jobs_by_side: dict[str, list[str]]
    results: dict[str, str]
    status: str
    reason: str
    clean_streak: int | None = None


# Back-compat alias used by earlier unit tests / importers.
CloseCandidate = IssueConfirmation


def normalize_job_name(name: str) -> str:
    s = _EMOJI_ALIAS_RE.sub(" ", name or "")
    s = s.replace("·", " ").replace("/", " ").replace("-", " ").replace("&", " ")
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def job_name_is_distinctive(name: str) -> bool:
    """True when a job title is specific enough to match an issue."""
    key = normalize_job_name(name)
    if not key:
        return False
    tokens = key.split()
    return len(key) >= _MIN_JOB_HINT_LEN or len(tokens) >= _MIN_JOB_HINT_TOKENS


def extract_node_ids(text: str) -> list[str]:
    """Unique pytest node ids in title/body order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _NODE_RE.finditer(text or ""):
        node = match.group(0).strip()
        if node not in seen:
            seen.add(node)
            out.append(node)
    return out


def _issue_labels(issue: dict[str, Any]) -> str:
    return " ".join(
        (lab.get("name") if isinstance(lab, dict) else str(lab)) or ""
        for lab in (issue.get("labels") or [])
    )


def _issue_blob(issue: dict[str, Any]) -> str:
    return f"{issue.get('title') or ''}\n{issue.get('body') or ''}\n{_issue_labels(issue)}"


def _title_labels(issue: dict[str, Any]) -> str:
    return f"{issue.get('title') or ''}\n{_issue_labels(issue)}"


def _pipeline_sides(text: str) -> set[str]:
    """CUDA / NPU sides named by a Buildkite pipeline slug."""
    sides: set[str] = set()
    if _NPU_PIPELINE_RE.search(text or ""):
        sides.add("NPU")
    if _CUDA_PIPELINE_RE.search(text or ""):
        sides.add("CUDA")
    return sides


def _mention_sides(text: str) -> set[str]:
    """CUDA / NPU sides named by a hardware tag or word."""
    sides: set[str] = set()
    blob = text or ""
    if _NPU_RE.search(blob) or _NPU_TAG_RE.search(blob):
        sides.add("NPU")
    if _CUDA_RE.search(blob) or _CUDA_TAG_RE.search(blob):
        sides.add("CUDA")
    return sides


def side_from_text(text: str) -> str | None:
    """Return CUDA or NPU when the text clearly names exactly one pipeline."""
    sides = _pipeline_sides(text) | _mention_sides(text)
    if sides == {"NPU"}:
        return "NPU"
    if sides == {"CUDA"}:
        return "CUDA"
    return None


def infer_sides(issue: dict[str, Any], *, matched_sides: set[str] | None = None) -> set[str]:
    """Which pipelines must be green before the issue can be closable.

    Primary signals are title/label tags and wording (``[NPU]``, ``[B200]``,
    H100, CUDA, …) and Buildkite pipeline slugs in the issue (``vllm-omni``
    vs ``vllm-omni-npu-ci``). Incidental mentions of the other hardware in
    the body — a related ticket, a decorator, a code path — do not expand
    the set once a primary signal is present. If the issue is silent on
    CUDA vs NPU, fall back to the full text and then to the pipelines
    where a job name actually matched.
    """
    blob = _issue_blob(issue)
    primary = _mention_sides(_title_labels(issue)) | _pipeline_sides(blob)
    if primary:
        return primary
    mentioned = _mention_sides(blob)
    if mentioned:
        return mentioned
    if matched_sides:
        return set(matched_sides)
    return set()


def match_jobs(issue_text: str, jobs: list[dict[str, Any]]) -> list[str]:
    """Return distinctive job names mentioned in the issue text."""
    hay = normalize_job_name(issue_text)
    if not hay:
        return []
    scored: list[tuple[int, str]] = []
    for rec in jobs:
        name = str(rec.get("name") or "").strip()
        key = normalize_job_name(name)
        if not key or not job_name_is_distinctive(name):
            continue
        if key in hay:
            scored.append((len(key), name))
    scored.sort(key=lambda pair: -pair[0])
    kept: list[str] = []
    kept_norm: list[str] = []
    for _length, name in scored:
        key = normalize_job_name(name)
        if any(key != other and key in other for other in kept_norm):
            continue
        kept.append(name)
        kept_norm.append(key)
    return kept


def node_still_failing(node: str, failed_nodes: list[str]) -> bool:
    """True when ``node`` appears among current failed/error node ids."""
    target = _node_key(node)
    return any(_node_key(n) == target or n.endswith(node) or node.endswith(n) for n in failed_nodes)


def _node_key(node: str) -> str:
    path, _, rest = (node or "").partition("::")
    stem = path.rsplit("/", 1)[-1]
    return f"{stem}::{rest}".strip().lower()


def job_kind(rec: dict[str, Any]) -> str:
    """``ok`` / ``fail`` / ``unknown`` — ``ok`` requires a passing pytest summary."""
    state = (rec.get("state") or "").strip().lower()
    if rec.get("log_error"):
        return "unknown"
    if state and state not in {"passed", "finished"}:
        if state in {"failed", "broken", "timed_out", "canceled", "cancelled"}:
            return "fail"
        return "unknown"
    info = rec.get("info")
    if not info:
        # A passed Buildkite step with no parsed pytest summary is not
        # positive evidence for a pytest job.
        return "unknown"
    if info.get("failed_nodes") or info.get("error_nodes"):
        return "fail"
    counts = extract_pytest_counts(info.get("summary"))
    if counts["failed"] or counts["error"]:
        return "fail"
    summ = (info.get("summary") or "").strip()
    if not summ:
        return "unknown"
    if _PYTEST_COUNTS_RE.search(summ):
        return "ok"
    return "unknown"


def job_state_passed(rec: dict[str, Any] | None) -> bool:
    """Historical-build check: Buildkite job state only (no log parse)."""
    if rec is None:
        return False
    return (rec.get("state") or "").strip().lower() in {"passed", "finished"}


def _failed_nodes_from_jobs(jobs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for rec in jobs:
        info = rec.get("info") or {}
        out.extend(info.get("failed_nodes") or [])
        out.extend(info.get("error_nodes") or [])
    return out


def _jobs_by_name(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(rec.get("name") or ""): rec for rec in jobs if rec.get("name")}


def _confirmation(
    issue: dict[str, Any],
    *,
    sides: tuple[str, ...],
    tests: list[str],
    jobs_by_side: dict[str, list[str]],
    results: dict[str, str],
    status: str,
    reason: str,
    clean_streak: int | None = None,
) -> IssueConfirmation:
    return IssueConfirmation(
        issue=issue,
        sides=sides,
        tests=tests,
        jobs_by_side=jobs_by_side,
        results=results,
        status=status,
        reason=reason,
        clean_streak=clean_streak,
    )


def evaluate_issue(
    issue: dict[str, Any],
    pipelines: dict[str, PipelineSnapshot],
) -> IssueConfirmation:
    """Confirm one issue against the latest loaded CUDA / NPU nightlies."""
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"
    tests = extract_node_ids(text)

    jobs_by_side: dict[str, list[str]] = {}
    matched_sides: set[str] = set()
    for label, snap in pipelines.items():
        names = match_jobs(text, snap.jobs)
        if names:
            jobs_by_side[label] = names
            matched_sides.add(label)

    sides = tuple(sorted(infer_sides(issue, matched_sides=matched_sides)))
    if not sides:
        return _confirmation(
            issue,
            sides=(),
            tests=tests,
            jobs_by_side=jobs_by_side,
            results={},
            status=STATUS_INSUFFICIENT,
            reason="Could not infer a CUDA or NPU scope, or no distinctive job name was found.",
        )
    if not any(jobs_by_side.get(side) for side in sides):
        return _confirmation(
            issue,
            sides=sides,
            tests=tests,
            jobs_by_side=jobs_by_side,
            results={},
            status=STATUS_INSUFFICIENT,
            reason="No distinctive job name in the issue matched a job on the required pipeline(s).",
        )

    results: dict[str, str] = {}
    reasons: list[str] = []
    saw_fail = False
    fail_bits: list[str] = []
    for side in sides:
        snap = pipelines.get(side)
        if snap is None or not snap.available or not snap.jobs:
            return _confirmation(
                issue,
                sides=sides,
                tests=tests,
                jobs_by_side=jobs_by_side,
                results=results,
                status=STATUS_INSUFFICIENT,
                reason=f"Latest {side} nightly results were not loaded.",
            )
        names = jobs_by_side.get(side) or []
        if not names:
            return _confirmation(
                issue,
                sides=sides,
                tests=tests,
                jobs_by_side=jobs_by_side,
                results=results,
                status=STATUS_INSUFFICIENT,
                reason=f"No matching {side} job was identified.",
            )
        index = _jobs_by_name(snap.jobs)
        failed = _failed_nodes_from_jobs(snap.jobs)
        build_ref = f"#{snap.build_number}" if snap.build_number is not None else "latest nightly"
        side_fail = False
        for name in names:
            rec = index.get(name)
            kind = job_kind(rec) if rec is not None else "unknown"
            if rec is None:
                return _confirmation(
                    issue,
                    sides=sides,
                    tests=tests,
                    jobs_by_side=jobs_by_side,
                    results=results,
                    status=STATUS_INSUFFICIENT,
                    reason=f"{side} job {name} was not present on {build_ref}.",
                )
            if kind == "unknown":
                return _confirmation(
                    issue,
                    sides=sides,
                    tests=tests,
                    jobs_by_side=jobs_by_side,
                    results=results,
                    status=STATUS_INSUFFICIENT,
                    reason=f"{side} job {name} on {build_ref} has no clear pass/fail evidence.",
                )
            if kind != "ok":
                side_fail = True
                fail_bits.append(f"{side} job {name} is not green on {build_ref}")
        node_hits = [node for node in tests if node_still_failing(node, failed)]
        if node_hits:
            side_fail = True
            fail_bits.append(
                f"tracked test node still failing on {side} {build_ref}: {', '.join(node_hits)}"
            )
        if side_fail:
            results[side] = f"failed on {build_ref}"
            saw_fail = True
        else:
            results[side] = f"passed on {build_ref}"
            reasons.append(
                f"{side} job{'s' if len(names) != 1 else ''} {', '.join(names)} fully green on {build_ref}"
            )

    if saw_fail:
        return _confirmation(
            issue,
            sides=sides,
            tests=tests,
            jobs_by_side=jobs_by_side,
            results=results,
            status=STATUS_FAILING,
            reason="; ".join(fail_bits) + ".",
            clean_streak=0,
        )

    if tests:
        reasons.append("tracked test node(s) are not in the current failure set")
    return _confirmation(
        issue,
        sides=sides,
        tests=tests,
        jobs_by_side=jobs_by_side,
        results=results,
        status=STATUS_CLOSABLE,
        reason="; ".join(reasons) + ".",
    )


def count_consecutive_clean_runs(
    job_names: list[str],
    history: list[tuple[int, list[dict[str, Any]]]],
) -> int:
    """Count newest-first scheduled nightlies whose named jobs all passed."""
    if not job_names or not history:
        return 0
    streak = 0
    for _number, jobs in history:
        index = _jobs_by_name(jobs)
        if not all(job_state_passed(index.get(name)) for name in job_names):
            break
        streak += 1
    return streak


def attach_clean_streaks(
    rows: list[IssueConfirmation],
    histories: dict[str, list[tuple[int, list[dict[str, Any]]]]],
) -> None:
    """Set ``clean_streak`` in place. Closable rows use min across required sides."""
    for row in rows:
        if row.status == STATUS_FAILING:
            row.clean_streak = 0
            continue
        if row.status != STATUS_CLOSABLE or not row.sides:
            row.clean_streak = None
            continue
        streaks: list[int] = []
        missing = False
        for side in row.sides:
            names = row.jobs_by_side.get(side) or []
            hist = histories.get(side) or []
            if not names or not hist:
                missing = True
                break
            streaks.append(count_consecutive_clean_runs(names, hist))
        row.clean_streak = None if missing else (min(streaks) if streaks else 0)


def snapshots_from_bk_results(
    bk_results: dict[Any, tuple[dict[str, Any] | None, list[dict[str, Any]] | None, Any]] | None,
) -> dict[str, PipelineSnapshot]:
    """Build CUDA/NPU snapshots from ``emit_report_html``'s ``bk_results`` map."""
    out: dict[str, PipelineSnapshot] = {}
    for target, payload in (bk_results or {}).items():
        label = getattr(target, "label", None) or str(target)
        org = getattr(target, "org", "vllm")
        pipeline = getattr(target, "pipeline", "")
        build, jobs, note = payload if payload else (None, None, None)
        number = int(build["number"]) if build and build.get("number") is not None else None
        url = None
        if number is not None and pipeline:
            url = f"https://buildkite.com/{org}/{pipeline}/builds/{number}"
        available = bool(build and jobs)
        out[label] = PipelineSnapshot(
            label=label,
            build_number=number,
            build_url=url,
            jobs=list(jobs or []),
            available=available,
            note=str(note) if note else None,
            org=str(org),
            pipeline=str(pipeline),
        )
    return out


def _bk_token() -> str:
    return (os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BUILDKITE_TOKEN") or "").strip()


def fetch_scheduled_nightly_history(
    snap: PipelineSnapshot,
    *,
    token: str | None = None,
    limit: int = HISTORY_LIMIT,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Newest-first history, starting with the snapshot already loaded for the report.

    Older entries include job *state* only (no log download). Failures to list
    or fetch older builds are ignored so the latest snapshot can still count
    as a streak of 1 when it is green.
    """
    history: list[tuple[int, list[dict[str, Any]]]] = []
    seen: set[int] = set()
    if snap.available and snap.build_number is not None:
        history.append((snap.build_number, snap.jobs))
        seen.add(snap.build_number)
    tok = (token if token is not None else _bk_token()).strip()
    if not tok or not snap.org or not snap.pipeline:
        return history
    try:
        from nightly_job_pytest_table import fetch_nightly_build, http_json
    except ImportError:
        return history
    try:
        url = (
            f"https://api.buildkite.com/v2/organizations/{snap.org}/pipelines/"
            f"{snap.pipeline}/builds?branch=main&per_page=50"
        )
        builds = http_json(url, tok)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return history
    numbers: list[int] = []
    ceiling = snap.build_number
    for build in builds or []:
        msg = build.get("message") or ""
        if not _SCHEDULED_NIGHTLY_RE.search(msg) or _WEEKLY_RE.search(msg):
            continue
        try:
            number = int(build["number"])
        except (KeyError, TypeError, ValueError):
            continue
        if ceiling is not None and number > ceiling:
            continue
        if number in seen:
            continue
        numbers.append(number)
        if len(history) + len(numbers) >= limit:
            break
    for number in numbers:
        try:
            raw = fetch_nightly_build(tok, number, org=snap.org, pipeline=snap.pipeline)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            RuntimeError,
            json.JSONDecodeError,
        ):
            break
        jobs = [
            {"name": j.get("name") or "", "state": j.get("state") or ""}
            for j in (raw.get("jobs") or [])
            if j.get("name")
        ]
        history.append((number, jobs))
        seen.add(number)
        if len(history) >= limit:
            break
    return history


def collect_confirmations(
    issues: list[dict[str, Any]],
    pipelines: dict[str, PipelineSnapshot],
    histories: dict[str, list[tuple[int, list[dict[str, Any]]]]] | None = None,
) -> list[IssueConfirmation]:
    rows = [evaluate_issue(issue, pipelines) for issue in issues]
    attach_clean_streaks(rows, histories or {})
    order = {STATUS_CLOSABLE: 0, STATUS_FAILING: 1, STATUS_INSUFFICIENT: 2}
    rows.sort(key=lambda row: (order.get(row.status, 9), int(row.issue.get("number") or 0)))
    return rows


def collect_close_candidates(
    issues: list[dict[str, Any]],
    pipelines: dict[str, PipelineSnapshot],
    histories: dict[str, list[tuple[int, list[dict[str, Any]]]]] | None = None,
) -> list[IssueConfirmation]:
    """Closable subset (kept for tests / callers that only want that list)."""
    return [row for row in collect_confirmations(issues, pipelines, histories) if row.status == STATUS_CLOSABLE]


def _github_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-nightly-report",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get_json(url: str, headers: dict[str, str], *, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers=headers)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_open_ci_failure_issues(gh_token: str | None) -> tuple[list[dict[str, Any]], str | None]:
    """Paginate open ``ci-failure`` issues (PRs excluded). Returns (items, note)."""
    headers = _github_headers(gh_token)
    items: list[dict[str, Any]] = []
    page = 1
    try:
        while True:
            url = f"{ISSUE_REPO_API}?state=open&labels=ci-failure&per_page=100&page={page}"
            batch = _http_get_json(url, headers=headers, timeout=60)
            if not batch:
                break
            for issue in batch:
                if issue.get("pull_request"):
                    continue
                items.append(issue)
            if len(batch) < 100:
                break
            page += 1
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        if items:
            return items, f"GitHub issue listing stopped early ({type(exc).__name__}); using a partial list."
        return [], "GitHub issue listing failed — open ci-failure issues could not be loaded."
    return items, None

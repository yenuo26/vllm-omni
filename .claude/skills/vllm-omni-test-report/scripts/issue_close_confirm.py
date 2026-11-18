#!/usr/bin/env python3
"""Issue close confirmation: open ``ci-failure`` issues vs CUDA / NPU CI.

Every still-open ``ci-failure`` issue is checked against the corresponding
CUDA and/or NPU scheduled nightlies already loaded for the report. An issue
is marked theoretically closable only when there is *positive* evidence that
the previously failing job is now fully green on every relevant pipeline.
A "consecutive clean runs" column counts how many successive scheduled
nightlies (newest first, up to ``HISTORY_LIMIT``) have not reproduced the
failure, using job state on each historical build.
"""

from __future__ import annotations

import html
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from pytest_log_parse import extract_pytest_counts

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
_PYTEST_COUNTS_RE = re.compile(r"\d+\s+(?:passed|failed|skipped|errors?)\b", re.I)

# A job-name hint must be at least this distinctive to count as a match.
_MIN_JOB_HINT_LEN = 18
_MIN_JOB_HINT_TOKENS = 3

STATUS_CLOSABLE = "closable"
STATUS_FAILING = "still_failing"
STATUS_INSUFFICIENT = "insufficient"

_STATUS_LABELS = {
    STATUS_CLOSABLE: "Theoretically closable",
    STATUS_FAILING: "Still failing",
    STATUS_INSUFFICIENT: "Insufficient evidence",
}

ISSUE_CLOSE_CONFIRM_CSS = """
.issue-close-root { margin: 0 0 0.25rem; }
.issue-close-intro, .issue-close-note {
  color: var(--dashboard-soft-text, #475569);
  font-size: 0.92rem;
  line-height: 1.45;
  margin: 0 0 0.85rem;
}
.issue-close-note { font-style: italic; }
.issue-close-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 0.7rem;
  margin: 0 0 0.85rem;
  font-size: 0.88rem;
}
.issue-close-stats strong { font-variant-numeric: tabular-nums; }
.issue-close-empty { color: var(--dashboard-soft-text, #64748b); }
.issue-close-tests code {
  font-size: 0.82rem;
  word-break: break-all;
}
.issue-close-badge {
  display: inline-block;
  border-radius: 999px;
  padding: 0.12rem 0.55rem;
  font-size: 0.78rem;
  font-weight: 700;
  white-space: nowrap;
}
.issue-close-badge--closable { background: rgba(16, 185, 129, 0.16); color: #047857; }
.issue-close-badge--still_failing { background: rgba(239, 68, 68, 0.15); color: #b91c1c; }
.issue-close-badge--insufficient { background: var(--dashboard-badge-bg, #edf3f8); color: var(--dashboard-badge-text, #435466); }
.issue-close-streak { font-variant-numeric: tabular-nums; font-weight: 650; }
.issue-close-reason { display: block; margin-top: 0.25rem; font-size: 0.78rem; color: var(--dashboard-soft-text, #64748b); }
"""

_SVG_CHECK = (
    '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
    '<polyline points="22 4 12 14.01 9 11.01"/>'
)


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


def extract_node_ids(text: str) -> list[str]:
    """Unique pytest node ids in title/body order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _NODE_RE.finditer(text or ""):
        node = m.group(0).strip()
        if node not in seen:
            seen.add(node)
            out.append(node)
    return out


def infer_sides(issue: dict[str, Any], *, matched_sides: set[str] | None = None) -> set[str]:
    """Which pipelines must be green before the issue can be closable.

    Label / title / body mentions win. If the issue is silent on CUDA vs
    NPU, fall back to the pipelines where a job name actually matched.
    """
    labels = " ".join(
        (lab.get("name") if isinstance(lab, dict) else str(lab)) or ""
        for lab in (issue.get("labels") or [])
    )
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}\n{labels}"
    has_npu = bool(_NPU_RE.search(text))
    has_cuda = bool(_CUDA_RE.search(text))
    if has_npu and not has_cuda:
        return {"NPU"}
    if has_cuda and not has_npu:
        return {"CUDA"}
    if has_npu and has_cuda:
        return {"CUDA", "NPU"}
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
        if not key:
            continue
        tokens = key.split()
        if len(key) < _MIN_JOB_HINT_LEN and len(tokens) < _MIN_JOB_HINT_TOKENS:
            continue
        if key in hay:
            scored.append((len(key), name))
    scored.sort(key=lambda p: -p[0])
    kept: list[str] = []
    kept_norm: list[str] = []
    for _n, name in scored:
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
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, RuntimeError, json.JSONDecodeError):
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
    rows.sort(key=lambda r: (order.get(r.status, 9), int(r.issue.get("number") or 0)))
    return rows


def collect_close_candidates(
    issues: list[dict[str, Any]],
    pipelines: dict[str, PipelineSnapshot],
    histories: dict[str, list[tuple[int, list[dict[str, Any]]]]] | None = None,
) -> list[IssueConfirmation]:
    """Closable subset (kept for tests / callers that only want that list)."""
    return [r for r in collect_confirmations(issues, pipelines, histories) if r.status == STATUS_CLOSABLE]


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


def _svg_icon(inner: str, *, size: int = 22) -> str:
    return (
        f'<svg class="ico" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'aria-hidden="true" focusable="false" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{inner}</svg>"
    )


def _heading_html(label: str, sub: str) -> str:
    return (
        "<h2>"
        '<span class="heading-row">'
        f'<span class="heading-ico">{_svg_icon(_SVG_CHECK)}</span>'
        '<span class="heading-text">'
        f'<span class="heading-label">{html.escape(label)}</span>'
        f'<span class="heading-sub">{html.escape(sub)}</span>'
        "</span></span></h2>"
    )


def _issue_link(issue: dict[str, Any]) -> str:
    num = issue.get("number")
    url = issue.get("html_url") or f"https://github.com/{ISSUE_REPO}/issues/{num}"
    title = (issue.get("title") or "").strip()
    short = title if len(title) <= 140 else title[:137] + "..."
    return (
        f'<a href="{html.escape(url, quote=True)}">#{html.escape(str(num))}</a> '
        f"{html.escape(short)}"
    )


def _side_result_cell(snap: PipelineSnapshot | None, result: str | None, *, required: bool) -> str:
    if not required:
        return '<span class="issue-close-empty">n/a</span>'
    if not result:
        return '<span class="issue-close-empty">insufficient evidence</span>'
    if snap and snap.build_url:
        return f'<a href="{html.escape(snap.build_url, quote=True)}">{html.escape(result)}</a>'
    return html.escape(result)


def _streak_cell(row: IssueConfirmation) -> str:
    if row.clean_streak is None:
        return '<span class="issue-close-empty">n/a</span>'
    n = row.clean_streak
    unit = "scheduled nightly" if n == 1 else "scheduled nightlies"
    return f'<span class="issue-close-streak">{n}</span> {html.escape(unit)}'


def _status_cell(row: IssueConfirmation) -> str:
    label = _STATUS_LABELS.get(row.status, row.status)
    return (
        f'<span class="issue-close-badge issue-close-badge--{html.escape(row.status, quote=True)}">'
        f"{html.escape(label)}</span>"
        f'<span class="issue-close-reason">{html.escape(row.reason)}</span>'
    )


def render_issue_close_confirm_html(
    *,
    issues: list[dict[str, Any]],
    candidates: list[IssueConfirmation],
    pipelines: dict[str, PipelineSnapshot],
    note: str | None = None,
    rows: list[IssueConfirmation] | None = None,
) -> str:
    table_rows = rows if rows is not None else candidates
    closable = sum(1 for r in table_rows if r.status == STATUS_CLOSABLE)
    failing = sum(1 for r in table_rows if r.status == STATUS_FAILING)
    insufficient = sum(1 for r in table_rows if r.status == STATUS_INSUFFICIENT)
    intro = (
        "Every open <code>ci-failure</code> issue on "
        f'<a href="https://github.com/{ISSUE_REPO}/issues?q=is%3Aissue+state%3Aopen+label%3Aci-failure">'
        f"{html.escape(ISSUE_REPO)}</a> is checked against the corresponding CUDA and/or NPU "
        "scheduled nightlies already loaded for this report. An issue is marked theoretically "
        "closable only when the previously failing job is fully green on every relevant pipeline, "
        "and any tracked test node is absent from the current failure set. "
        "<strong>Consecutive clean runs</strong> counts how many successive scheduled nightlies "
        f"(newest first, up to {HISTORY_LIMIT}), starting from this report's builds, have not "
        "reproduced the failure. Advisory only — confirm before closing."
    )
    notes_html = f'<p class="issue-close-note">{html.escape(note)}</p>' if note else ""
    stats = (
        '<div class="issue-close-stats">'
        f"<span><strong>{len(issues)}</strong> open ci-failure issues scanned</span>"
        f"<span><strong>{closable}</strong> theoretically closable</span>"
        f"<span><strong>{failing}</strong> still failing</span>"
        f"<span><strong>{insufficient}</strong> insufficient evidence</span>"
        "</div>"
    )

    if table_rows:
        body: list[str] = []
        for row in table_rows:
            tests = (
                "<br>".join(f"<code>{html.escape(n)}</code>" for n in row.tests)
                if row.tests
                else "—"
            )
            job_bits: list[str] = []
            for side in row.sides:
                for name in row.jobs_by_side.get(side, []):
                    job_bits.append(f"{html.escape(side)}: {html.escape(name)}")
            jobs = "<br>".join(job_bits) if job_bits else "—"
            scope = " + ".join(row.sides) if row.sides else "—"
            required = set(row.sides)
            cuda = _side_result_cell(
                pipelines.get("CUDA"), row.results.get("CUDA"), required="CUDA" in required
            )
            npu = _side_result_cell(
                pipelines.get("NPU"), row.results.get("NPU"), required="NPU" in required
            )
            body.append(
                "<tr>"
                f"<td>{_issue_link(row.issue)}</td>"
                f"<td>{html.escape(scope)}</td>"
                f'<td class="issue-close-tests">{tests}<br>{jobs}</td>'
                f"<td>{cuda}</td>"
                f"<td>{npu}</td>"
                f"<td>{_streak_cell(row)}</td>"
                f"<td>{_status_cell(row)}</td>"
                "</tr>"
            )
        table = (
            '<div class="table-scroll"><table class="summary">'
            "<thead><tr>"
            "<th>Issue</th><th>Scope</th><th>Tracked test / job</th>"
            "<th>Latest CUDA</th><th>Latest NPU</th>"
            "<th>Consecutive clean runs</th><th>Confirmation</th>"
            "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div>"
        )
    else:
        table = (
            '<p class="issue-close-empty">No open ci-failure issues were loaded for this run.</p>'
        )

    return "\n".join(
        [
            '<details class="bk-section panel nightly-root nightly-root--issue-close" open>',
            '<summary class="bk-section-summary">',
            _heading_html(
                "Issue close confirmation",
                "Open ci-failure issues checked against the latest CUDA / NPU nightlies (click to fold)",
            ),
            "</summary>",
            '<div class="bk-section-body"><div class="issue-close-root">',
            f'<p class="issue-close-intro">{intro}</p>',
            notes_html,
            stats,
            table,
            "</div></div></details>",
        ]
    )


def render_issue_close_confirm_section(
    bk_results: dict[Any, tuple[dict[str, Any] | None, list[dict[str, Any]] | None, Any]] | None,
    gh_token: str | None,
    *,
    enabled: bool = True,
) -> str:
    """Fetch, evaluate, and render the chapter (empty string if disabled)."""
    if not enabled:
        return ""
    pipelines = snapshots_from_bk_results(bk_results)
    issues, fetch_note = fetch_open_ci_failure_issues(gh_token)
    if not issues and fetch_note:
        return render_issue_close_confirm_html(
            issues=[],
            candidates=[],
            pipelines=pipelines,
            note=fetch_note,
            rows=[],
        )
    histories = {label: fetch_scheduled_nightly_history(snap) for label, snap in pipelines.items()}
    rows = collect_confirmations(issues, pipelines, histories)
    note = fetch_note
    if not any(s.available for s in pipelines.values()):
        extra = "Latest CUDA / NPU nightly results were not loaded, so no issue can be marked closable."
        note = f"{note} {extra}".strip() if note else extra
    return render_issue_close_confirm_html(
        issues=issues,
        candidates=[r for r in rows if r.status == STATUS_CLOSABLE],
        pipelines=pipelines,
        note=note,
        rows=rows,
    )

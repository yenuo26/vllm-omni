#!/usr/bin/env python3
"""Issue 关闭确认: open ``ci-failure`` issues vs latest CUDA / NPU nightlies.

An issue is listed as theoretically closable only when there is *positive*
evidence that the previously failing job is now fully green on every
relevant pipeline (CUDA and/or NPU). Missing logs, skipped jobs, or a
weak match are not enough.
"""

from __future__ import annotations

import html
import json
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from pytest_log_parse import extract_pytest_counts

ISSUE_REPO = "vllm-project/vllm-omni"
ISSUE_REPO_API = f"https://api.github.com/repos/{ISSUE_REPO}/issues"

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


@dataclass
class CloseCandidate:
    issue: dict[str, Any]
    sides: tuple[str, ...]
    tests: list[str]
    jobs_by_side: dict[str, list[str]]
    results: dict[str, str]
    reason: str


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
    """Which pipelines must be green before the issue can be listed.

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
    # Prefer longer (more specific) names; drop a shorter name only when it
    # is a strict substring of a longer match *and* both came from hints —
    # keep siblings like "TTS Perf Test 2 GPU" and "TTS Perf Test Single GPU".
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


def _failed_nodes_from_jobs(jobs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for rec in jobs:
        info = rec.get("info") or {}
        out.extend(info.get("failed_nodes") or [])
        out.extend(info.get("error_nodes") or [])
    return out


def _jobs_by_name(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(rec.get("name") or ""): rec for rec in jobs if rec.get("name")}


def evaluate_issue(
    issue: dict[str, Any],
    pipelines: dict[str, PipelineSnapshot],
) -> CloseCandidate | None:
    """Return a close candidate, or None when evidence is insufficient."""
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"
    tests = extract_node_ids(text)

    jobs_by_side: dict[str, list[str]] = {}
    matched_sides: set[str] = set()
    for label, snap in pipelines.items():
        names = match_jobs(text, snap.jobs)
        if names:
            jobs_by_side[label] = names
            matched_sides.add(label)

    sides = infer_sides(issue, matched_sides=matched_sides)
    if not sides:
        return None
    if not any(jobs_by_side.get(side) for side in sides):
        return None

    results: dict[str, str] = {}
    reasons: list[str] = []
    for side in sorted(sides):
        snap = pipelines.get(side)
        if snap is None or not snap.available or not snap.jobs:
            return None
        names = jobs_by_side.get(side) or []
        if not names:
            return None
        index = _jobs_by_name(snap.jobs)
        failed = _failed_nodes_from_jobs(snap.jobs)
        for name in names:
            rec = index.get(name)
            if rec is None or job_kind(rec) != "ok":
                return None
        for node in tests:
            if node_still_failing(node, failed):
                return None
        build_ref = f"#{snap.build_number}" if snap.build_number is not None else "latest nightly"
        job_list = ", ".join(names)
        results[side] = f"passed on {build_ref}"
        reasons.append(f"{side} job{'s' if len(names) != 1 else ''} {job_list} fully green on {build_ref}")

    if tests:
        reasons.append("tracked test node(s) are not in the current failure set")
    reason = "; ".join(reasons) + "."
    return CloseCandidate(
        issue=issue,
        sides=tuple(sorted(sides)),
        tests=tests,
        jobs_by_side=jobs_by_side,
        results=results,
        reason=reason,
    )


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
        )
    return out


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


def collect_close_candidates(
    issues: list[dict[str, Any]],
    pipelines: dict[str, PipelineSnapshot],
) -> list[CloseCandidate]:
    out: list[CloseCandidate] = []
    for issue in issues:
        cand = evaluate_issue(issue, pipelines)
        if cand is not None:
            out.append(cand)
    out.sort(key=lambda c: int(c.issue.get("number") or 0))
    return out


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


def render_issue_close_confirm_html(
    *,
    issues: list[dict[str, Any]],
    candidates: list[CloseCandidate],
    pipelines: dict[str, PipelineSnapshot],
    note: str | None = None,
) -> str:
    skipped = max(0, len(issues) - len(candidates))
    intro = (
        "Open <code>ci-failure</code> issues on "
        f'<a href="https://github.com/{ISSUE_REPO}/issues?q=is%3Aissue+state%3Aopen+label%3Aci-failure">'
        f"{html.escape(ISSUE_REPO)}</a> are compared against the latest CUDA and NPU "
        "scheduled nightlies already loaded for this report. An issue is listed only when "
        "the previously failing job is fully green on every relevant pipeline, and any "
        "tracked test node is absent from the current failure set. Advisory only — confirm "
        "before closing."
    )
    notes_html = f'<p class="issue-close-note">{html.escape(note)}</p>' if note else ""
    stats = (
        '<div class="issue-close-stats">'
        f"<span><strong>{len(issues)}</strong> open ci-failure issues scanned</span>"
        f"<span><strong>{len(candidates)}</strong> theoretically closable</span>"
        f"<span><strong>{skipped}</strong> not listed (insufficient evidence or still failing)</span>"
        "</div>"
    )

    if candidates:
        rows: list[str] = []
        for cand in candidates:
            tests = (
                "<br>".join(f"<code>{html.escape(n)}</code>" for n in cand.tests)
                if cand.tests
                else "—"
            )
            job_bits: list[str] = []
            for side in cand.sides:
                for name in cand.jobs_by_side.get(side, []):
                    job_bits.append(f"{html.escape(side)}: {html.escape(name)}")
            jobs = "<br>".join(job_bits) if job_bits else "—"
            scope = " + ".join(cand.sides)
            cuda = _side_result_cell(
                pipelines.get("CUDA"), cand.results.get("CUDA"), required="CUDA" in cand.sides
            )
            npu = _side_result_cell(
                pipelines.get("NPU"), cand.results.get("NPU"), required="NPU" in cand.sides
            )
            rows.append(
                "<tr>"
                f"<td>{_issue_link(cand.issue)}</td>"
                f"<td>{html.escape(scope)}</td>"
                f'<td class="issue-close-tests">{tests}<br>{jobs}</td>'
                f"<td>{cuda}</td>"
                f"<td>{npu}</td>"
                f"<td>{html.escape(cand.reason)}</td>"
                "</tr>"
            )
        table = (
            '<div class="table-scroll"><table class="summary">'
            "<thead><tr>"
            "<th>Issue</th><th>Scope</th><th>Tracked test / job</th>"
            "<th>Latest CUDA</th><th>Latest NPU</th><th>Why closable</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
    else:
        table = (
            '<p class="issue-close-empty">No theoretically closable issues on this run.</p>'
        )

    return "\n".join(
        [
            '<details class="bk-section panel nightly-root nightly-root--issue-close" open>',
            '<summary class="bk-section-summary">',
            _heading_html(
                "Issue 关闭确认 · close candidates",
                "Open ci-failure issues that latest CUDA / NPU nightlies suggest can be closed (click to fold)",
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
        )
    candidates = collect_close_candidates(issues, pipelines)
    note = fetch_note
    if not any(s.available for s in pipelines.values()):
        extra = "Latest CUDA / NPU nightly results were not loaded, so no issue can be marked closable."
        note = f"{note} {extra}".strip() if note else extra
    return render_issue_close_confirm_html(
        issues=issues,
        candidates=candidates,
        pipelines=pipelines,
        note=note,
    )

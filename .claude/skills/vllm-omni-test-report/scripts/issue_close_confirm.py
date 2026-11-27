#!/usr/bin/env python3
"""Open ``ci-failure`` issues vs the CUDA / NPU nightlies already loaded.

Ported from the unmerged Issue close confirmation work (PR #13) and used by
the nightly report's Issue section. An issue is marked closable only when
there is positive evidence that the previously failing job is now fully
green on every relevant pipeline. Missing data, a skipped or not-run job,
a weak name match, or a CUDA vs NPU mismatch is not closable.

Issues tagged ``[B200]`` are judged from Scheduled Release B200 builds (the
named ``release/vX.Y.Z`` branch, or ``main``), not from the daily scheduled
nightly. Daily nightlies are never used as a stand-in for that lane.

This module does not render a chapter and never closes or comments on issues.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.parse
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
_PYTEST_COUNTS_RE = re.compile(r"\d+\s+(?:passed|failed|skipped|errors?)\b", re.I)
_B200_TAG_RE = re.compile(r"\[\s*B200\s*\]", re.I)
_DAILY_CUDA_TAG_RE = re.compile(r"\[\s*(H100|H200|H800|L4)\s*\]", re.I)
# Same release-branch grammar as the duration classifier: exact release/vX.Y.Z
# (optional rcN / .postN). Sidecars like release/v0.34.0-docker are excluded.
_RELEASE_BRANCH_RE = re.compile(r"^release/v\d+\.\d+\.\d+(?:rc\d+|\.post\d+)?$")
_RELEASE_BRANCH_IN_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9._-])release/v\d+\.\d+\.\d+(?:rc\d+|\.post\d+)?(?![A-Za-z0-9._-])"
)
_B200_MESSAGE_TITLE = "scheduled release b200 build"
# B200 runs ~monthly; page deeply enough to find the latest (and a short streak).
_B200_LIST_PER_PAGE = 100
_B200_LIST_MAX_PAGES = 15

LANE_NIGHTLY = "scheduled-nightly"
LANE_B200 = "scheduled-release-b200"

# A job-name hint must be at least this distinctive to count as a match.
_MIN_JOB_HINT_LEN = 18
_MIN_JOB_HINT_TOKENS = 3

STATUS_CLOSABLE = "closable"
STATUS_FAILING = "still_failing"
STATUS_INSUFFICIENT = "insufficient"


@dataclass
class PipelineSnapshot:
    """Latest relevant build for one evidence lane (nightly or Release B200)."""

    label: str
    build_number: int | None = None
    build_url: str | None = None
    jobs: list[dict[str, Any]] = field(default_factory=list)
    available: bool = False
    note: str | None = None
    org: str = "vllm"
    pipeline: str = ""
    lane: str = LANE_NIGHTLY
    branch: str = "main"


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
    evidence: dict[str, PipelineSnapshot] = field(default_factory=dict)


# Back-compat alias used by earlier unit tests / importers.
CloseCandidate = IssueConfirmation


def _issue_blob(issue: dict[str, Any]) -> str:
    labels = " ".join(
        (lab.get("name") if isinstance(lab, dict) else str(lab)) or ""
        for lab in (issue.get("labels") or [])
    )
    return f"{issue.get('title') or ''}\n{issue.get('body') or ''}\n{labels}"


def is_b200_issue(issue: dict[str, Any]) -> bool:
    """True when the issue is tagged ``[B200]`` in the title, body, or labels."""
    return bool(_B200_TAG_RE.search(_issue_blob(issue)))


def _has_daily_cuda_tag(issue: dict[str, Any]) -> bool:
    """True when the issue is also tagged for a daily CUDA lane (e.g. ``[H100]``)."""
    return bool(_DAILY_CUDA_TAG_RE.search(_issue_blob(issue)))


def is_release_branch(branch: str) -> bool:
    """True for ``release/v<version>`` branches this repo actually cuts."""
    return bool(_RELEASE_BRANCH_RE.fullmatch((branch or "").strip()))


def _normalized_message_title(message: str | None) -> str:
    """First line of a build message, case-folded, ordinary whitespace collapsed.

    ``source`` is ignored. Only the first line is considered, so a commit body
    that quotes a scheduled title cannot reclassify the build.
    """
    line = (message or "").split("\n", 1)[0].replace("\r", "")
    return re.sub(r"[ \t]+", " ", line).strip().casefold()


def is_scheduled_release_b200(build: dict[str, Any]) -> bool:
    """True when the first-line message is ``Scheduled Release B200 build``."""
    return _normalized_message_title(build.get("message")) == _B200_MESSAGE_TITLE


def b200_branches_for_issue(issue: dict[str, Any]) -> tuple[str, ...]:
    """Release branches named in the issue, or ``main`` when none are named.

    Matches actual ``release/vX.Y.Z`` (optional rc / .post) tokens. Does not
    treat ``release/v0.34.0-docker`` or ``backport/10687-release-v0.34.0`` as
    release branches.
    """
    found: list[str] = []
    seen: set[str] = set()
    for match in _RELEASE_BRANCH_IN_TEXT_RE.finditer(_issue_blob(issue)):
        branch = match.group(0)
        if branch not in seen and is_release_branch(branch):
            seen.add(branch)
            found.append(branch)
    return tuple(found) if found else ("main",)


def _unavailable_b200_snapshot(branch: str, *, note: str | None = None) -> PipelineSnapshot:
    return PipelineSnapshot(
        label="CUDA",
        available=False,
        org="vllm",
        pipeline="vllm-omni",
        lane=LANE_B200,
        branch=branch or "main",
        note=note or f"Scheduled Release B200 build was not loaded on {branch}.",
    )


def evidence_pipelines_for_issue(
    issue: dict[str, Any],
    nightly: dict[str, PipelineSnapshot],
    b200_by_branch: dict[str, PipelineSnapshot] | None = None,
) -> dict[str, PipelineSnapshot]:
    """Pipelines that may close this issue.

    Ordinary issues keep the daily CUDA / NPU nightlies. A ``[B200]`` issue is
    judged from Scheduled Release B200 builds on the named release branch (or
    ``main``). Daily nightlies are not used as a stand-in for that lane. A
    dual-tagged ``[B200][H100]`` (or H200 / H800 / L4) issue also requires the
    daily CUDA nightly. NPU nightlies are never B200 evidence.
    """
    if not is_b200_issue(issue):
        return nightly
    by_branch = b200_by_branch or {}
    branches = b200_branches_for_issue(issue)

    def snap_for(branch: str) -> PipelineSnapshot:
        snap = by_branch.get(branch)
        if snap is not None:
            return snap
        return _unavailable_b200_snapshot(branch)

    ev: dict[str, PipelineSnapshot] = {}
    if _has_daily_cuda_tag(issue):
        if len(branches) == 1:
            ev["B200"] = snap_for(branches[0])
        else:
            for branch in branches:
                ev[f"B200 ({branch})"] = snap_for(branch)
        if "CUDA" in nightly:
            ev["CUDA"] = nightly["CUDA"]
    elif len(branches) == 1:
        ev["CUDA"] = snap_for(branches[0])
    else:
        for branch in branches:
            ev[f"CUDA ({branch})"] = snap_for(branch)
    if _NPU_RE.search(_issue_blob(issue)) and "NPU" in nightly:
        ev["NPU"] = nightly["NPU"]
    return ev


def _snap_ref(snap: PipelineSnapshot | None) -> str:
    if snap is not None and snap.build_number is not None:
        ref = f"#{snap.build_number}"
        if snap.lane == LANE_B200:
            branch = snap.branch or "main"
            return f"Scheduled Release B200 {ref} ({branch})"
        return ref
    if snap is not None and snap.lane == LANE_B200:
        branch = snap.branch or "main"
        return f"latest Scheduled Release B200 build ({branch})"
    return "latest nightly"


def _missing_results_reason(
    side: str,
    snap: PipelineSnapshot | None,
    issue: dict[str, Any],
) -> str:
    if snap is not None and snap.lane == LANE_B200:
        return (
            f"Latest {side} Scheduled Release B200 results "
            f"on {snap.branch or 'main'} were not loaded."
        )
    if is_b200_issue(issue) and (snap is None or snap.lane == LANE_B200):
        return f"Latest {side} Scheduled Release B200 results were not loaded."
    return f"Latest {side} nightly results were not loaded."


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


def infer_sides(
    issue: dict[str, Any],
    *,
    matched_sides: set[str] | None = None,
    pipeline_keys: set[str] | None = None,
) -> set[str]:
    """Which pipelines must be green before the issue can be closable.

    Label / title / body mentions win. If the issue is silent on CUDA vs
    NPU, fall back to the pipelines where a job name actually matched.

    ``[B200]`` issues require the B200 evidence keys already selected for
    this issue (Scheduled Release B200, plus daily CUDA only when a daily
    CUDA tag such as ``[H100]`` is also present).
    """
    blob = _issue_blob(issue)
    has_npu = bool(_NPU_RE.search(blob))
    has_cuda = bool(_CUDA_RE.search(blob))
    if is_b200_issue(issue):
        # The evidence map already chose B200 (and optional daily CUDA / NPU) keys.
        sides = set(pipeline_keys or ())
        if has_npu:
            sides.add("NPU")
        if sides:
            return sides
        if matched_sides:
            return set(matched_sides)
        return set()
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
    evidence: dict[str, PipelineSnapshot] | None = None,
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
        evidence=dict(evidence or {}),
    )


def evaluate_issue(
    issue: dict[str, Any],
    pipelines: dict[str, PipelineSnapshot],
) -> IssueConfirmation:
    """Confirm one issue against the pipelines provided for it.

    Callers that already know the issue is ``[B200]`` should pass Scheduled
    Release B200 snapshots, not the daily scheduled nightly.
    """
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"
    tests = extract_node_ids(text)

    jobs_by_side: dict[str, list[str]] = {}
    matched_sides: set[str] = set()
    for label, snap in pipelines.items():
        names = match_jobs(text, snap.jobs)
        if names:
            jobs_by_side[label] = names
            matched_sides.add(label)

    sides = tuple(
        sorted(
            infer_sides(
                issue,
                matched_sides=matched_sides,
                pipeline_keys=set(pipelines),
            )
        )
    )

    def out(**kwargs: Any) -> IssueConfirmation:
        kwargs.setdefault("evidence", pipelines)
        return _confirmation(issue, tests=tests, jobs_by_side=jobs_by_side, **kwargs)

    if not sides:
        return out(
            sides=(),
            results={},
            status=STATUS_INSUFFICIENT,
            reason="Could not infer a CUDA or NPU scope, or no distinctive job name was found.",
        )
    missing_reasons = [
        _missing_results_reason(side, pipelines.get(side), issue)
        for side in sides
        if (pipelines.get(side) is None or not pipelines[side].available or not pipelines[side].jobs)
    ]
    if missing_reasons:
        return out(
            sides=sides,
            results={},
            status=STATUS_INSUFFICIENT,
            reason=" ".join(missing_reasons),
        )
    if not any(jobs_by_side.get(side) for side in sides):
        return out(
            sides=sides,
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
            return out(
                sides=sides,
                results=results,
                status=STATUS_INSUFFICIENT,
                reason=_missing_results_reason(side, snap, issue),
            )
        names = jobs_by_side.get(side) or []
        if not names:
            return out(
                sides=sides,
                results=results,
                status=STATUS_INSUFFICIENT,
                reason=f"No matching {side} job was identified.",
            )
        index = _jobs_by_name(snap.jobs)
        failed = _failed_nodes_from_jobs(snap.jobs)
        build_ref = _snap_ref(snap)
        side_fail = False
        for name in names:
            rec = index.get(name)
            kind = job_kind(rec) if rec is not None else "unknown"
            if rec is None:
                return out(
                    sides=sides,
                    results=results,
                    status=STATUS_INSUFFICIENT,
                    reason=f"{side} job {name} was not present on {build_ref}.",
                )
            if kind == "unknown":
                return out(
                    sides=sides,
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
        return out(
            sides=sides,
            results=results,
            status=STATUS_FAILING,
            reason="; ".join(fail_bits) + ".",
            clean_streak=0,
        )

    if tests:
        reasons.append("tracked test node(s) are not in the current failure set")
    return out(
        sides=sides,
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


def _history_for_side(
    row: IssueConfirmation,
    side: str,
    histories: dict[str, list[tuple[int, list[dict[str, Any]]]]],
    b200_histories: dict[str, list[tuple[int, list[dict[str, Any]]]]] | None,
) -> list[tuple[int, list[dict[str, Any]]]]:
    snap = (row.evidence or {}).get(side)
    if snap is not None and snap.lane == LANE_B200:
        by_branch = b200_histories or {}
        return by_branch.get(snap.branch or "main") or by_branch.get(side) or []
    return histories.get(side) or []


def attach_clean_streaks(
    rows: list[IssueConfirmation],
    histories: dict[str, list[tuple[int, list[dict[str, Any]]]]],
    b200_histories: dict[str, list[tuple[int, list[dict[str, Any]]]]] | None = None,
) -> None:
    """Set ``clean_streak`` in place. Closable rows use min across required sides.

    ``[B200]`` sides count consecutive Scheduled Release B200 builds from
    ``b200_histories`` (keyed by branch). Daily nightlies are not mixed in.
    """
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
            hist = _history_for_side(row, side, histories, b200_histories)
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
        branch = "main"
        if build and build.get("branch"):
            branch = str(build.get("branch") or "main")
        out[label] = PipelineSnapshot(
            label=label,
            build_number=number,
            build_url=url,
            jobs=list(jobs or []),
            available=available,
            note=str(note) if note else None,
            org=str(org),
            pipeline=str(pipeline),
            lane=LANE_NIGHTLY,
            branch=branch,
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


def select_scheduled_release_b200_builds(
    builds: list[dict[str, Any]] | None,
    *,
    ceiling: int | None = None,
    exclude: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Newest-first Scheduled Release B200 builds from an already-fetched list.

    Matches the first line of ``message`` only (case-insensitive, ordinary
    whitespace collapsed). Does not require ``source == schedule``. The
    caller is responsible for listing the intended branch; this filter does
    not inspect the branch name.
    """
    out: list[dict[str, Any]] = []
    seen: set[int] = set(exclude or ())
    for build in builds or []:
        if not is_scheduled_release_b200(build):
            continue
        try:
            number = int(build["number"])
        except (KeyError, TypeError, ValueError):
            continue
        if ceiling is not None and number > ceiling:
            continue
        if number in seen:
            continue
        seen.add(number)
        out.append(build)
    return out


def _iter_pipeline_builds(
    *,
    org: str,
    pipeline: str,
    branch: str,
    token: str,
    http_json: Any,
    max_pages: int = _B200_LIST_MAX_PAGES,
    per_page: int = _B200_LIST_PER_PAGE,
) -> list[dict[str, Any]]:
    quoted = urllib.parse.quote(branch, safe="")
    all_builds: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://api.buildkite.com/v2/organizations/{org}/pipelines/{pipeline}"
            f"/builds?branch={quoted}&per_page={per_page}&page={page}"
        )
        batch = http_json(url, token)
        if not isinstance(batch, list) or not batch:
            break
        all_builds.extend(batch)
        if len(batch) < per_page:
            break
    return all_builds


def fetch_scheduled_release_b200_history(
    snap: PipelineSnapshot,
    *,
    token: str | None = None,
    limit: int = HISTORY_LIMIT,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Newest-first Scheduled Release B200 history on ``snap.branch``.

    Starts with the already-loaded snapshot. Older entries include job
    *state* only. Daily scheduled nightlies are never mixed in.
    """
    history: list[tuple[int, list[dict[str, Any]]]] = []
    seen: set[int] = set()
    if snap.available and snap.build_number is not None:
        history.append((snap.build_number, snap.jobs))
        seen.add(snap.build_number)
    tok = (token if token is not None else _bk_token()).strip()
    branch = snap.branch or "main"
    if not tok or not snap.org or not snap.pipeline:
        return history
    try:
        from nightly_job_pytest_table import fetch_nightly_build, http_json
    except ImportError:
        return history
    try:
        builds = _iter_pipeline_builds(
            org=snap.org,
            pipeline=snap.pipeline,
            branch=branch,
            token=tok,
            http_json=http_json,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return history
    selected = select_scheduled_release_b200_builds(
        builds,
        ceiling=snap.build_number,
        exclude=seen,
    )
    for build in selected:
        if len(history) >= limit:
            break
        number = int(build["number"])
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
    return history


def load_b200_snapshots(
    branches: list[str] | tuple[str, ...],
    *,
    token: str | None = None,
    org: str = "vllm",
    pipeline: str = "vllm-omni",
) -> dict[str, PipelineSnapshot]:
    """Load the latest Scheduled Release B200 build + job logs for each branch.

    A missing token, a failed listing, or no matching build yields an
    unavailable snapshot for that branch (the issue is not judged from the
    daily nightly as a fallback).
    """
    tok = (token if token is not None else _bk_token()).strip()
    out: dict[str, PipelineSnapshot] = {}
    unique: list[str] = []
    for branch in branches:
        br = (branch or "main").strip() or "main"
        if br not in unique:
            unique.append(br)
    if not tok:
        note = "BUILDKITE_API_TOKEN / BUILDKITE_TOKEN is not set; Scheduled Release B200 was not loaded."
        return {br: _unavailable_b200_snapshot(br, note=note) for br in unique}
    try:
        from nightly_job_pytest_table import (
            collect_nightly_job_log_analyses,
            fetch_nightly_build,
            http_json,
        )
    except ImportError as exc:
        note = f"Could not import Buildkite helpers ({exc})."
        return {br: _unavailable_b200_snapshot(br, note=note) for br in unique}

    for branch in unique:
        try:
            builds = _iter_pipeline_builds(
                org=org,
                pipeline=pipeline,
                branch=branch,
                token=tok,
                http_json=http_json,
            )
            selected = select_scheduled_release_b200_builds(builds)
            if not selected:
                out[branch] = _unavailable_b200_snapshot(
                    branch,
                    note=(
                        f"No Scheduled Release B200 build found on {branch} "
                        f"within {_B200_LIST_MAX_PAGES * _B200_LIST_PER_PAGE} recent builds."
                    ),
                )
                continue
            number = int(selected[0]["number"])
            raw = fetch_nightly_build(tok, number, org=org, pipeline=pipeline)
            jobs = collect_nightly_job_log_analyses(raw, tok, org=org, pipeline=pipeline)
            out[branch] = PipelineSnapshot(
                label="CUDA",
                build_number=number,
                build_url=f"https://buildkite.com/{org}/{pipeline}/builds/{number}",
                jobs=list(jobs or []),
                available=bool(raw and jobs),
                org=org,
                pipeline=pipeline,
                lane=LANE_B200,
                branch=str(raw.get("branch") or branch),
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            out[branch] = _unavailable_b200_snapshot(
                branch,
                note=f"Failed to load Scheduled Release B200 on {branch}: {type(exc).__name__}.",
            )
    return out


def b200_branches_from_issues(issues: list[dict[str, Any]]) -> list[str]:
    """Unique B200 branches required by ``[B200]`` issues, in first-seen order."""
    branches: list[str] = []
    for issue in issues:
        if not is_b200_issue(issue):
            continue
        for branch in b200_branches_for_issue(issue):
            if branch not in branches:
                branches.append(branch)
    return branches


def collect_confirmations(
    issues: list[dict[str, Any]],
    pipelines: dict[str, PipelineSnapshot],
    histories: dict[str, list[tuple[int, list[dict[str, Any]]]]] | None = None,
    *,
    b200_pipelines: dict[str, PipelineSnapshot] | None = None,
    b200_histories: dict[str, list[tuple[int, list[dict[str, Any]]]]] | None = None,
) -> list[IssueConfirmation]:
    rows = [
        evaluate_issue(issue, evidence_pipelines_for_issue(issue, pipelines, b200_pipelines))
        for issue in issues
    ]
    attach_clean_streaks(rows, histories or {}, b200_histories=b200_histories)
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

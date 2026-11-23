#!/usr/bin/env python3
"""Nightly report Issue section: 待新增 / 待关闭 / 待更新.

Buckets come from today's run (Buildkite nightly, weekly, and local jobs)
plus already-open issues. Closable rows use ``issue_close_confirm`` and are
listed only with positive evidence. Confirm buttons do not call the GitHub
Row buttons stay per-item: 开单确认 opens the prefilled new-issue form, and
关闭确认 / 更新评论 open the existing issue with a ready-to-paste comment.
一键操作 sends the whole set to a local executor (or a one-line command)
that uses GITHUB_TOKEN / GH_TOKEN from the environment. The HTML never
embeds a token.
"""

from __future__ import annotations

import html
import json
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from issue_close_confirm import (
    STATUS_CLOSABLE,
    STATUS_FAILING,
    STATUS_INSUFFICIENT,
    ISSUE_REPO,
    IssueConfirmation,
    PipelineSnapshot,
    job_name_is_distinctive,
    match_jobs,
)

# Keep in sync with nightly_local_log_report.py and
# vllm-omni .github/ISSUE_TEMPLATE/400-bug-report.yml (URL prefill needs `id`).
_ISSUE_NEW = f"https://github.com/{ISSUE_REPO}/issues/new"
_BUG_TEMPLATE = "400-bug-report.yml"
_ENV_FIELD = "current-environment"
_CODE_FIELD = "code-version"
_DESCRIBE_FIELD = "bug-description"
_ENV_PLACEHOLDER = "Your output of `python collect_env.py` here"
_ENV_DEFAULT = (
    "<details>\n"
    "<summary>The output of <code>python collect_env.py</code></summary>\n\n"
    "```text\n"
    f"{_ENV_PLACEHOLDER}\n"
    "```\n\n"
    "</details>"
)
_ENV_CI = _ENV_DEFAULT.replace(_ENV_PLACEHOLDER, "ci env")
_LABELS_CI = ("bug", "ci-failure", "high priority")
_LABELS_LOCAL = ("bug",)
_MAX_ISSUE_URL_LEN = 7800

_JOB_FAIL_STATES = {"failed", "timed_out"}
_WEEKLY_SOURCE_LABELS = {
    "weekly": "Scheduled weekly",
    "noncritical-weekly": "Scheduled Non-Critical weekly",
}

_CLASSIFICATION = {
    "new": (
        "No open issue passed the duplicate thresholds "
        "(node id, exception class, and signature tokens). Listed to file."
    ),
    "recurred": (
        "The same signature matches a closed issue. Listed to file a new issue; "
        "reopen the closed issue instead if it is the same failure."
    ),
    "tracked": (
        "An open issue matches this failure. Update that issue with this run; "
        "do not file a duplicate."
    ),
    "env-local": "Local environment signature (for example an HF cache-lock permission error). Not a repository issue.",
    "unknown": "Issue search did not produce a confident match. Not listed as a new issue.",
}


@dataclass
class IssueContext:
    """Already-fetched issue list and nightly confirmations (no network)."""

    issues: list[dict[str, Any]]
    pipelines: dict[str, PipelineSnapshot]
    confirmations: list[IssueConfirmation]
    issues_loaded: bool
    note: str | None = None


@dataclass
class BareJobFailure:
    """Buildkite job that failed without an extracted pytest node."""

    source: str
    job_name: str
    state: str
    build_url: str = ""
    step_url: str = ""
    log_error: str = ""
    issue_env: str = "ci"
    vllm_version: str = ""
    omni_version: str = ""
    build_commit: str = ""


@dataclass
class IssueItem:
    bucket: str
    description: str
    analysis: str
    links: list[tuple[str, str]]
    action_url: str
    button_label: str
    ready_comment: str = ""
    issue_url: str = ""
    issue_number: int | None = None
    issue_title: str = ""
    labels: list[str] = field(default_factory=list)


@dataclass
class IssueSection:
    summary: str
    to_file: list[IssueItem] = field(default_factory=list)
    to_close: list[IssueItem] = field(default_factory=list)
    to_update: list[IssueItem] = field(default_factory=list)


@dataclass
class _UpdateAcc:
    issue: dict[str, Any]
    analyses: list[str]
    links: list[tuple[str, str]]


def collect_bare_job_failures(
    bk_results: dict[Any, tuple[dict[str, Any] | None, list[dict[str, Any]] | None, Any]] | None,
    weekly_results: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]] | None, Any]] | None,
) -> list[BareJobFailure]:
    """Failed/timed-out jobs that produced no pytest node.

    Skipped, broken, canceled, and not-run jobs are omitted. A job that
    already has failed/error nodes is omitted too — those nodes are handled
    as failure groups.
    """
    out: list[BareJobFailure] = []

    def _consider(
        source: str,
        build: dict[str, Any] | None,
        jobs: list[dict[str, Any]] | None,
        *,
        org: str,
        pipeline: str,
    ) -> None:
        if not build or not jobs:
            return
        number = build.get("number")
        build_url = ""
        if number is not None and org and pipeline:
            build_url = f"https://buildkite.com/{org}/{pipeline}/builds/{int(number)}"
        commit = (build.get("commit") or "")[:12]
        for rec in jobs:
            state = (rec.get("state") or "").strip().lower()
            if state not in _JOB_FAIL_STATES:
                continue
            info = rec.get("info") or {}
            if info.get("failed_nodes") or info.get("error_nodes"):
                continue
            name = str(rec.get("name") or "").strip()
            if not name:
                continue
            versions = rec.get("ci_versions") or {}
            out.append(
                BareJobFailure(
                    source=source,
                    job_name=name,
                    state=state,
                    build_url=build_url,
                    step_url=str(rec.get("step_link") or ""),
                    log_error=str(rec.get("log_error") or ""),
                    issue_env="ci",
                    vllm_version=str(versions.get("vllm") or ""),
                    omni_version=str(versions.get("vllm_omni") or ""),
                    build_commit=str(rec.get("build_commit_short") or commit),
                )
            )

    for target, payload in (bk_results or {}).items():
        build, jobs, _note = payload if payload else (None, None, None)
        label = getattr(target, "label", None) or str(target)
        number = build.get("number") if build else None
        source = f"Buildkite ({label}) nightly" + (f" · #{number}" if number is not None else "")
        _consider(
            source,
            build,
            jobs,
            org=str(getattr(target, "org", "vllm")),
            pipeline=str(getattr(target, "pipeline", "")),
        )
    for kind, payload in (weekly_results or {}).items():
        build, jobs, _note = payload if payload else (None, None, None)
        number = build.get("number") if build else None
        label = _WEEKLY_SOURCE_LABELS.get(str(kind), str(kind))
        source = f"Buildkite (CUDA) {label}" + (f" · #{number}" if number is not None else "")
        _consider(
            source,
            build,
            jobs,
            org="vllm",
            pipeline="vllm-omni",
        )
    return out


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_links(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for label, url in links:
        url = (url or "").strip()
        if not url.startswith(("https://", "http://")) or url in seen:
            continue
        seen.add(url)
        out.append(((label or url).strip(), url))
    return out


def _group_links(group: Any) -> list[tuple[str, str]]:
    raw = list(getattr(group, "links", []) or [])
    return _dedupe_links(raw)


def _group_env(group: Any) -> str:
    envs = list(getattr(group, "issue_envs", []) or [])
    if "ci" in envs:
        return "ci"
    sources = " ".join(getattr(group, "sources", []) or [])
    if "Buildkite" in sources:
        return "ci"
    return "local"


def _issue_title(node: str, reason: str, ctx: str) -> str:
    cleaned = (node or "").replace(" (ERROR)", "").strip() or "unknown failure"
    prefix = "[Bug][NPU]: " if "npu" in (ctx or "").lower() else "[Bug]: "
    reason_bit = f" - {reason}" if reason else ""
    title = f"{prefix}Nightly / CI failed - {cleaned}{reason_bit}"
    return title if len(title) <= 220 else title[:217] + "..."


def _issue_url(issue: dict[str, Any]) -> str:
    number = issue.get("number")
    return issue.get("html_url") or f"https://github.com/{ISSUE_REPO}/issues/{number}"


def _matching_open_issues(job_name: str, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not job_name_is_distinctive(job_name):
        return []
    hits: list[dict[str, Any]] = []
    for issue in issues:
        if (issue.get("state") or "open") != "open":
            continue
        text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"
        if job_name in match_jobs(text, [{"name": job_name}]):
            hits.append(issue)
    return hits


def group_analysis(group: Any) -> str:
    analyses = _unique(list(getattr(group, "analyses", []) or []))
    bits: list[str] = []
    bits.append("\n---\n".join(analyses) if analyses else "No heuristic analysis was extracted from the log.")
    reason = (getattr(group, "primary_reason", "") or "").strip()
    if reason:
        bits.append(f"Log reason: {reason}")
    verdict = getattr(group, "verdict", "unknown") or "unknown"
    bits.append(_CLASSIFICATION.get(verdict, verdict))
    best = getattr(group, "best_issue", None)
    if verdict == "new" and best:
        bits.append(
            f"Possible related issue #{best.get('number')} ({(best.get('title') or '')[:160]}) "
            "did not pass the duplicate thresholds and is not treated as the same issue."
        )
    if verdict == "recurred" and best:
        bits.append(f"Closed issue #{best.get('number')} {_issue_url(best)}".strip())
    nodes = list(getattr(group, "nodes", []) or [])
    if nodes:
        bits.append("Nodes:\n" + "\n".join(nodes))
    sources = list(getattr(group, "sources", []) or [])
    if sources:
        bits.append("Sources:\n" + "\n".join(sources))
    return "\n\n".join(bits)


def file_description(group: Any) -> str:
    nodes = list(getattr(group, "nodes", []) or [])
    steps = list(getattr(group, "steps", []) or [])
    sources = list(getattr(group, "sources", []) or [])
    reason = (getattr(group, "primary_reason", "") or "").strip()
    node = nodes[0] if nodes else (steps[0] if steps else "unknown failure")
    lines = [_issue_title(node, reason, " ".join(sources)), ""]
    if nodes:
        lines.append("Tests:")
        lines.extend(f"- {node_id}" for node_id in nodes)
    if steps:
        lines.append("Jobs:")
        lines.extend(f"- {step}" for step in steps)
    if sources:
        lines.append("Where:")
        lines.extend(f"- {source}" for source in sources)
    best = getattr(group, "best_issue", None)
    if getattr(group, "verdict", "") == "recurred" and best:
        lines.append(f"Matches closed issue #{best.get('number')}: {_issue_url(best)}")
    return "\n".join(lines)


def _code_version_field(env: str, vllm: str, omni: str, commit: str) -> str:
    if env == "ci":
        vllm_line = vllm.strip() if vllm and vllm.strip() else "(not found in Buildkite step log)"
        if omni and omni.strip():
            omni_line = omni.strip()
        elif commit and commit.strip():
            omni_line = commit.strip()
        else:
            omni_line = "(not found in Buildkite step log)"
    else:
        vllm_line, omni_line = "(pending)", "(pending)"
    return "\n".join(
        [
            "<details>",
            "<summary>The commit id or version of vllm</summary>",
            "",
            "```text",
            vllm_line,
            "```",
            "</details>",
            "<details>",
            "<summary>The commit id or version of vllm-omni</summary>",
            "",
            "```text",
            omni_line,
            "```",
            "</details>",
        ]
    )


def _describe_field(
    *,
    node: str,
    reason: str,
    analysis: str,
    excerpt: str,
    is_err: bool,
    build_url: str,
    step_url: str,
    step_name: str,
    extra_nodes: list[str],
) -> str:
    kind = "pytest ERROR" if is_err else "pytest FAILED"
    lines: list[str] = []
    if build_url.strip():
        lines.append(f"**Buildkite build:** {build_url.strip()}")
    if step_url.strip():
        if step_name.strip():
            lines.append(f"**Buildkite step:** [{step_name.strip()}]({step_url.strip()})")
        else:
            lines.append(f"**Buildkite step:** {step_url.strip()}")
    if lines:
        lines.append("")
    lines.append(f"**Failure kind:** {kind}")
    lines.append(f"**Test node:** `{node}`")
    if extra_nodes:
        lines.append("")
        lines.append("**Other nodes in this group:**")
        lines.extend(f"- `{item}`" for item in extra_nodes)
    lines.append("")
    lines.append("**Log reason:**")
    lines.append(reason.strip() if reason and reason.strip() else "(none in report)")
    lines.append("")
    lines.append("**Analysis:**")
    lines.append(analysis.strip() if analysis and analysis.strip() else "(none in report)")
    lines.append("")
    lines.append("**Error log excerpt:**")
    lines.append("")
    lines.append("```text")
    lines.append(excerpt or "(empty)")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append(
        "*Generated from a nightly HTML report. Redact secrets before submitting; "
        "complete the checkboxes on GitHub.*"
    )
    return "\n".join(lines)


def _encode_issue_url(params: list[tuple[str, str]]) -> str:
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f"{_ISSUE_NEW}?{query}"


def build_prefilled_issue_url(
    *,
    node: str,
    reason: str,
    analysis: str,
    excerpt: str,
    env: str,
    ctx: str,
    vllm_version: str = "",
    omni_version: str = "",
    build_commit: str = "",
    build_url: str = "",
    step_url: str = "",
    step_name: str = "",
    extra_nodes: list[str] | None = None,
    is_err: bool = False,
) -> str:
    """Same GitHub bug-template prefill the failure-table Submit issue button built."""
    labels = _LABELS_CI if env == "ci" else _LABELS_LOCAL
    extra = list(extra_nodes or [])

    def _build(excerpt_text: str, analysis_text: str) -> str:
        params = [
            ("template", _BUG_TEMPLATE),
            ("title", _issue_title(node, reason, ctx)),
            ("labels", ",".join(labels)),
            (_ENV_FIELD, _ENV_CI if env == "ci" else _ENV_DEFAULT),
            (_CODE_FIELD, _code_version_field(env, vllm_version, omni_version, build_commit)),
            (
                _DESCRIBE_FIELD,
                _describe_field(
                    node=node,
                    reason=reason,
                    analysis=analysis_text,
                    excerpt=excerpt_text,
                    is_err=is_err,
                    build_url=build_url,
                    step_url=step_url,
                    step_name=step_name,
                    extra_nodes=extra,
                ),
            ),
        ]
        return _encode_issue_url(params)

    excerpt_text = excerpt or ""
    analysis_text = analysis or ""
    url = _build(excerpt_text, analysis_text)
    if len(url) > _MAX_ISSUE_URL_LEN:
        over = len(url) - _MAX_ISSUE_URL_LEN + 220
        keep = max(400, len(excerpt_text) - over)
        excerpt_text = (
            excerpt_text[:keep] + "\n\n...(truncated for GitHub URL length; see HTML report for full log)"
        )
        url = _build(excerpt_text, analysis_text)
    if len(url) > _MAX_ISSUE_URL_LEN and analysis_text:
        over = len(url) - _MAX_ISSUE_URL_LEN + 120
        keep = max(200, len(analysis_text) - over)
        analysis_text = analysis_text[:keep] + "\n\n...(analysis truncated; see HTML report)"
        url = _build(excerpt_text, analysis_text)
    if len(url) > _MAX_ISSUE_URL_LEN:
        url = _build("(excerpt omitted; see HTML report)", analysis_text[:500])
    return url


def _first(values: list[str]) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _file_item_from_group(group: Any) -> IssueItem:
    nodes = list(getattr(group, "nodes", []) or [])
    steps = list(getattr(group, "steps", []) or [])
    node = nodes[0] if nodes else (steps[0] if steps else "unknown failure")
    links = _group_links(group)
    build_url = next((url for label, url in links if label.lower().startswith("build")), "")
    if not build_url and links:
        build_url = links[0][1]
    step_url = next((url for label, url in links if label.lower().startswith("step")), "")
    step_name = steps[0] if steps else ""
    excerpts = list(getattr(group, "excerpts", []) or [])
    versions = list(getattr(group, "vllm_versions", []) or [])
    omni_versions = list(getattr(group, "omni_versions", []) or [])
    commits = list(getattr(group, "build_commits", []) or [])
    analysis = group_analysis(group)
    description = file_description(group)
    env = _group_env(group)
    url = build_prefilled_issue_url(
        node=node,
        reason=(getattr(group, "primary_reason", "") or ""),
        analysis=analysis,
        excerpt=excerpts[0] if excerpts else "",
        env=env,
        ctx=" ".join(getattr(group, "sources", []) or []),
        vllm_version=_first(versions),
        omni_version=_first(omni_versions),
        build_commit=_first(commits),
        build_url=build_url,
        step_url=step_url,
        step_name=step_name,
        extra_nodes=nodes[1:],
        is_err="(ERROR)" in node or (getattr(group, "primary_reason", "") or "").lower().startswith("error"),
    )
    return IssueItem(
        bucket="file",
        description=description,
        analysis=analysis,
        links=links,
        action_url=url,
        button_label="开单确认",
        issue_title=description.splitlines()[0] if description else "",
        labels=list(_LABELS_CI if env == "ci" else _LABELS_LOCAL),
    )


def _file_item_from_bare(job: BareJobFailure) -> IssueItem:
    ctx = job.source
    title = _issue_title(job.job_name, f"Buildkite state {job.state}", ctx)
    description = "\n".join(
        [
            title,
            "",
            f"Job: {job.job_name}",
            f"State: {job.state}",
            f"Where: {job.source}",
            "No pytest failure node was extracted from this job.",
        ]
    )
    analysis = (
        f"Buildkite reports this job as {job.state}. "
        "No per-test failure node or heuristic analysis was extracted, "
        "so this item does not claim a specific test result."
    )
    if job.log_error:
        analysis += f" Log fetch error: {job.log_error[:500]}"
    links: list[tuple[str, str]] = []
    if job.build_url:
        links.append((f"Build · {job.source}", job.build_url))
    if job.step_url:
        links.append((f"Step {job.job_name}", job.step_url))
    url = build_prefilled_issue_url(
        node=job.job_name,
        reason=f"Buildkite job state {job.state}; no pytest node extracted",
        analysis=analysis,
        excerpt=job.log_error[:1500],
        env="ci",
        ctx=ctx,
        vllm_version=job.vllm_version,
        omni_version=job.omni_version,
        build_commit=job.build_commit,
        build_url=job.build_url,
        step_url=job.step_url,
        step_name=job.job_name,
    )
    return IssueItem(
        bucket="file",
        description=description,
        analysis=analysis,
        links=_dedupe_links(links),
        action_url=url,
        button_label="开单确认",
        issue_title=title,
        labels=list(_LABELS_CI),
    )


def _confirmation_links(row: IssueConfirmation, pipelines: dict[str, PipelineSnapshot]) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for side in row.sides:
        snap = pipelines.get(side)
        if snap and snap.build_url:
            label = row.results.get(side) or f"{side} nightly"
            links.append((f"{side} {label}", snap.build_url))
    return _dedupe_links(links)


def _streak_sentence(row: IssueConfirmation) -> str:
    if row.status != STATUS_CLOSABLE:
        return ""
    if row.clean_streak is None:
        return (
            "Consecutive clean scheduled nightlies: n/a "
            "(older history was not loaded; closable is based only on this report's nightly)."
        )
    unit = "scheduled nightly" if row.clean_streak == 1 else "scheduled nightlies"
    return f"Consecutive clean scheduled nightlies: {row.clean_streak} {unit}."


def _close_comment(analysis: str, links: list[tuple[str, str]]) -> str:
    lines = [
        "Nightly report: the previously failing job is green on the relevant pipeline(s) in this report.",
        "",
        analysis.strip(),
        "",
        "Builds:",
    ]
    if links:
        lines.extend(f"- {label}: {url}" for label, url in links)
    else:
        lines.append("- (no build URL was recorded)")
    return "\n".join(lines)


def _update_comment(analysis: str, links: list[tuple[str, str]]) -> str:
    lines = [
        "Nightly report update. This issue is not ready to close.",
        "",
        analysis.strip(),
        "",
        "Builds:",
    ]
    if links:
        lines.extend(f"- {label}: {url}" for label, url in links)
    else:
        lines.append("- (no build URL was recorded)")
    return "\n".join(lines)


def _summary_text(
    *,
    to_file: list[IssueItem],
    to_close: list[IssueItem],
    to_update: list[IssueItem],
    insufficient: int,
    unclassified_groups: int,
    env_local: int,
    weak_jobs: int,
    withheld_jobs: int,
    search_ran: bool,
    group_count: int,
    issues_loaded: bool,
    note: str | None,
) -> str:
    parts = [
        (
            f"待新增 has {len(to_file)} item(s): failures from this report that need a new issue "
            "(no open duplicate passed the match rules)."
        ),
        (
            f"待关闭 has {len(to_close)} item(s): open ci-failure issues whose previously failing "
            "job is fully green on every relevant pipeline in this report, with any tracked test "
            "absent from the current failure set."
        ),
        (
            f"待更新 has {len(to_update)} item(s): open issues that are still failing or have new "
            "evidence from this report and are not ready to close."
        ),
    ]
    if insufficient:
        parts.append(
            f"{insufficient} open ci-failure issue(s) were left off 待关闭 and 待更新 because the "
            "evidence was missing, the job name was too weak, the job was skipped or not run, "
            "or the CUDA vs NPU scope did not match."
        )
    if env_local:
        parts.append(f"{env_local} local-environment failure group(s) were left out of 待新增.")
    if not search_ran and group_count:
        parts.append(
            "GitHub issue search did not run, so node-level failures were not classified as new issues."
        )
    elif unclassified_groups:
        parts.append(
            f"{unclassified_groups} failure group(s) were not filed under 待新增 because issue search "
            "did not return a confident result."
        )
    if weak_jobs:
        parts.append(
            f"{weak_jobs} failed job(s) had no pytest node and a non-distinctive name, so they were not listed."
        )
    if withheld_jobs:
        parts.append(
            f"{withheld_jobs} failed job(s) without a pytest node were not listed under 待新增 "
            "because the open-issue list was unavailable and a duplicate could not be ruled out."
        )
    if not issues_loaded:
        parts.append(
            "The open ci-failure issue list was not loaded, so nothing was marked closable "
            "or matched against existing tickets."
        )
    if note:
        parts.append(note.strip())
    return " ".join(parts)


def assemble_issue_section(
    *,
    groups: list[Any],
    bare_jobs: list[BareJobFailure],
    context: IssueContext,
    search_ran: bool,
) -> IssueSection:
    """Split today's failures and open issues into the three action buckets."""
    open_issues = [issue for issue in context.issues if (issue.get("state") or "open") == "open"]
    issues_by_number = {int(issue["number"]): issue for issue in open_issues if issue.get("number") is not None}
    updates: dict[int, _UpdateAcc] = {}
    to_file: list[IssueItem] = []
    unclassified_groups = 0
    env_local = 0
    weak_jobs = 0
    withheld_jobs = 0

    def _add_update(issue: dict[str, Any], analysis: str, links: list[tuple[str, str]]) -> None:
        number = issue.get("number")
        if number is None:
            return
        number = int(number)
        acc = updates.get(number)
        text = (analysis or "").strip()
        if acc is None:
            updates[number] = _UpdateAcc(issue=issue, analyses=[text] if text else [], links=list(links))
            return
        if text and text not in acc.analyses:
            acc.analyses.append(text)
        acc.links.extend(links)

    for group in groups:
        verdict = getattr(group, "verdict", "unknown") or "unknown"
        if verdict == "env-local":
            env_local += 1
            continue
        if not search_ran or verdict == "unknown":
            # Search did not confidently classify this group. A distinctive
            # job-name hit on an open issue is still enough to update that
            # issue; it is not enough to file a new one.
            if context.issues_loaded:
                matched: list[dict[str, Any]] = []
                for step in getattr(group, "steps", []) or []:
                    for issue in _matching_open_issues(step, open_issues):
                        if all(issue.get("number") != prev.get("number") for prev in matched):
                            matched.append(issue)
                if matched:
                    analysis = group_analysis(group)
                    links = _group_links(group)
                    for issue in matched:
                        _add_update(issue, analysis, links)
                    continue
            unclassified_groups += 1
            continue
        if verdict == "tracked":
            best = getattr(group, "best_issue", None)
            if best and best.get("state") == "open" and best.get("number") is not None:
                issue = issues_by_number.get(int(best["number"]), best)
                _add_update(issue, group_analysis(group), _group_links(group))
            else:
                unclassified_groups += 1
            continue
        if verdict in {"new", "recurred"}:
            to_file.append(_file_item_from_group(group))
            continue
        unclassified_groups += 1

    covered_steps = {step for group in groups for step in (getattr(group, "steps", []) or [])}
    for job in bare_jobs:
        if job.job_name in covered_steps:
            continue
        hits = _matching_open_issues(job.job_name, open_issues) if context.issues_loaded else []
        links = _dedupe_links(
            [
                *( [(f"Build · {job.source}", job.build_url)] if job.build_url else [] ),
                *( [(f"Step {job.job_name}", job.step_url)] if job.step_url else [] ),
            ]
        )
        analysis = (
            f"{job.source}: job {job.job_name} finished {job.state} with no pytest node extracted."
        )
        if job.log_error:
            analysis += f" Log fetch error: {job.log_error[:500]}"
        if hits:
            for issue in hits:
                _add_update(issue, analysis, links)
            continue
        if not context.issues_loaded:
            withheld_jobs += 1
            continue
        if not job_name_is_distinctive(job.job_name):
            weak_jobs += 1
            continue
        to_file.append(_file_item_from_bare(job))

    blocked = set(updates)
    to_close: list[IssueItem] = []
    insufficient = 0
    for row in context.confirmations:
        number = row.issue.get("number")
        links = _confirmation_links(row, context.pipelines)
        if row.status == STATUS_INSUFFICIENT:
            insufficient += 1
            continue
        if row.status == STATUS_CLOSABLE and number not in blocked:
            streak = _streak_sentence(row)
            analysis = row.reason if not streak else f"{row.reason}\n\n{streak}"
            to_close.append(
                IssueItem(
                    bucket="close",
                    description="",
                    analysis=analysis,
                    links=links,
                    action_url=_issue_url(row.issue),
                    button_label="关闭确认",
                    ready_comment=_close_comment(analysis, links),
                    issue_url=_issue_url(row.issue),
                    issue_number=int(number) if number is not None else None,
                    issue_title=(row.issue.get("title") or "").strip(),
                )
            )
            continue
        if row.status == STATUS_CLOSABLE and number in blocked:
            note = (
                f"Not closable despite a green nightly reading ({row.reason}) "
                "because a failure in this report still matches the issue."
            )
            _add_update(row.issue, note, links)
            continue
        if row.status == STATUS_FAILING:
            _add_update(row.issue, row.reason, links)

    to_update: list[IssueItem] = []
    for number in sorted(updates):
        acc = updates[number]
        analysis = "\n\n".join(bit for bit in acc.analyses if bit)
        links = _dedupe_links(acc.links)
        to_update.append(
            IssueItem(
                bucket="update",
                description="",
                analysis=analysis,
                links=links,
                action_url=_issue_url(acc.issue),
                button_label="更新评论",
                ready_comment=_update_comment(analysis, links),
                issue_url=_issue_url(acc.issue),
                issue_number=number,
                issue_title=(acc.issue.get("title") or "").strip(),
            )
        )

    summary = _summary_text(
        to_file=to_file,
        to_close=to_close,
        to_update=to_update,
        insufficient=insufficient,
        unclassified_groups=unclassified_groups,
        env_local=env_local,
        weak_jobs=weak_jobs,
        withheld_jobs=withheld_jobs,
        search_ran=search_ran,
        group_count=len(groups),
        issues_loaded=context.issues_loaded,
        note=context.note,
    )
    return IssueSection(summary=summary, to_file=to_file, to_close=to_close, to_update=to_update)


ISSUE_BULK_PORT = 8931
_APPLY_SCRIPT = Path(__file__).resolve().parent / "apply_nightly_issue_actions.py"


def _api_body(item: IssueItem) -> str:
    lines = [item.description.strip(), "", "Analysis:", item.analysis.strip() or "(none)", "", "Builds:"]
    if item.links:
        lines.extend(f"- {label}: {url}" for label, url in item.links)
    else:
        lines.append("- (no build URL was recorded)")
    lines.extend(["", "Filed from the nightly report 一键操作 action."])
    return "\n".join(lines)


def bulk_payload(section: IssueSection) -> dict[str, Any]:
    """JSON the local executor applies. No tokens."""
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(section.to_file):
        actions.append(
            {
                "id": f"file-{index}",
                "op": "create",
                "title": item.issue_title or (item.description.splitlines() or ["Nightly failure"])[0],
                "body": _api_body(item),
                "labels": list(item.labels),
            }
        )
    for item in section.to_close:
        actions.append(
            {
                "id": f"close-{item.issue_number}",
                "op": "close",
                "issue_number": item.issue_number,
                "comment": item.ready_comment,
            }
        )
    for item in section.to_update:
        actions.append(
            {
                "id": f"update-{item.issue_number}",
                "op": "comment",
                "issue_number": item.issue_number,
                "comment": item.ready_comment,
            }
        )
    return {"repo": ISSUE_REPO, "actions": actions}


def write_issue_actions(path: Path, section: IssueSection) -> dict[str, Any]:
    payload = bulk_payload(section)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def issue_apply_commands(payload_path: Path | None) -> tuple[str, str]:
    """(one-shot apply command, localhost executor command). Neither includes a token."""
    script = str(_APPLY_SCRIPT)
    serve = f"python3 {script} --serve"
    target = str(payload_path) if payload_path is not None else "<report>.issue-actions.json"
    apply = f"python3 {script} --payload {target} --apply"
    return apply, serve


_ISSUE_SECTION_CSS = """
.issue-section-root { margin: 0; }
.issue-section-intro, .issue-action-note, .issue-bucket-empty {
  color: var(--dashboard-soft-text, #475569);
  font-size: 0.92rem;
  line-height: 1.45;
}
.issue-summary { margin: 0 0 1rem; }
.issue-summary h3 { margin: 0 0 0.4rem; font-size: 1.05rem; }
.issue-summary-prose { margin: 0; }
.issue-bulk {
  margin: 0 0 1rem;
  padding: 0.75rem 1rem;
  border: 1px solid var(--omni-divider, #e5e7eb);
  border-radius: 12px;
  background: var(--dashboard-card-tint, #f6f8fa);
}
.issue-bulk-command, .issue-copy, .issue-ready-comment {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.82rem;
  margin: 0.45rem 0 0;
  padding: 0.55rem 0.7rem;
  background: var(--dashboard-panel-bg, #fff);
  border-radius: 8px;
}
.issue-bucket-card {
  border: 1px solid var(--omni-divider, #e5e7eb);
  border-radius: 12px;
  background: var(--dashboard-panel-bg, #fff);
  margin: 0 0 0.75rem;
  overflow: hidden;
}
.issue-bucket-card-summary {
  cursor: pointer;
  padding: 0.8rem 1rem;
  font-weight: 750;
  font-size: 1.02rem;
}
.issue-bucket-card-body { padding: 0 1rem 1rem; }
.issue-table { width: 100%; border-collapse: collapse; }
.issue-table th, .issue-table td {
  border-bottom: 1px solid var(--omni-divider, #e5e7eb);
  text-align: left;
  vertical-align: top;
  padding: 0.55rem 0.6rem;
}
.issue-table th { font-size: 0.78rem; color: var(--dashboard-soft-text, #475569); }
.issue-table .issue-copy { max-height: 9rem; overflow: auto; margin: 0; }
.issue-builds { margin: 0; padding-left: 1.1rem; }
.issue-table-empty td { color: var(--dashboard-soft-text, #64748b); font-style: italic; }
.issue-confirm { white-space: nowrap; }
a.btn-github-issue, button.btn-issue-bulk { text-decoration: none; }
button.btn-issue-bulk { margin: 0 0 0.35rem; }
button.btn-issue-bulk:disabled { opacity: 0.55; cursor: not-allowed; }
.issue-action-status { min-height: 1.1em; margin: 0.35rem 0 0; font-size: 0.8rem; }
"""


def _pre_block(text: str, css: str) -> str:
    return f'<pre class="{css}">{html.escape(text or "")}</pre>'


def _links_html(links: list[tuple[str, str]]) -> str:
    items: list[str] = []
    for label, url in links:
        if not url.startswith(("https://", "http://")):
            continue
        href = html.escape(url, quote=True)
        items.append(f'<li><a href="{href}">{html.escape(label)}</a></li>')
    if not items:
        return '<p class="issue-bucket-empty">No build URL was recorded for this item.</p>'
    return '<ul class="issue-builds">' + "".join(items) + "</ul>"


def _identity_html(item: IssueItem) -> str:
    if item.bucket == "file":
        title = item.issue_title or (item.description.splitlines() or ["(no title)"])[0]
        rest = "\n".join(item.description.splitlines()[1:]).strip()
        extra = _pre_block(rest, "issue-copy") if rest else ""
        return f'<div class="issue-identity"><strong>{html.escape(title)}</strong>{extra}</div>'
    title = item.issue_title or "(no title)"
    number = f"#{item.issue_number}" if item.issue_number is not None else "issue"
    href = html.escape(item.issue_url or item.action_url, quote=True)
    return f'<div class="issue-identity"><a href="{href}">{html.escape(number)}</a> {html.escape(title)}</div>'


def _confirm_html(item: IssueItem, ready_id: str) -> str:
    href = html.escape(item.action_url, quote=True)
    label = html.escape(item.button_label)
    if item.bucket == "file":
        css = "btn-github-issue btn-issue-confirm-file"
        return (
            f'<a class="{css}" href="{href}" target="_blank" rel="noopener noreferrer">{label}</a>'
        )
    css = (
        "btn-github-issue btn-issue-confirm-close"
        if item.bucket == "close"
        else "btn-github-issue btn-issue-confirm-update"
    )
    comment = html.escape(item.ready_comment or "")
    return (
        f'<pre id="{ready_id}" class="issue-ready-comment" hidden>{comment}</pre>'
        f'<a class="{css}" href="{href}" target="_blank" rel="noopener noreferrer" '
        f'data-ready-comment-id="{ready_id}">{label}</a>'
        f'<p id="{ready_id}-status" class="issue-action-status"></p>'
    )


def _table_html(items: list[IssueItem], empty: str, *, id_prefix: str) -> str:
    head = (
        '<table class="issue-table"><thead><tr>'
        "<th>Issue</th><th>Analysis</th><th>Builds</th><th>Confirm</th>"
        "</tr></thead><tbody>"
    )
    if not items:
        return (
            head
            + f'<tr class="issue-table-empty"><td colspan="4">{html.escape(empty)}</td></tr>'
            + "</tbody></table>"
        )
    rows: list[str] = []
    for index, item in enumerate(items, start=1):
        rows.append(
            "<tr data-issue-row=\"1\">"
            f"<td>{_identity_html(item)}</td>"
            f"<td>{_pre_block(item.analysis, 'issue-copy')}</td>"
            f"<td>{_links_html(item.links)}</td>"
            f'<td class="issue-confirm">{_confirm_html(item, f"{id_prefix}-{index}")}</td>'
            "</tr>"
        )
    return head + "".join(rows) + "</tbody></table>"


def _card_html(title: str, items: list[IssueItem], empty: str, note: str, *, id_prefix: str) -> str:
    return (
        f'<details class="issue-bucket-card" id="{html.escape(id_prefix, quote=True)}">'
        f'<summary class="issue-bucket-card-summary">{html.escape(title)}</summary>'
        '<div class="issue-bucket-card-body">'
        f'<p class="issue-action-note">{html.escape(note)}</p>'
        f"{_table_html(items, empty, id_prefix=id_prefix)}"
        "</div></details>"
    )


def _payload_script(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'<script type="application/json" id="issue-bulk-payload">{raw}</script>'


def _bulk_client_script() -> str:
    return f"""
<script>
(function () {{
  var btn = document.getElementById("issue-bulk-apply");
  var payloadEl = document.getElementById("issue-bulk-payload");
  var statusEl = document.getElementById("issue-bulk-status");
  var commandEl = document.getElementById("issue-bulk-command");
  if (!btn || !payloadEl) return;
  var endpoint = "http://127.0.0.1:{ISSUE_BULK_PORT}/apply";
  function setStatus(text) {{ if (statusEl) statusEl.textContent = text; }}
  function copyCommand() {{
    var text = commandEl ? commandEl.textContent || "" : "";
    if (!text || !navigator.clipboard || !window.isSecureContext) return;
    navigator.clipboard.writeText(text).catch(function () {{}});
  }}
  btn.addEventListener("click", function (ev) {{
    ev.preventDefault();
    ev.stopPropagation();
    if (btn.disabled) return;
    setStatus("正在把待新增、待关闭、待更新一次提交…");
    var body = payloadEl.textContent || "";
    fetch(endpoint, {{
      method: "POST",
      mode: "cors",
      headers: {{"Content-Type": "text/plain;charset=UTF-8"}},
      body: body
    }}).then(function (resp) {{
      return resp.text().then(function (text) {{
        var data = null;
        try {{ data = JSON.parse(text); }} catch (e) {{ data = null; }}
        if (!resp.ok || !data || data.ok === false) {{
          var err = (data && data.error) ? data.error : ("HTTP " + resp.status);
          throw new Error(err);
        }}
        setStatus(
          "一键操作已执行：提单 " + (data.created || 0)
          + "，关闭 " + (data.closed || 0)
          + "，评论 " + (data.commented || 0)
          + "，跳过 " + (data.skipped || 0)
          + "，失败 " + (data.failed || 0) + "。"
        );
      }});
    }}).catch(function (err) {{
      copyCommand();
      var why = (err && err.message) ? err.message : "未连上本机执行器";
      setStatus(
        "一键操作没有改 GitHub（" + why + "）。静态页不能保存 token。"
        + "请在终端执行下方命令（GITHUB_TOKEN 或 GH_TOKEN 只放在环境变量里）。"
        + "先运行 --serve 后再点一次，或直接运行 --payload … --apply，两条都会处理全部三张表。"
      );
    }});
  }});
}})();
</script>
"""


def render_issue_section_body(
    section: IssueSection,
    *,
    apply_command: str | None = None,
    serve_command: str | None = None,
) -> str:
    """Inner HTML for the Issue part. Bucket titles are collapsed cards around tables."""
    apply_cmd, serve_cmd = issue_apply_commands(None)
    apply_command = apply_command or apply_cmd
    serve_command = serve_command or serve_cmd
    payload = bulk_payload(section)
    disabled = "" if payload["actions"] else " disabled"
    summary = (
        '<section class="issue-summary" id="issue-summary">'
        "<h3>Summary</h3>"
        f'<p class="issue-summary-prose">{html.escape(section.summary)}</p>'
        "</section>"
    )
    bulk = (
        '<div class="issue-bulk" id="issue-bulk">'
        f'<button type="button" class="btn-github-issue btn-issue-bulk" id="issue-bulk-apply"{disabled}>一键操作</button>'
        '<p class="issue-action-note">一键操作一次处理三张表里的全部待处理项：为待新增开 issue，'
        "关闭待关闭，并给待更新发准备好的评论。打开页面时不会自动执行。"
        "本机执行器在跑时，这一下会直接调用 GitHub；否则使用下面这条命令，token 只从环境变量读取。</p>"
        f'<p class="issue-action-note">本机执行器：<code>{html.escape(serve_command)}</code></p>'
        f'<pre id="issue-bulk-command" class="issue-bulk-command">{html.escape(apply_command)}</pre>'
        '<p id="issue-bulk-status" class="issue-action-status"></p>'
        "</div>"
    )
    to_file = _card_html(
        "待新增",
        section.to_file,
        "No failures need a new issue.",
        "每行一个待开 issue，行内只有开单确认。它只打开这一条的预填表单。",
        id_prefix="issue-to-file",
    )
    to_close = _card_html(
        "待关闭",
        section.to_close,
        "No open issue is closable from this report.",
        "每行一个可关闭 issue，行内只有关闭确认。它打开该 issue 并复制关闭说明，不会直接关闭。",
        id_prefix="issue-to-close",
    )
    to_update = _card_html(
        "待更新",
        section.to_update,
        "No open issue needs an update from this report.",
        "每行一个待更新 issue，行内只有更新评论。它打开该 issue 并复制评论，不会直接发表。",
        id_prefix="issue-to-update",
    )
    return (
        '<div class="issue-section-root">'
        '<p class="issue-section-intro">Today\'s run plus already-open issues. '
        "An issue is listed under 待关闭 only when the previously failing job is fully green "
        "on every relevant pipeline and any tracked test is absent from the current failures. "
        "Missing data, a skipped or not-run job, a weak match, or a CUDA vs NPU mismatch is not closable.</p>"
        f"{summary}{bulk}{to_file}{to_close}{to_update}"
        f"{_payload_script(payload)}"
        f"{_bulk_client_script()}"
        "</div>"
    )


def issue_section_stylesheet() -> str:
    return _ISSUE_SECTION_CSS

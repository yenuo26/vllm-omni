#!/usr/bin/env python3
"""Issue section buckets, confirm links, and nightly HTML layout. No network."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.parse
from html import unescape
from pathlib import Path
from types import SimpleNamespace

from issue_close_confirm import (
    STATUS_CLOSABLE,
    evaluate_issue,
)
from issue_section import (
    BareJobFailure,
    IssueContext,
    assemble_issue_section,
    build_prefilled_issue_url,
    collect_bare_job_failures,
    render_issue_section_body,
)
from test_issue_close_confirm import _issue, _job, _snap


def _group(**kwargs):
    base = dict(
        verdict="unknown",
        nodes=["tests/e2e/test_wan.py::test_ssim"],
        sources=["Buildkite (CUDA) nightly · #20013"],
        steps=["Diffusion X2V Wan2.2 Accuracy Test"],
        primary_reason="AssertionError: boom",
        reasons=["AssertionError: boom"],
        analyses=["heuristic: assertion failed"],
        excerpts=["E AssertionError: boom"],
        links=[("Build · CUDA nightly", "https://buildkite.com/vllm/vllm-omni/builds/20013")],
        issue_envs=["ci"],
        vllm_versions=["0.1"],
        omni_versions=["abc"],
        build_commits=["abc"],
        best_issue=None,
        stem="test_wan",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _ctx(issues, pipelines, confirmations, *, loaded=True, note=None) -> IssueContext:
    return IssueContext(
        issues=issues,
        pipelines=pipelines,
        confirmations=confirmations,
        issues_loaded=loaded,
        note=note,
    )


class BucketTests(unittest.TestCase):
    def test_new_group_is_file_not_close(self):
        section = assemble_issue_section(
            groups=[_group(verdict="new")],
            bare_jobs=[],
            context=_ctx([], {}, []),
            search_ran=True,
        )
        self.assertEqual(len(section.to_file), 1)
        self.assertEqual(section.to_close, [])
        self.assertEqual(section.to_update, [])
        item = section.to_file[0]
        self.assertIn("tests/e2e/test_wan.py::test_ssim", item.description)
        self.assertIn("heuristic: assertion failed", item.analysis)
        self.assertEqual(item.links[0][1], "https://buildkite.com/vllm/vllm-omni/builds/20013")
        self.assertEqual(item.button_label, "开单确认")
        self.assertIn("/issues/new?", item.action_url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(item.action_url).query)
        self.assertEqual(query["template"], ["400-bug-report.yml"])
        self.assertIn("ci-failure", query["labels"][0])
        self.assertIn("test_ssim", query["title"][0])

    def test_unknown_and_env_local_are_not_filed(self):
        section = assemble_issue_section(
            groups=[_group(verdict="unknown"), _group(verdict="env-local", stem="env")],
            bare_jobs=[],
            context=_ctx([], {}, []),
            search_ran=True,
        )
        self.assertEqual(section.to_file, [])
        self.assertEqual(section.to_close, [])
        self.assertEqual(section.to_update, [])
        self.assertIn("local-environment", section.summary)
        self.assertIn("not filed under 待新增", section.summary)

    def test_search_not_run_does_not_file_node_failures(self):
        section = assemble_issue_section(
            groups=[_group(verdict="unknown")],
            bare_jobs=[],
            context=_ctx([], {}, []),
            search_ran=False,
        )
        self.assertEqual(section.to_file, [])
        self.assertIn("issue search did not run", section.summary.lower())

    def test_recurred_is_filed_and_names_the_closed_issue(self):
        closed = {
            "number": 9,
            "title": "old failure",
            "html_url": "https://github.com/vllm-project/vllm-omni/issues/9",
            "state": "closed",
        }
        section = assemble_issue_section(
            groups=[_group(verdict="recurred", best_issue=closed)],
            bare_jobs=[],
            context=_ctx([], {}, []),
            search_ran=True,
        )
        self.assertEqual(len(section.to_file), 1)
        self.assertIn("#9", section.to_file[0].description)
        self.assertIn("issues/9", section.to_file[0].description)

    def test_tracked_open_issue_is_update_not_file(self):
        issue = _issue(21, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")
        section = assemble_issue_section(
            groups=[_group(verdict="tracked", best_issue=issue)],
            bare_jobs=[],
            context=_ctx([issue], {}, []),
            search_ran=True,
        )
        self.assertEqual(section.to_file, [])
        self.assertEqual(len(section.to_update), 1)
        self.assertEqual(section.to_update[0].button_label, "更新评论")
        self.assertEqual(section.to_update[0].issue_number, 21)
        self.assertIn("issues/21", section.to_update[0].action_url)
        self.assertIn("https://buildkite.com/vllm/vllm-omni/builds/20013", section.to_update[0].ready_comment)
        self.assertNotIn("api.github.com", section.to_update[0].action_url)

    def test_closable_only_with_positive_evidence(self):
        issue = _issue(
            10,
            "Diffusion X2V Wan2.2 Accuracy Test failed on H100",
            "tests/e2e/test_wan.py::test_ssim",
        )
        pipelines = {"CUDA": _snap("CUDA", [_job("Diffusion X2V Wan2.2 Accuracy Test")], number=20013)}
        row = evaluate_issue(issue, pipelines)
        self.assertEqual(row.status, STATUS_CLOSABLE)
        section = assemble_issue_section(
            groups=[],
            bare_jobs=[],
            context=_ctx([issue], pipelines, [row]),
            search_ran=True,
        )
        self.assertEqual(len(section.to_close), 1)
        self.assertEqual(section.to_update, [])
        self.assertEqual(section.to_close[0].button_label, "关闭确认")
        self.assertEqual(
            section.to_close[0].action_url,
            "https://github.com/vllm-project/vllm-omni/issues/10",
        )
        self.assertIn("fully green on #20013", section.to_close[0].analysis)
        self.assertTrue(any("20013" in url for _label, url in section.to_close[0].links))

    def test_still_failing_is_update_and_insufficient_is_omitted(self):
        failing = _issue(11, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")
        weak = _issue(17, "something is broken on H100")
        pipelines = {
            "CUDA": _snap(
                "CUDA",
                [_job("Diffusion X2V Wan2.2 Accuracy Test", state="failed", failed=["a.py::t"])],
                number=5,
            )
        }
        rows = [evaluate_issue(failing, pipelines), evaluate_issue(weak, pipelines)]
        section = assemble_issue_section(
            groups=[],
            bare_jobs=[],
            context=_ctx([failing, weak], pipelines, rows),
            search_ran=True,
        )
        self.assertEqual(section.to_close, [])
        self.assertEqual([item.issue_number for item in section.to_update], [11])
        self.assertIn("1 open ci-failure issue", section.summary)

    def test_npu_failure_is_not_closed_because_cuda_is_green(self):
        issue = _issue(13, "Omni Function Test NPU failed")
        pipelines = {
            "CUDA": _snap("CUDA", [_job("Omni Function Test NPU")], number=1),
            "NPU": _snap(
                "NPU",
                [_job("Omni Function Test NPU", state="failed", failed=["a.py::t"])],
                number=2,
            ),
        }
        row = evaluate_issue(issue, pipelines)
        section = assemble_issue_section(
            groups=[],
            bare_jobs=[],
            context=_ctx([issue], pipelines, [row]),
            search_ran=True,
        )
        self.assertEqual(section.to_close, [])
        self.assertEqual(section.to_update[0].issue_number, 13)

    def test_tracked_failure_demotes_a_green_reading(self):
        issue = _issue(10, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")
        pipelines = {"CUDA": _snap("CUDA", [_job("Diffusion X2V Wan2.2 Accuracy Test")], number=20013)}
        row = evaluate_issue(issue, pipelines)
        self.assertEqual(row.status, STATUS_CLOSABLE)
        section = assemble_issue_section(
            groups=[_group(verdict="tracked", best_issue=issue)],
            bare_jobs=[],
            context=_ctx([issue], pipelines, [row]),
            search_ran=True,
        )
        self.assertEqual(section.to_close, [])
        self.assertEqual(section.to_update[0].issue_number, 10)
        self.assertIn("Not closable", section.to_update[0].analysis)

    def test_bare_job_rules(self):
        distinctive = BareJobFailure(
            source="Buildkite (CUDA) nightly · #9",
            job_name="Diffusion X2V Wan2.2 Accuracy Test",
            state="failed",
            build_url="https://buildkite.com/vllm/vllm-omni/builds/9",
            step_url="https://buildkite.com/vllm/vllm-omni/builds/9#step",
        )
        weak = BareJobFailure(source="Buildkite (CUDA) nightly · #9", job_name="CI", state="failed")
        skipped = {
            "weekly": (
                {"number": 3, "message": "weekly"},
                [{"name": "Diffusion X2V Wan2.2 Accuracy Test", "state": "skipped", "info": None}],
                None,
            )
        }
        self.assertEqual(collect_bare_job_failures(None, skipped), [])
        filed = assemble_issue_section(
            groups=[],
            bare_jobs=[distinctive, weak],
            context=_ctx([], {}, []),
            search_ran=True,
        )
        self.assertEqual(len(filed.to_file), 1)
        self.assertIn("no pytest", filed.to_file[0].description.lower())
        self.assertIn("non-distinctive", filed.summary)
        withheld = assemble_issue_section(
            groups=[],
            bare_jobs=[distinctive],
            context=_ctx([], {}, [], loaded=False, note="GitHub issue listing failed."),
            search_ran=True,
        )
        self.assertEqual(withheld.to_file, [])
        self.assertEqual(withheld.to_close, [])
        self.assertIn("not loaded", withheld.summary)

    def test_bare_job_matching_open_issue_updates_instead_of_filing(self):
        issue = _issue(30, "Diffusion X2V Wan2.2 Accuracy Test flaked")
        job = BareJobFailure(
            source="Buildkite (CUDA) Scheduled weekly · #4",
            job_name="Diffusion X2V Wan2.2 Accuracy Test",
            state="timed_out",
            build_url="https://buildkite.com/vllm/vllm-omni/builds/4",
        )
        section = assemble_issue_section(
            groups=[],
            bare_jobs=[job],
            context=_ctx([issue], {}, []),
            search_ran=True,
        )
        self.assertEqual(section.to_file, [])
        self.assertEqual(section.to_update[0].issue_number, 30)
        self.assertIn("builds/4", section.to_update[0].ready_comment)

    def test_render_has_four_subsections_and_confirm_links(self):
        issue = _issue(10, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")
        pipelines = {"CUDA": _snap("CUDA", [_job("Diffusion X2V Wan2.2 Accuracy Test")], number=20013)}
        row = evaluate_issue(issue, pipelines)
        section = assemble_issue_section(
            groups=[_group(verdict="new")],
            bare_jobs=[],
            context=_ctx([issue], pipelines, [row]),
            search_ran=True,
        )
        # The new group is a different failure mode than the green job, so both
        # a file item and a close item are present. Add an update via a second pass
        # by including a failing issue too.
        failing = _issue(11, "Omni Function Test Extra Long Name failed on H100")
        pipelines["CUDA"].jobs.append(_job("Omni Function Test Extra Long Name", state="failed", failed=["a.py::t"]))
        rows = [row, evaluate_issue(failing, pipelines)]
        section = assemble_issue_section(
            groups=[_group(verdict="new")],
            bare_jobs=[],
            context=_ctx([issue, failing], pipelines, rows),
            search_ran=True,
        )
        html = render_issue_section_body(section)
        self.assertIn("<h3>Summary</h3>", html)
        file_at = html.index('<summary class="issue-bucket-card-summary">待新增</summary>')
        close_at = html.index('<summary class="issue-bucket-card-summary">待关闭</summary>')
        update_at = html.index('<summary class="issue-bucket-card-summary">待更新</summary>')
        self.assertLess(html.index("一键操作"), file_at)
        self.assertLess(file_at, close_at)
        self.assertLess(close_at, update_at)
        self.assertNotIn('class="issue-bucket-card" open', html)
        self.assertEqual(html.count('data-issue-row="1"'), 3)
        self.assertEqual(html.count(">开单确认</a>"), 1)
        self.assertEqual(html.count(">关闭确认</a>"), 1)
        self.assertEqual(html.count(">更新评论</a>"), 1)
        self.assertIn("<th>Issue</th><th>Analysis</th><th>Builds</th><th>Confirm</th>", html)
        self.assertNotIn('id="issue-bulk-apply" disabled', html)
        self.assertIn('class="btn-github-issue btn-issue-confirm-file"', html)
        self.assertIn('class="btn-github-issue btn-issue-confirm-close"', html)
        self.assertIn('class="btn-github-issue btn-issue-confirm-update"', html)
        self.assertNotIn("api.github.com", html)
        self.assertIn("127.0.0.1:8931/apply", html)
        self.assertNotIn("GITHUB_TOKEN=", html)
        self.assertNotIn("Issue Alert", html)
        self.assertNotIn("Issue close confirmation", html)
        self.assertNotIn(">Ready to paste<", html)
        file_href = unescape(html.split('btn-issue-confirm-file" href="', 1)[1].split('"', 1)[0])
        self.assertTrue(file_href.startswith("https://github.com/vllm-project/vllm-omni/issues/new?"))
        close_href = unescape(html.split('btn-issue-confirm-close" href="', 1)[1].split('"', 1)[0])
        self.assertEqual(close_href, "https://github.com/vllm-project/vllm-omni/issues/10")

    def test_prefilled_url_matches_bug_template_fields(self):
        url = build_prefilled_issue_url(
            node="tests/test_mod.py::test_bar",
            reason="AssertionError: boom",
            analysis="looks like a product bug",
            excerpt="E boom",
            env="local",
            ctx="Local nightly_jobs",
        )
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(query["labels"], ["bug"])
        self.assertIn("bug-description", query)
        self.assertIn("current-environment", query)
        self.assertIn("code-version", query)
        self.assertIn("looks like a product bug", query["bug-description"][0])


class NightlyLayoutTests(unittest.TestCase):
    def test_report_is_three_parts_and_moves_filing_out_of_local(self):
        from nightly_local_log_report import (
            CUDA_TARGET,
            NPU_TARGET,
            _render_buildkite_section_html,
            _render_failures_table_html,
            emit_report_html,
        )

        info = {
            "failed_nodes": ["tests/test_mod.py::test_bar"],
            "error_nodes": [],
            "failed_reasons": {"tests/test_mod.py::test_bar": "AssertionError: boom"},
            "error_reasons": {},
            "failure_excerpts": {"tests/test_mod.py::test_bar": "E boom"},
            "error_excerpts": {},
            "failure_analyses": {"tests/test_mod.py::test_bar": "assertion"},
            "error_analyses": {},
            "summary": "1 failed in 0.01s",
        }
        table = _render_failures_table_html(info, report_context="Local nightly_jobs · job: sample")
        self.assertIn("View full log", table)
        self.assertNotIn("Submit issue", table)
        self.assertNotIn("提单确认", table)
        self.assertNotIn("开单确认", table)
        self.assertNotIn("btn-github-issue", table)
        self.assertNotIn("GitHub Issue", table)

        bk_html = _render_buildkite_section_html(
            {"number": 10, "state": "failed", "message": "Scheduled nightly", "commit": "abc"},
            [{
                "name": "Diffusion X2V Wan2.2 Accuracy Test",
                "state": "failed",
                "info": info,
                "step_link": "https://buildkite.com/vllm/vllm-omni/builds/10#aaa",
                "log_error": None,
            }],
            note=None,
            target=CUDA_TARGET,
        )
        self.assertIn("Summary (per-job execution)", bk_html)
        self.assertNotIn("Failure analysis", bk_html)
        self.assertNotIn("View full log", bk_html)
        self.assertNotIn("提单确认", bk_html)
        self.assertNotIn("开单确认", bk_html)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "nightly_jobs"
            log_dir.mkdir()
            (log_dir / "sample_job.log").write_text(
                "\n".join(
                    [
                        "=================================== FAILURES ===================================",
                        "__________________________________ test_bar ___________________________________",
                        "E       AssertionError: boom",
                        "tests/test_mod.py:3: AssertionError",
                        "FAILED tests/test_mod.py::test_bar - AssertionError: boom",
                        "========================= 1 failed in 0.01s =========================",
                    ]
                ),
                encoding="utf-8",
            )
            buf = io.StringIO()
            emit_report_html(
                title="Nightly layout",
                repo_root=root,
                log_dir=log_dir,
                out_fp=buf,
                bk_results={
                    CUDA_TARGET: (
                        {"number": 10, "state": "failed", "message": "Scheduled nightly", "commit": "abc"},
                        [{
                            "name": "Diffusion X2V Wan2.2 Accuracy Test",
                            "state": "failed",
                            "info": info,
                            "step_link": "https://buildkite.com/vllm/vllm-omni/builds/10#aaa",
                            "log_error": None,
                            "ci_versions": {},
                            "build_commit_short": "abc",
                        }],
                        None,
                    ),
                    NPU_TARGET: (None, None, "NPU nightly was not loaded."),
                },
                weekly_results={},
                issue_alert=False,
                issue_context=IssueContext(issues=[], pipelines={}, confirmations=[], issues_loaded=True),
            )
            html = buf.getvalue()

        issue_at = html.index('<span class="heading-label">Issue</span>')
        bk_at = html.index('<span class="heading-label">Buildkite</span>')
        local_at = html.index('<span class="heading-label">Local test</span>')
        self.assertLess(issue_at, bk_at)
        self.assertLess(bk_at, local_at)
        issue_chunk = html[issue_at:bk_at]
        bk_chunk = html[bk_at:local_at]
        script_at = html.find("<script>", local_at)
        self.assertGreater(script_at, local_at)
        local_chunk = html[local_at:script_at]
        self.assertIn("<h3>Summary</h3>", issue_chunk)
        self.assertLess(
            issue_chunk.index(">一键操作</button>"),
            issue_chunk.index('<summary class="issue-bucket-card-summary">待新增</summary>'),
        )
        self.assertIn('<summary class="issue-bucket-card-summary">待关闭</summary>', issue_chunk)
        self.assertIn('<summary class="issue-bucket-card-summary">待更新</summary>', issue_chunk)
        self.assertNotIn('class="issue-bucket-card" open', issue_chunk)
        self.assertEqual(issue_chunk.count("issue-table-empty"), 3)
        self.assertNotIn('data-issue-row="1"', issue_chunk)
        self.assertIn('id="issue-bulk-apply" disabled', issue_chunk)
        self.assertNotIn("Issue Alert", html)
        self.assertNotIn("Issue close confirmation", html)
        self.assertNotIn("Failure analysis", bk_chunk)
        self.assertIn("Failure analysis", local_chunk)
        self.assertIn("View full log", local_chunk)
        self.assertNotIn("Submit issue", local_chunk)
        self.assertNotIn("提单确认", local_chunk)
        self.assertNotIn("开单确认", local_chunk)
        self.assertNotIn("一键操作", local_chunk)
        self.assertNotIn("btn-github-issue", local_chunk)
        self.assertNotIn("关闭确认", local_chunk)
        self.assertNotIn("更新评论", local_chunk)


class BulkApplyTests(unittest.TestCase):
    def _section(self):
        issue = _issue(10, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")
        failing = _issue(11, "Omni Function Test Extra Long Name failed on H100")
        pipelines = {
            "CUDA": _snap(
                "CUDA",
                [
                    _job("Diffusion X2V Wan2.2 Accuracy Test"),
                    _job("Omni Function Test Extra Long Name", state="failed", failed=["a.py::t"]),
                ],
                number=20013,
            )
        }
        rows = [evaluate_issue(issue, pipelines), evaluate_issue(failing, pipelines)]
        return assemble_issue_section(
            groups=[_group(verdict="new")],
            bare_jobs=[],
            context=_ctx([issue, failing], pipelines, rows),
            search_ran=True,
        )

    def test_payload_covers_three_ops_and_has_no_token(self):
        from issue_section import bulk_payload

        payload = bulk_payload(self._section())
        ops = [action["op"] for action in payload["actions"]]
        self.assertEqual(ops, ["create", "close", "comment"])
        self.assertEqual(payload["actions"][1]["issue_number"], 10)
        self.assertEqual(payload["actions"][2]["issue_number"], 11)
        keys = set()

        def _walk(obj):
            if isinstance(obj, dict):
                keys.update(str(key).lower() for key in obj)
                for value in obj.values():
                    _walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    _walk(value)

        _walk(payload)
        self.assertFalse(keys & {"token", "github_token", "authorization", "gh_token"})
        self.assertNotIn("ghp_", json.dumps(payload))

    def test_dry_run_does_not_call_github_or_print_token(self):
        import os
        from apply_nightly_issue_actions import apply_payload, main

        token = "ghp_SHOULD_NOT_LEAK_9911"
        payload = bulk_payload_safe(self._section())
        calls: list[str] = []

        def opener(*_args, **_kwargs):
            calls.append("called")
            raise AssertionError("dry run must not call GitHub")

        result = apply_payload(payload, write=False, token=token, opener=opener)
        self.assertEqual(calls, [])
        self.assertEqual(result["failed"], 0)
        self.assertEqual(len(result["results"]), 3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            old_token = os.environ.get("GITHUB_TOKEN")
            os.environ["GITHUB_TOKEN"] = token
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                from contextlib import redirect_stderr, redirect_stdout

                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(["--payload", str(path)])
            finally:
                if old_token is None:
                    os.environ.pop("GITHUB_TOKEN", None)
                else:
                    os.environ["GITHUB_TOKEN"] = old_token
        text = stdout.getvalue() + stderr.getvalue()
        self.assertEqual(code, 0)
        self.assertNotIn(token, text)
        self.assertIn("create", text)
        self.assertIn("Dry run only", text)

    def test_apply_files_closes_and_comments(self):
        from apply_nightly_issue_actions import apply_payload

        token = "ghp_SHOULD_NOT_LEAK_9911"
        payload = bulk_payload_safe(self._section())
        calls: list[tuple[str, str]] = []

        class _Resp:
            def __init__(self, body: dict):
                self._raw = json.dumps(body).encode()

            def read(self):
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def opener(req, timeout=60):
            self.assertEqual(req.get_header("Authorization"), f"Bearer {token}")
            calls.append((req.get_method(), req.full_url))
            if req.full_url.endswith("/comments"):
                return _Resp({"id": 1})
            if req.get_method() == "PATCH":
                return _Resp({"state": "closed"})
            return _Resp({"number": 50, "html_url": "https://github.com/vllm-project/vllm-omni/issues/50"})

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.json"
            outcome = apply_payload(
                payload, write=True, token=token, opener=opener, results_path=results
            )
            again = apply_payload(
                payload, write=True, token=token, opener=opener, results_path=results
            )
        self.assertEqual(outcome["created"], 1)
        self.assertEqual(outcome["closed"], 1)
        self.assertEqual(outcome["commented"], 1)
        self.assertEqual(outcome["failed"], 0)
        self.assertEqual(calls[0][0], "POST")
        self.assertTrue(calls[0][1].endswith("/issues"))
        self.assertTrue(any(url.endswith("/issues/10/comments") for _method, url in calls))
        self.assertTrue(any(method == "PATCH" and url.endswith("/issues/10") for method, url in calls))
        self.assertTrue(any(url.endswith("/issues/11/comments") for _method, url in calls))
        self.assertEqual(again["skipped"], 3)
        self.assertEqual(len(calls), 4)
        blob = json.dumps(outcome)
        self.assertNotIn(token, blob)


def bulk_payload_safe(section):
    from issue_section import bulk_payload

    return bulk_payload(section)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for issue_close_confirm (no network)."""

from __future__ import annotations

import unittest
from collections import namedtuple

from issue_close_confirm import (
    STATUS_CLOSABLE,
    STATUS_FAILING,
    STATUS_INSUFFICIENT,
    PipelineSnapshot,
    attach_clean_streaks,
    collect_close_candidates,
    collect_confirmations,
    count_consecutive_clean_runs,
    evaluate_issue,
    extract_node_ids,
    infer_sides,
    job_kind,
    match_jobs,
    node_still_failing,
    normalize_job_name,
    render_issue_close_confirm_html,
    snapshots_from_bk_results,
)


def _issue(number: int, title: str, body: str = "", labels: list[str] | None = None) -> dict:
    return {
        "number": number,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/vllm-project/vllm-omni/issues/{number}",
        "labels": [{"name": n} for n in (labels or ["ci-failure"])],
    }


def _job(name: str, *, state: str = "passed", failed: list[str] | None = None,
         errors: list[str] | None = None, summary: str | None = "3 passed in 1.0s",
         log_error: str | None = None) -> dict:
    info = {"failed_nodes": list(failed or []), "error_nodes": list(errors or []), "summary": summary}
    return {"name": name, "state": state, "info": info, "log_error": log_error, "raw_url": "https://example"}


def _snap(label: str, jobs: list[dict], *, number: int = 1, available: bool = True) -> PipelineSnapshot:
    return PipelineSnapshot(
        label=label,
        build_number=number,
        build_url=f"https://buildkite.com/vllm/pipe/builds/{number}",
        jobs=jobs,
        available=available,
        org="vllm",
        pipeline="pipe",
    )


class ExtractAndMatchTests(unittest.TestCase):
    def test_extract_node_ids_unique_in_order(self):
        text = (
            "fails tests/e2e/test_wan.py::test_ssim[case] and also "
            "tests/e2e/test_wan.py::test_ssim[case] again, plus "
            "foo.py::test_bar"
        )
        self.assertEqual(
            extract_node_ids(text),
            ["tests/e2e/test_wan.py::test_ssim[case]", "foo.py::test_bar"],
        )

    def test_normalize_strips_emoji_and_separators(self):
        self.assertEqual(
            normalize_job_name(":rocket: Diffusion X2V · Wan2.2 Accuracy Test"),
            "diffusion x2v wan2.2 accuracy test",
        )

    def test_match_jobs_requires_distinctive_name(self):
        jobs = [
            _job("CI"),
            _job("Diffusion X2V Wan2.2 Accuracy Test"),
        ]
        self.assertEqual(
            match_jobs("The Diffusion X2V Wan2.2 Accuracy Test is failing", jobs),
            ["Diffusion X2V Wan2.2 Accuracy Test"],
        )
        self.assertEqual(match_jobs("CI is red", jobs), [])

    def test_match_jobs_drops_shorter_substring(self):
        jobs = [
            _job("TTS Perf Test"),
            _job("TTS Perf Test 2 GPU"),
        ]
        self.assertEqual(
            match_jobs("TTS Perf Test 2 GPU failed last night", jobs),
            ["TTS Perf Test 2 GPU"],
        )

    def test_infer_sides_from_mentions_and_fallback(self):
        self.assertEqual(infer_sides(_issue(1, "NPU job failed")), {"NPU"})
        self.assertEqual(infer_sides(_issue(2, "H100 Single-GPU failed")), {"CUDA"})
        self.assertEqual(infer_sides(_issue(3, "CUDA and NPU both fail")), {"CUDA", "NPU"})
        self.assertEqual(
            infer_sides(_issue(4, "generic fail"), matched_sides={"CUDA"}),
            {"CUDA"},
        )
        self.assertEqual(infer_sides(_issue(5, "generic fail")), set())


class JobKindAndEvaluateTests(unittest.TestCase):
    def test_job_kind_ok_fail_unknown(self):
        self.assertEqual(job_kind(_job("A")), "ok")
        self.assertEqual(job_kind(_job("A", failed=["a.py::t"])), "fail")
        self.assertEqual(job_kind(_job("A", state="failed")), "fail")
        self.assertEqual(job_kind(_job("A", log_error="timeout")), "unknown")
        self.assertEqual(job_kind(_job("A", summary="")), "unknown")
        self.assertEqual(job_kind(_job("A", summary="no counts here")), "unknown")

    def test_node_still_failing_matches_basename(self):
        self.assertTrue(
            node_still_failing(
                "tests/e2e/test_wan.py::test_ssim[case]",
                ["test_wan.py::test_ssim[case]"],
            )
        )
        self.assertFalse(
            node_still_failing("foo.py::test_a", ["foo.py::test_b"])
        )

    def test_closable_when_job_green_and_node_absent(self):
        issue = _issue(
            10,
            "Diffusion X2V Wan2.2 Accuracy Test failed on H100",
            "tests/e2e/test_wan.py::test_ssim[case]",
        )
        pipelines = {
            "CUDA": _snap("CUDA", [_job("Diffusion X2V Wan2.2 Accuracy Test")], number=20013),
            "NPU": _snap("NPU", [_job("Unrelated NPU Job")], number=11925),
        }
        cand = evaluate_issue(issue, pipelines)
        self.assertEqual(cand.status, STATUS_CLOSABLE)
        self.assertEqual(cand.sides, ("CUDA",))
        self.assertIn("CUDA", cand.results)
        self.assertNotIn("NPU", cand.results)
        self.assertIn("passed on #20013", cand.results["CUDA"])

    def test_not_closable_when_job_still_fails(self):
        issue = _issue(11, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")
        pipelines = {
            "CUDA": _snap(
                "CUDA",
                [_job("Diffusion X2V Wan2.2 Accuracy Test", state="failed", failed=["a.py::t"])],
            ),
        }
        row = evaluate_issue(issue, pipelines)
        self.assertEqual(row.status, STATUS_FAILING)
        self.assertEqual(row.clean_streak, 0)

    def test_not_closable_when_node_still_failing_elsewhere(self):
        issue = _issue(
            12,
            "Diffusion X2V Wan2.2 Accuracy Test failed on H100",
            "tests/e2e/test_wan.py::test_ssim[case]",
        )
        pipelines = {
            "CUDA": _snap(
                "CUDA",
                [
                    _job("Diffusion X2V Wan2.2 Accuracy Test"),
                    _job("Other Job", failed=["tests/e2e/test_wan.py::test_ssim[case]"]),
                ],
            ),
        }
        self.assertEqual(evaluate_issue(issue, pipelines).status, STATUS_FAILING)

    def test_npu_only_issue_not_closed_by_cuda_green(self):
        issue = _issue(13, "Omni Function Test NPU failed")
        pipelines = {
            "CUDA": _snap("CUDA", [_job("Omni Function Test NPU")]),
            "NPU": _snap(
                "NPU",
                [_job("Omni Function Test NPU", state="failed", failed=["a.py::t"])],
            ),
        }
        self.assertEqual(evaluate_issue(issue, pipelines).status, STATUS_FAILING)

    def test_both_sides_required_when_issue_mentions_both(self):
        issue = _issue(14, "Omni Function Test failed on CUDA and NPU")
        pipelines = {
            "CUDA": _snap("CUDA", [_job("Omni Function Test")]),
            "NPU": _snap("NPU", [_job("Omni Function Test", state="failed", failed=["a.py::t"])], number=2),
        }
        self.assertEqual(evaluate_issue(issue, pipelines).status, STATUS_FAILING)
        pipelines["NPU"] = _snap("NPU", [_job("Omni Function Test")], number=2)
        cand = evaluate_issue(issue, pipelines)
        self.assertEqual(cand.status, STATUS_CLOSABLE)
        self.assertEqual(cand.sides, ("CUDA", "NPU"))

    def test_missing_pipeline_is_not_closable(self):
        issue = _issue(15, "Omni Function Test failed on NPU")
        pipelines = {"CUDA": _snap("CUDA", [_job("Omni Function Test")])}
        self.assertEqual(evaluate_issue(issue, pipelines).status, STATUS_INSUFFICIENT)

    def test_unknown_kind_is_not_closable(self):
        issue = _issue(16, "Omni Function Test failed on H100")
        pipelines = {
            "CUDA": _snap("CUDA", [_job("Omni Function Test", summary="", log_error="fetch failed")]),
        }
        self.assertEqual(evaluate_issue(issue, pipelines).status, STATUS_INSUFFICIENT)

    def test_no_job_identity_is_not_closable(self):
        issue = _issue(17, "something is broken on H100")
        pipelines = {"CUDA": _snap("CUDA", [_job("Omni Function Test")])}
        self.assertEqual(evaluate_issue(issue, pipelines).status, STATUS_INSUFFICIENT)


class StreakAndRenderTests(unittest.TestCase):
    def test_count_consecutive_clean_runs(self):
        name = "Omni Function Test"
        history = [
            (3, [_job(name, state="passed")]),
            (2, [_job(name, state="passed")]),
            (1, [_job(name, state="failed")]),
        ]
        self.assertEqual(count_consecutive_clean_runs([name], history), 2)
        history_fail_latest = [
            (3, [_job(name, state="failed")]),
            (2, [_job(name, state="passed")]),
        ]
        self.assertEqual(count_consecutive_clean_runs([name], history_fail_latest), 0)
        self.assertEqual(count_consecutive_clean_runs([name], []), 0)

    def test_attach_streak_uses_min_across_required_sides(self):
        issue = _issue(14, "Omni Function Test failed on CUDA and NPU")
        pipelines = {
            "CUDA": _snap("CUDA", [_job("Omni Function Test")], number=10),
            "NPU": _snap("NPU", [_job("Omni Function Test")], number=20),
        }
        rows = collect_confirmations([issue], pipelines, histories={})
        self.assertEqual(rows[0].status, STATUS_CLOSABLE)
        # No history yet → n/a until attach gets histories
        self.assertIsNone(rows[0].clean_streak)
        histories = {
            "CUDA": [
                (10, [_job("Omni Function Test", state="passed")]),
                (9, [_job("Omni Function Test", state="passed")]),
                (8, [_job("Omni Function Test", state="passed")]),
            ],
            "NPU": [
                (20, [_job("Omni Function Test", state="passed")]),
                (19, [_job("Omni Function Test", state="failed")]),
            ],
        }
        attach_clean_streaks(rows, histories)
        self.assertEqual(rows[0].clean_streak, 1)

    def test_snapshots_from_bk_results(self):
        Target = namedtuple("Target", "label org pipeline")
        target = Target(label="CUDA", org="vllm", pipeline="vllm-omni")
        build = {"number": 20013}
        jobs = [_job("A")]
        snaps = snapshots_from_bk_results({target: (build, jobs, None)})
        self.assertTrue(snaps["CUDA"].available)
        self.assertEqual(snaps["CUDA"].build_number, 20013)
        self.assertEqual(snaps["CUDA"].build_url, "https://buildkite.com/vllm/vllm-omni/builds/20013")

    def test_render_lists_all_issues_in_english_with_streak_column(self):
        empty = render_issue_close_confirm_html(issues=[], candidates=[], pipelines={}, rows=[])
        self.assertIn("Issue close confirmation", empty)
        self.assertIn("bk-section panel nightly-root nightly-root--issue-close", empty)
        self.assertNotIn("Issue 关闭确认", empty)
        self.assertIn("No open ci-failure issues were loaded", empty)

        closable = _issue(
            10,
            "Diffusion X2V Wan2.2 Accuracy Test failed on H100",
            "tests/e2e/test_wan.py::test_ssim",
        )
        failing = _issue(11, "Diffusion X2V Wan2.2 Accuracy Test still failing on H100")
        pipelines = {
            "CUDA": _snap(
                "CUDA",
                [_job("Diffusion X2V Wan2.2 Accuracy Test")],
                number=20013,
            ),
        }
        # Make #11 fail by giving a failing job — but same name green in snap.
        # Evaluate #11 against a failing pipeline instead, then combine rows.
        failing_pipelines = {
            "CUDA": _snap(
                "CUDA",
                [_job("Diffusion X2V Wan2.2 Accuracy Test", state="failed", failed=["a.py::t"])],
                number=20013,
            ),
        }
        rows = collect_confirmations([closable], pipelines, histories={
            "CUDA": [
                (20013, [_job("Diffusion X2V Wan2.2 Accuracy Test", state="passed")]),
                (20000, [_job("Diffusion X2V Wan2.2 Accuracy Test", state="passed")]),
                (19990, [_job("Diffusion X2V Wan2.2 Accuracy Test", state="passed")]),
            ]
        })
        rows += collect_confirmations([failing], failing_pipelines)
        html = render_issue_close_confirm_html(
            issues=[closable, failing],
            candidates=[r for r in rows if r.status == STATUS_CLOSABLE],
            pipelines=pipelines,
            rows=rows,
        )
        self.assertIn("Issue close confirmation", html)
        self.assertIn("Consecutive clean runs", html)
        self.assertIn("Confirmation", html)
        self.assertIn("#10", html)
        self.assertIn("#11", html)
        self.assertIn('<span class="issue-close-streak">3</span> scheduled nightlies', html)
        self.assertIn("Theoretically closable", html)
        self.assertIn("Still failing", html)
        self.assertIn("<strong>2</strong> open ci-failure issues scanned", html)


if __name__ == "__main__":
    unittest.main()

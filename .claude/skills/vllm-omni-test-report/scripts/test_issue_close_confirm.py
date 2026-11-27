#!/usr/bin/env python3
"""Unit tests for issue_close_confirm (no network)."""

from __future__ import annotations

import unittest
from collections import namedtuple

from issue_close_confirm import (
    LANE_B200,
    STATUS_CLOSABLE,
    STATUS_FAILING,
    STATUS_INSUFFICIENT,
    PipelineSnapshot,
    attach_clean_streaks,
    b200_branches_for_issue,
    collect_confirmations,
    count_consecutive_clean_runs,
    evaluate_issue,
    extract_node_ids,
    infer_sides,
    is_b200_issue,
    is_scheduled_release_b200,
    job_kind,
    match_jobs,
    node_still_failing,
    normalize_job_name,
    select_scheduled_release_b200_builds,
    snapshots_from_bk_results,
)


def _issue(number: int, title: str, body: str = "", labels: list[str] | None = None) -> dict:
    return {
        "number": number,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/vllm-project/vllm-omni/issues/{number}",
        "labels": [{"name": name} for name in (labels or ["ci-failure"])],
        "state": "open",
    }


def _job(
    name: str,
    *,
    state: str = "passed",
    failed: list[str] | None = None,
    errors: list[str] | None = None,
    summary: str | None = "3 passed in 1.0s",
    log_error: str | None = None,
) -> dict:
    info = {
        "failed_nodes": list(failed or []),
        "error_nodes": list(errors or []),
        "summary": summary,
    }
    return {"name": name, "state": state, "info": info, "log_error": log_error, "raw_url": "https://example"}


def _snap(
    label: str,
    jobs: list[dict],
    *,
    number: int = 1,
    available: bool = True,
    lane: str | None = None,
    branch: str = "main",
    pipeline: str = "pipe",
) -> PipelineSnapshot:
    kwargs = dict(
        label=label,
        build_number=number,
        build_url=f"https://buildkite.com/vllm/{pipeline}/builds/{number}",
        jobs=jobs,
        available=available,
        org="vllm",
        pipeline=pipeline,
        branch=branch,
    )
    if lane is not None:
        kwargs["lane"] = lane
    return PipelineSnapshot(**kwargs)


def _b200_snap(
    jobs: list[dict],
    *,
    number: int = 20135,
    branch: str = "release/v0.34.0",
    available: bool = True,
) -> PipelineSnapshot:
    return _snap(
        "CUDA",
        jobs,
        number=number,
        available=available,
        lane=LANE_B200,
        branch=branch,
        pipeline="vllm-omni",
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
        self.assertFalse(node_still_failing("foo.py::test_a", ["foo.py::test_b"]))

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

    def test_skipped_state_is_not_positive_pass(self):
        issue = _issue(18, "Omni Function Test failed on H100")
        pipelines = {"CUDA": _snap("CUDA", [_job("Omni Function Test", state="skipped", summary="")])}
        self.assertEqual(evaluate_issue(issue, pipelines).status, STATUS_INSUFFICIENT)


class StreakAndSnapshotTests(unittest.TestCase):
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
        snaps = snapshots_from_bk_results({target: ({"number": 20013}, [_job("A")], None)})
        self.assertTrue(snaps["CUDA"].available)
        self.assertEqual(snaps["CUDA"].build_number, 20013)
        self.assertEqual(snaps["CUDA"].build_url, "https://buildkite.com/vllm/vllm-omni/builds/20013")


class B200CloseContractTests(unittest.TestCase):
    """[B200] issues must be judged from Scheduled Release B200, not daily nightly."""

    def test_detects_b200_tag_in_title_body_or_labels(self):
        self.assertTrue(is_b200_issue(_issue(1, "[Bug][B200]: job failed on H100")))
        self.assertTrue(is_b200_issue(_issue(2, "job failed", "see [B200] log")))
        self.assertTrue(is_b200_issue(_issue(3, "job failed", labels=["ci-failure", "[B200]"])))
        self.assertTrue(is_b200_issue(_issue(4, "[Bug][ b200 ]: job failed")))
        self.assertFalse(is_b200_issue(_issue(5, "job failed on a B200 node")))
        self.assertFalse(is_b200_issue(_issue(6, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")))

    def test_branches_from_issue_ignore_sidecar_and_backport_names(self):
        issue = _issue(
            10660,
            "[Bug][B200]: Nightly CI failed - Diffusion X2V MiniMax H3 Function Test",
            "Nightly CI (Buildkite vllm/vllm-omni #20135, Scheduled Release B200 build)\n"
            "Branch: release/v0.34.0\n",
        )
        self.assertEqual(b200_branches_for_issue(issue), ("release/v0.34.0",))
        self.assertEqual(b200_branches_for_issue(_issue(1, "[B200] failed on main")), ("main",))
        text = (
            "also mentioned release/v0.34.0-docker and "
            "backport/10687-release-v0.34.0 and release/v0.34.0rc1"
        )
        self.assertEqual(b200_branches_for_issue(_issue(2, "[B200] x", text)), ("release/v0.34.0rc1",))

    def test_b200_build_match_is_first_line_and_ignores_source(self):
        self.assertTrue(
            is_scheduled_release_b200(
                {"message": "Scheduled Release B200 build", "source": "ui", "number": 20135}
            )
        )
        self.assertTrue(
            is_scheduled_release_b200(
                {"message": "  scheduled   release   b200   build  \nbody", "source": "schedule"}
            )
        )
        self.assertFalse(
            is_scheduled_release_b200({"message": "Scheduled nightly build", "source": "schedule"})
        )
        self.assertFalse(
            is_scheduled_release_b200(
                {"message": "Fix typo\nScheduled Release B200 build", "source": "ui"}
            )
        )
        self.assertFalse(
            is_scheduled_release_b200(
                {"message": "Add experimental Scheduled Release B200 build docs", "source": "ui"}
            )
        )
        selected = select_scheduled_release_b200_builds(
            [
                {"number": 21145, "message": "Scheduled nightly build", "source": "schedule"},
                {"number": 20135, "message": "Scheduled Release B200 build", "source": "ui"},
                {"number": 20000, "message": "Scheduled weekly build", "source": "schedule"},
                {"number": 19900, "message": "PR: mention Scheduled Release B200 build"},
            ]
        )
        self.assertEqual([int(b["number"]) for b in selected], [20135])

    def test_b200_issue_is_not_closable_from_daily_nightly(self):
        issue = _issue(
            10660,
            "[Bug][B200]: Diffusion X2V · MiniMax H3 Function Test failed",
            "Branch: release/v0.34.0\n"
            "Job title is \"Diffusion X2V · MiniMax H3 Function Test\".\n"
            "tests/e2e/online_serving/minimax_h3/test_minimax_h3_dlo_dp2_fl2va_ref2va.py::"
            "test_minimax_h3_ref2va_partition[minimax_h3_dlo_dp2_ref2va_partition]",
        )
        nightly = {
            "CUDA": _snap("CUDA", [_job("Diffusion X2V · MiniMax H3 Function Test")], number=21145),
            "NPU": _snap("NPU", [_job("Unrelated NPU Job")], number=13097),
        }
        rows = collect_confirmations([issue], nightly, b200_pipelines={})
        self.assertEqual(rows[0].status, STATUS_INSUFFICIENT)
        self.assertIn("Scheduled Release B200", rows[0].reason)

        failing_b200 = {
            "release/v0.34.0": _b200_snap(
                [_job("Diffusion X2V · MiniMax H3 Function Test", state="failed", failed=["a.py::t"])],
                number=20135,
            )
        }
        rows = collect_confirmations([issue], nightly, b200_pipelines=failing_b200)
        self.assertEqual(rows[0].status, STATUS_FAILING)
        self.assertIn("20135", rows[0].reason)
        self.assertNotIn("#21145", rows[0].reason)
        self.assertEqual(rows[0].evidence["CUDA"].build_number, 20135)
        self.assertEqual(rows[0].evidence["CUDA"].lane, LANE_B200)

    def test_b200_issue_is_closable_only_from_release_b200_build(self):
        issue = _issue(
            10661,
            "[Bug][B200]: Diffusion X2V · MiniMax H3 I2VA/Ref2VA/Timeline Guide Accuracy Test failed",
            "Branch: release/v0.34.0\n"
            "Job title is \"Diffusion X2V · MiniMax H3 I2VA/Ref2VA/Timeline Guide Accuracy Test\".\n"
            "tests/e2e/accuracy/minimax_h3/test_minimax_h3_i2va_ref2va_similarity.py::"
            "test_minimax_h3_i2va_matches_official_reference",
        )
        job = "Diffusion X2V · MiniMax H3 I2VA/Ref2VA/Timeline Guide Accuracy Test"
        nightly = {"CUDA": _snap("CUDA", [_job(job, state="failed", failed=["a.py::t"])], number=21145)}
        # A green B200 on main must not close a release/v0.34.0 issue.
        b200 = {
            "main": _b200_snap([_job(job)], number=19800, branch="main"),
            "release/v0.34.0": _b200_snap([_job(job)], number=20140, branch="release/v0.34.0"),
        }
        rows = collect_confirmations([issue], nightly, b200_pipelines=b200)
        self.assertEqual(rows[0].status, STATUS_CLOSABLE)
        self.assertEqual(rows[0].evidence["CUDA"].build_number, 20140)
        self.assertIn("Scheduled Release B200 #20140 (release/v0.34.0)", rows[0].reason)
        self.assertNotIn("#21145", rows[0].reason)
        self.assertNotIn("#19800", rows[0].reason)

    def test_non_b200_issue_still_uses_daily_nightly(self):
        issue = _issue(10, "Diffusion X2V Wan2.2 Accuracy Test failed on H100")
        nightly = {"CUDA": _snap("CUDA", [_job("Diffusion X2V Wan2.2 Accuracy Test")], number=21145)}
        b200 = {"main": _b200_snap([_job("Diffusion X2V Wan2.2 Accuracy Test")], number=20135, branch="main")}
        rows = collect_confirmations([issue], nightly, b200_pipelines=b200)
        self.assertEqual(rows[0].status, STATUS_CLOSABLE)
        self.assertEqual(rows[0].evidence["CUDA"].build_number, 21145)
        self.assertIn("passed on #21145", rows[0].results["CUDA"])
        self.assertNotIn("Scheduled Release B200", rows[0].reason)

    def test_npu_nightly_is_not_b200_evidence(self):
        issue = _issue(
            8,
            "[Bug][B200]: Omni Function Test failed",
            "Job title is \"Omni Function Test\".",
        )
        nightly = {
            "CUDA": _snap("CUDA", [_job("Omni Function Test")], number=21145),
            "NPU": _snap("NPU", [_job("Omni Function Test")], number=13097),
        }
        rows = collect_confirmations([issue], nightly, b200_pipelines={})
        self.assertEqual(rows[0].status, STATUS_INSUFFICIENT)
        self.assertNotIn("13097", rows[0].reason)

    def test_b200_streak_ignores_daily_nightly_history(self):
        issue = _issue(
            9,
            "[Bug][B200]: Omni Function Test failed",
            "Job title is \"Omni Function Test\". Branch: release/v0.34.0",
        )
        nightly = {"CUDA": _snap("CUDA", [_job("Omni Function Test")], number=21145)}
        b200 = {"release/v0.34.0": _b200_snap([_job("Omni Function Test")], number=20140)}
        rows = collect_confirmations([issue], nightly, b200_pipelines=b200)
        self.assertEqual(rows[0].status, STATUS_CLOSABLE)
        self.assertIsNone(rows[0].clean_streak)
        nightly_hist = {
            "CUDA": [
                (21145, [_job("Omni Function Test", state="passed")]),
                (21052, [_job("Omni Function Test", state="passed")]),
                (20900, [_job("Omni Function Test", state="passed")]),
            ]
        }
        b200_hist = {
            "release/v0.34.0": [
                (20140, [_job("Omni Function Test", state="passed")]),
                (20135, [_job("Omni Function Test", state="failed")]),
            ]
        }
        attach_clean_streaks(rows, nightly_hist, b200_histories=b200_hist)
        self.assertEqual(rows[0].clean_streak, 1)

    def test_dual_tagged_b200_h100_requires_both_lanes(self):
        issue = _issue(
            10655,
            "[Bug][B200][H100][Release]: Omni Function Test failed",
            "Branch: release/v0.34.0\nJob title is \"Omni Function Test\".",
        )
        nightly = {"CUDA": _snap("CUDA", [_job("Omni Function Test")], number=21145)}
        b200_fail = {
            "release/v0.34.0": _b200_snap(
                [_job("Omni Function Test", state="failed", failed=["a.py::t"])],
                number=20135,
            )
        }
        rows = collect_confirmations([issue], nightly, b200_pipelines=b200_fail)
        self.assertEqual(rows[0].status, STATUS_FAILING)
        self.assertIn("B200", rows[0].sides)
        self.assertIn("CUDA", rows[0].sides)
        b200_ok = {"release/v0.34.0": _b200_snap([_job("Omni Function Test")], number=20140)}
        rows = collect_confirmations([issue], nightly, b200_pipelines=b200_ok)
        self.assertEqual(rows[0].status, STATUS_CLOSABLE)
        self.assertEqual(rows[0].evidence["B200"].build_number, 20140)
        self.assertEqual(rows[0].evidence["CUDA"].build_number, 21145)


if __name__ == "__main__":
    unittest.main()

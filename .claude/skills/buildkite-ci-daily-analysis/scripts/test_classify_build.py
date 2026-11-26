#!/usr/bin/env python3
"""Unit tests for CI duration buckets in ci_daily_analysis.classify_build. No network."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import ci_daily_analysis as m


def _build(
    branch: str,
    message: str = "",
    source: str = "webhook",
) -> dict:
    return {"branch": branch, "message": message, "source": source}


class ClassifyBuildTests(unittest.TestCase):
    def test_feature_branch_is_ready(self) -> None:
        self.assertEqual(m.classify_build(_build("alice/diffusion-fp8", "PR title")), m.CI_BUCKET_READY)

    def test_feature_branch_stays_ready_even_with_scheduled_title(self) -> None:
        self.assertEqual(
            m.classify_build(_build("feat/nightly-lane", "Scheduled nightly build", source="schedule")),
            m.CI_BUCKET_READY,
        )

    def test_main_and_release_ordinary_are_merge(self) -> None:
        for branch in ("main", "release/v0.34.0", "release/v0.34.0rc1", "release/v0.14.0.post1"):
            with self.subTest(branch=branch):
                self.assertEqual(
                    m.classify_build(_build(branch, "PR #4521 merged", source="schedule")),
                    m.CI_BUCKET_MERGE,
                )

    def test_scheduled_nightly_and_release_b200_are_nightly(self) -> None:
        titles = (
            "Scheduled nightly build",
            "  Scheduled   nightly   build  \nbody that mentions weekly",
            "SCHEDULED NIGHTLY BUILD",
            "Scheduled Release B200 build",
            "scheduled release b200 build\r\nextra",
        )
        branches = ("main", "release/v0.34.0")
        for branch in branches:
            for title in titles:
                for source in ("schedule", "ui"):
                    with self.subTest(branch=branch, title=title, source=source):
                        self.assertEqual(
                            m.classify_build(_build(branch, title, source=source)),
                            m.CI_BUCKET_NIGHTLY,
                        )

    def test_scheduled_weekly_titles_are_weekly(self) -> None:
        titles = (
            "Scheduled weekly build",
            "Scheduled Non-Critical weekly build",
            "scheduled   non-critical   weekly   build",
            "SCHEDULED WEEKLY BUILD",
        )
        for branch in ("main", "release/v0.34.0"):
            for title in titles:
                for source in ("schedule", "ui"):
                    with self.subTest(branch=branch, title=title, source=source):
                        self.assertEqual(
                            m.classify_build(_build(branch, title, source=source)),
                            m.CI_BUCKET_WEEKLY,
                        )

    def test_main_pr_title_that_mentions_nightly_stays_merge(self) -> None:
        self.assertEqual(
            m.classify_build(
                _build("main", "Add experimental AMD MI300 nightly lane", source="webhook"),
            ),
            m.CI_BUCKET_MERGE,
        )

    def test_loose_substrings_and_later_lines_do_not_match(self) -> None:
        cases = (
            ("main", "Add experimental AMD MI300 nightly lane"),
            ("release/v0.34.0", "Enable weekly regression on MI300"),
            ("main", "Scheduled weekly build(Release)"),
            ("main", "Scheduled nightly build extra"),
            ("release/v0.34.0", "AMD nightly"),
            ("main", "PR title\nScheduled nightly build"),
            ("main", "Scheduled Non Critical weekly build"),
        )
        for branch, message in cases:
            with self.subTest(branch=branch, message=message):
                self.assertEqual(m.classify_build(_build(branch, message, source="ui")), m.CI_BUCKET_MERGE)

    def test_version_sidecar_and_lookalike_branches_stay_ready(self) -> None:
        for branch in (
            "release/v0.34.0-docker",
            "backport/10687-release-v0.34.0",
            "release/nightly",
            "main-doc",
        ):
            with self.subTest(branch=branch):
                self.assertEqual(
                    m.classify_build(_build(branch, "Scheduled nightly build", source="ui")),
                    m.CI_BUCKET_READY,
                )

    def test_release_rc_and_post_follow_main_rules(self) -> None:
        self.assertEqual(
            m.classify_build(_build("release/v0.32.0rc1", "Scheduled Release B200 build", source="ui")),
            m.CI_BUCKET_NIGHTLY,
        )
        self.assertEqual(
            m.classify_build(_build("release/v0.18.0.post1", "Scheduled Non-Critical weekly build")),
            m.CI_BUCKET_WEEKLY,
        )

    def test_message_less_scheduled_build_is_merge(self) -> None:
        sunday = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
        build = _build("main", "", source="schedule")
        build["created_at"] = sunday.isoformat()
        self.assertEqual(m.classify_build(build), m.CI_BUCKET_MERGE)

    def test_report_legend_matches_bucket_rules(self) -> None:
        html_out = m.render_html([], "2026-11-26", ["vllm-omni"])
        self.assertIn("Scheduled Release B200 build", html_out)
        self.assertIn("Scheduled nightly build", html_out)
        self.assertIn("Scheduled weekly build", html_out)
        self.assertIn("Scheduled Non-Critical weekly build", html_out)
        self.assertIn("release/v0.34.0", html_out)
        self.assertNotIn("source=schedule", html_out)
        self.assertNotIn("message contains", html_out)


if __name__ == "__main__":
    unittest.main()

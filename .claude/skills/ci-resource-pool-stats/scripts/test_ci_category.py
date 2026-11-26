#!/usr/bin/env python3
"""Unit tests for resource-pool CI buckets. No network."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import resource_pool_stats as pool


class DetermineCiCategoryTests(unittest.TestCase):
    def _cat(self, branch: str, message: str = "", source: str = "webhook", created_at: datetime | None = None) -> str:
        return pool._determine_ci_category(branch, source, created_at, message)

    def test_feature_branch_is_ready(self) -> None:
        self.assertEqual(self._cat("alice/diffusion-fp8", "PR title"), "ready")
        self.assertEqual(
            self._cat("feat/nightly-lane", "Scheduled nightly build", source="schedule"),
            "ready",
        )

    def test_main_and_release_ordinary_are_merge(self) -> None:
        for branch in ("main", "release/v0.34.0", "release/v0.34.0rc1", "release/v0.14.0.post1"):
            with self.subTest(branch=branch):
                self.assertEqual(self._cat(branch, "PR #4521 merged", source="schedule"), "merge")

    def test_scheduled_nightly_and_release_b200_are_nightly(self) -> None:
        titles = (
            "Scheduled nightly build",
            "  Scheduled   nightly   build  \nbody that mentions weekly",
            "Scheduled Release B200 build",
        )
        for branch in ("main", "release/v0.34.0"):
            for title in titles:
                for source in ("schedule", "ui"):
                    with self.subTest(branch=branch, title=title, source=source):
                        self.assertEqual(self._cat(branch, title, source=source), "nightly")

    def test_scheduled_weekly_titles_are_weekly(self) -> None:
        titles = (
            "Scheduled weekly build",
            "Scheduled Non-Critical weekly build",
            "scheduled   non-critical   weekly   build",
        )
        for branch in ("main", "release/v0.34.0"):
            for title in titles:
                for source in ("schedule", "ui"):
                    with self.subTest(branch=branch, title=title, source=source):
                        self.assertEqual(self._cat(branch, title, source=source), "weekly")

    def test_main_pr_title_that_mentions_nightly_stays_merge(self) -> None:
        self.assertEqual(
            self._cat("main", "Add experimental AMD MI300 nightly lane", source="webhook"),
            "merge",
        )
        self.assertEqual(
            self._cat("release/v0.34.0", "Add experimental AMD MI300 nightly lane", source="ui"),
            "merge",
        )

    def test_loose_substrings_do_not_match(self) -> None:
        self.assertEqual(self._cat("main", "Scheduled weekly build(Release)", source="ui"), "merge")
        self.assertEqual(self._cat("main", "PR title\nScheduled nightly build"), "merge")
        self.assertEqual(self._cat("main", "AMD nightly", source="schedule"), "merge")

    def test_version_sidecar_stays_ready(self) -> None:
        self.assertEqual(
            self._cat("release/v0.34.0-docker", "Scheduled Release B200 build", source="ui"),
            "ready",
        )
        self.assertEqual(self._cat("backport/10687-release-v0.34.0", "Scheduled weekly build"), "ready")

    def test_message_less_sunday_schedule_is_merge(self) -> None:
        sunday = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(sunday.astimezone(pool.CST).weekday(), 6)
        self.assertEqual(self._cat("main", "", source="schedule", created_at=sunday), "merge")
        self.assertEqual(self._cat("release/v0.34.0", "", source="schedule", created_at=sunday), "merge")

    def test_category_subtitles_describe_release_branches(self) -> None:
        by_key = {key: sub for key, _label, sub in pool.CATEGORY_ORDER}
        self.assertIn("feature / PR", by_key["ready"])
        self.assertIn("release/v", by_key["merge"])
        self.assertIn("Release B200", by_key["nightly"])
        self.assertIn("Scheduled weekly", by_key["weekly"])
        self.assertNotIn("non-main", by_key["ready"])


if __name__ == "__main__":
    unittest.main()

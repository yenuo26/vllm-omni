"""Run the nightly Issue-section unit tests under pytest (no network)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = (
    Path(os.environ.get("NIGHTLY_REPORT_SKILL_DIR", "")).resolve()
    if os.environ.get("NIGHTLY_REPORT_SKILL_DIR")
    else REPO_ROOT / ".claude/skills/vllm-omni-test-report/scripts"
)


def _run(module: str) -> None:
    if str(SKILL_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SKILL_SCRIPTS))
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful(), result.failures + result.errors


def test_issue_close_confirm_suite() -> None:
    _run("test_issue_close_confirm")


def test_issue_section_suite() -> None:
    _run("test_issue_section")

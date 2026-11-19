"""Smoke tests for release-report evidence editors, PDF export, and the
performance/accuracy summary chapter.

Run from this directory:

    python3 test_release_report_features.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from perf_accuracy_summary import render_perf_accuracy_summary_section  # noqa: E402
from release_md_to_html import (  # noqa: E402
    convert_release_report_markdown,
    release_conclusion_widget_html,
    wrap_html_document,
)
from compose_full_report import preview_report_markdown  # noqa: E402

SKILL_DIR = SCRIPTS.parent
SAMPLE_LOG_DIR = Path("/workspace/nightly_jobs")


class EvidenceEditorsTests(unittest.TestCase):
    def test_widget_emits_an_evidence_column_and_paste_box(self) -> None:
        html = release_conclusion_widget_html()
        self.assertIn("Evidence", html)
        self.assertIn("conc-evidence", html)
        self.assertIn("Paste a link or image as evidence", html)
        self.assertIn("Add image", html)
        self.assertIn("data-evidence", html)
        self.assertIn("release-conclusion-evidence:", html)
        # One editor per check-item row (7 items + the Go/Rejected row is ignored)
        self.assertGreaterEqual(html.count("conc-evidence-input"), 7)
        self.assertIn('type="file"', html)
        self.assertIn('accept="image/*"', html)


class ExportPdfTests(unittest.TestCase):
    def test_wrapped_html_has_export_pdf_button_and_print_hook(self) -> None:
        html = wrap_html_document(title="Release preview", body_inner="<p>body</p>")
        self.assertIn('id="release-export-pdf"', html)
        self.assertIn("Export PDF", html)
        self.assertIn("window.print()", html)
        self.assertIn("release-printing", html)
        self.assertIn("@media print", html)
        self.assertIn("release-export-pdf-btn", html)


class PerfAccuracySummaryTests(unittest.TestCase):
    def test_missing_sources_use_honest_empty_notes(self) -> None:
        md = render_perf_accuracy_summary_section(
            log_dirs=[("H200", None), ("H800", None), ("A100", None), ("A3", None)],
            report_date="2026-11-19",
            buildkite_sources=[],
        )
        self.assertIn("## Performance and Accuracy Summary", md)
        self.assertIn("report date `2026-11-19`", md)
        self.assertIn("*No Buildkite builds were fetched for this chapter.*", md)
        for host in ("h200", "h800", "a100", "a3"):
            self.assertIn(f"*No `--log-dir-{host}` supplied", md)
        # Must not invent a pass/fail or a metric when nothing was provided.
        self.assertNotIn("IA_QTF1=", md)
        self.assertNotIn("throughput_qps=", md)

    def test_quotes_only_real_local_values_when_logs_exist(self) -> None:
        if not SAMPLE_LOG_DIR.is_dir():
            self.skipTest(f"sample log dir missing: {SAMPLE_LOG_DIR}")
        md = render_perf_accuracy_summary_section(
            log_dirs=[("H200", SAMPLE_LOG_DIR)],
            report_date="2026-11-18",
            buildkite_sources=[
                (
                    "B200 (scheduled release)",
                    [
                        {
                            "name": "e2e perf llama",
                            "state": "passed",
                            "url": "https://buildkite.com/vllm/vllm-omni/builds/1#j",
                            "build": "B200 #1",
                        },
                        {
                            "name": "accuracy omniinteract",
                            "state": "failed",
                            "url": "https://buildkite.com/vllm/vllm-omni/builds/1#k",
                            "build": "B200 #1",
                        },
                    ],
                )
            ],
        )
        self.assertIn("## Performance and Accuracy Summary", md)
        self.assertIn("e2e perf llama", md)
        self.assertIn("accuracy omniinteract", md)
        self.assertIn("https://buildkite.com/vllm/vllm-omni/builds/1#j", md)
        # Values taken from the sample result files (do not invent others).
        self.assertIn("IA_QTF1=0.3655", md)
        self.assertIn("omniinteract_aggregate_min_ia_qtf1=0.2000", md)
        self.assertIn("throughput_qps=0.4194", md)
        self.assertIn("request_throughput=0.0517", md)
        self.assertIn(
            "result_acc_fb09628e98b1f55d8d5b863e0eccfcdd11bff499_omniinteract_aggregate_20261118_035611.json",
            md,
        )


class PreviewWiringTests(unittest.TestCase):
    def test_preview_html_includes_chapter_evidence_and_export_pdf(self) -> None:
        md = preview_report_markdown(
            SKILL_DIR, stats_from="2026-11-01", stats_to="2026-11-19"
        )
        self.assertIn("## Performance and Accuracy Summary", md)
        self.assertIn("*No Buildkite builds were fetched for this chapter.*", md)
        html = wrap_html_document(
            title="vLLM-Omni Test Report preview",
            body_inner=convert_release_report_markdown(md),
        )
        self.assertIn("Performance and Accuracy Summary", html)
        self.assertIn("Paste a link or image as evidence", html)
        self.assertIn('id="release-export-pdf"', html)
        self.assertIn("Export PDF", html)


if __name__ == "__main__":
    unittest.main()

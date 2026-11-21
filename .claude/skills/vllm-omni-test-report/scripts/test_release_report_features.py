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
        self.assertIn("## Performance Summary", md)
        self.assertIn("## Accuracy Summary", md)
        self.assertNotIn("## Performance and Accuracy Summary", md)
        self.assertIn("report date `2026-11-19`", md)
        for host in ("h200", "h800", "a100", "a3"):
            self.assertIn(f"*No `--log-dir-{host}` supplied", md)
        # Must not invent a pass/fail or a metric when nothing was provided.
        self.assertNotIn("| IA_QTF1 |", md)
        self.assertNotIn("0.3655", md)
        self.assertNotIn("0.4194", md)

    def test_quotes_metric_values_in_columns_not_filenames(self) -> None:
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
        self.assertIn("## Performance Summary", md)
        self.assertIn("## Accuracy Summary", md)
        self.assertNotIn("## Performance and Accuracy Summary", md)
        # Buildkite has no metric payload — mention it, but do not tabulate job names.
        self.assertIn("does not include metric values", md)
        self.assertNotIn("e2e perf llama", md)
        self.assertNotIn("| Job | State | Build | Link |", md)
        # Values taken from the sample result files, as their own columns.
        self.assertIn("| IA_QTF1 |", md)
        self.assertIn("| Precision |", md)
        self.assertIn("| Recall |", md)
        self.assertIn("0.3655", md)
        self.assertIn("0.2000", md)  # baseline threshold
        self.assertIn("| throughput_qps |", md)
        self.assertIn("| request_throughput |", md)
        self.assertIn("| mean_ttft_ms |", md)
        self.assertIn("| latency_mean |", md)
        self.assertIn("0.4194", md)
        self.assertIn("0.0517", md)
        # Packed key=value cells and filename columns are gone.
        self.assertNotIn("IA_QTF1=0.3655", md)
        self.assertNotIn("throughput_qps=0.4194", md)
        self.assertNotIn("request_throughput=0.0517", md)
        self.assertNotIn(
            "result_acc_fb09628e98b1f55d8d5b863e0eccfcdd11bff499_omniinteract_aggregate_20261118_035611.json",
            md,
        )
        self.assertNotIn("diffusion_result_test_hunyuan_image3_it2i", md)
        self.assertNotIn("| Source |", md)


class PreviewWiringTests(unittest.TestCase):
    def test_preview_html_includes_chapter_evidence_and_export_pdf(self) -> None:
        md = preview_report_markdown(
            SKILL_DIR, stats_from="2026-11-01", stats_to="2026-11-19"
        )
        self.assertIn("## Performance Summary", md)
        self.assertIn("## Accuracy Summary", md)
        html = wrap_html_document(
            title="vLLM-Omni Test Report preview",
            body_inner=convert_release_report_markdown(md),
        )
        self.assertIn("Performance Summary", html)
        self.assertIn("Accuracy Summary", html)
        self.assertIn("Paste a link or image as evidence", html)
        self.assertIn('id="release-export-pdf"', html)
        self.assertIn("Export PDF", html)


class OmniReviewResultsTests(unittest.TestCase):
    def test_latest_dirs_prefer_newest_date(self) -> None:
        import tempfile
        from omni_review_results import (
            latest_vllm_omni_result_dirs,
            perf_acc_log_dirs_from_omni_review,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bk = root / "kanban/data/vllm_omni/buildkite_nightly_raw"
            local = root / "kanban/data/vllm_omni/local_nightly_raw"
            (bk / "20261118").mkdir(parents=True)
            (bk / "20261119").mkdir()
            (bk / "12330").mkdir()  # numeric build folder, must not win
            (local / "manual_20261117").mkdir(parents=True)
            (local / "manual_20261118").mkdir()
            found = latest_vllm_omni_result_dirs(root)
            self.assertEqual(found["buildkite_date"], "20261119")
            self.assertEqual(found["local_date"], "20261118")
            labels = [label for label, _ in perf_acc_log_dirs_from_omni_review(root)]
            self.assertEqual(labels, ["Buildkite"])

    def test_archive_json_is_tabulated_under_buildkite_and_local_hosts(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bk = Path(tmp) / "bk" / "tests" / "dfx" / "perf" / "results"
            loc = Path(tmp) / "local"
            bk.mkdir(parents=True)
            loc.mkdir()
            (bk / "result_test_sample.json").write_text(
                json.dumps(
                    {
                        "test_name": "bk_qwen_perf",
                        "model_id": "Qwen3-Omni",
                        "Hardware": "B200",
                        "date": "2026-11-18",
                        "request_throughput": 12.5,
                        "completed": 10,
                        "failed": 0,
                    }
                )
            )
            (loc / "result_acc_sample.json").write_text(
                json.dumps(
                    {
                        "test_name": "local_omniinteract",
                        "name": "1q1a",
                        "dataset": "omniinteract",
                        "Hardware": "L20X",
                        "date": "2026-11-18",
                        "accuracy": {"IA_QTF1": 0.3655, "Precision": 0.4, "Recall": 0.3},
                    }
                )
            )
            md = render_perf_accuracy_summary_section(
                log_dirs=[("Buildkite", Path(tmp) / "bk"), ("Local", loc)],
                report_date="2026-11-19",
                buildkite_sources=[],
                source_notes=["Buildkite `20261119`", "Local `20261118`"],
            )
            self.assertIn("Sources: Buildkite `20261119`; Local `20261118`", md)
            self.assertIn("| Buildkite |", md)
            self.assertIn("bk_qwen_perf", md)
            self.assertIn("12.5", md)
            self.assertIn("| Local |", md)
            self.assertIn("0.3655", md)
            self.assertNotIn("does not include metric values", md)


if __name__ == "__main__":
    unittest.main()

"""Smoke tests for release-report evidence editors and the
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
    def test_critical_row_is_manual_and_unfilled_when_not_scanned(self) -> None:
        html = release_conclusion_widget_html(
            l2_l3_row_ok=None,
            di_row_ok=None,
            critical_row_ok=None,
        )
        # Locate the "No remaining critical issues" row.
        marker = "No remaining critical issues"
        idx = html.index(marker)
        # Walk back to the opening <tr ...> for this row.
        tr_start = html.rfind("<tr ", 0, idx)
        tr_end = html.index("</tr>", idx) + len("</tr>")
        row = html[tr_start:tr_end]
        self.assertIn('data-conc-auto="0"', row)
        self.assertNotIn("conc-auto-hint", row)
        self.assertIn('class="conc-btns"', row)  # no "conc-auto" class on the button group
        self.assertNotIn('class="conc-btns conc-auto"', row)
        # Neither Pass nor Fail lit — unfilled until the operator picks.
        import re
        pass_btn = re.search(r'class="conc-btn conc-pass ([^"]*)"', row)
        fail_btn = re.search(r'class="conc-btn conc-fail ([^"]*)"', row)
        self.assertIsNotNone(pass_btn)
        self.assertIsNotNone(fail_btn)
        self.assertNotIn("is-on", pass_btn.group(1))
        self.assertNotIn("is-on", fail_btn.group(1))

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


class NoExportControlTests(unittest.TestCase):
    def test_wrapped_html_has_no_export_or_print_control(self) -> None:
        html = wrap_html_document(title="Release preview", body_inner="<p>body</p>")
        lowered = html.lower()
        self.assertNotIn('id="release-export-pdf"', html)
        self.assertNotIn("Export PDF", html)
        self.assertNotIn("window.print()", html)
        self.assertNotIn("release-printing", html)
        self.assertNotIn("release-export-pdf-btn", html)
        self.assertNotIn("@media print", html)
        self.assertNotIn("brand-actions", html)
        self.assertNotIn(">print<", lowered)
        self.assertNotIn(">export<", lowered)


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


class PerformanceSchemaTests(unittest.TestCase):
    def test_performance_schemas_render_as_separate_tables(self) -> None:
        import json
        import tempfile

        fixtures = {
            "result_test_text.json": {
                "test_name": "text_serving_case",
                "model_id": "Qwen/Qwen3-4B",
                "Hardware": "H100",
                "date": "20261121-120000",
                "completed": 8,
                "failed": 0,
                "request_throughput": 1.25,
                "output_throughput": 40.0,
                "mean_ttft_ms": 120.5,
                "mean_tpot_ms": 6.5,
                "mean_itl_ms": 6.25,
                "mean_e2el_ms": 800.0,
                "p99_ttft_ms": 200.0,
                "duration": 12.0,
                "audio_throughput": 0,
                "image_throughput": 0,
                "video_throughput": 0,
                "random_input_len": 100,
            },
            "result_test_omni_audio.json": {
                "test_name": "omni_chunk_case",
                "model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "Hardware": "H100",
                "date": "20261121-115826",
                "completed": 100,
                "failed": 0,
                "request_throughput": 0.9671681213117175,
                "mean_ttft_ms": 409.21592249069363,
                "mean_tpot_ms": 5.971110715681329,
                "mean_itl_ms": 5.97111071568133,
                "audio_throughput": 26.06564027420841,
                "mean_audio_ttfp_ms": 524.8251610761508,
                "p99_audio_ttfp_ms": 822.0186460111292,
                "mean_audio_rtf": 0.1317546776212871,
                "duration": 103.39464028691873,
                "baseline": {
                    "H100": {"request_throughput": 0.9562, "mean_ttft_ms": 460.0396},
                    "A3": {"request_throughput": 0.5098},
                },
            },
            "result_test_tts.json": {
                "test_name": "seed_tts_case",
                "model_id": "openbmb/MiniCPM-o-4_5",
                "Hardware": "910",
                "date": "20261121-110604",
                "completed": 256,
                "failed": 0,
                "request_throughput": 5.307059388166985,
                "audio_throughput": 22.38749833777067,
                "mean_ttft_ms": 174.96904570975857,
                "p99_ttft_ms": 488.2261612365253,
                "mean_audio_ttfp_ms": 1716.163834147892,
                "p99_audio_ttfp_ms": 2564.477450452978,
                "mean_audio_rtf": 0.7298888217948072,
                "p99_audio_rtf": 1.2008500398326454,
                "mean_tpot_ms": 0,
                "image_throughput": 0,
                "duration": 48.237636189034674,
            },
            "result_test_image.json": {
                "test_name": "image_case",
                "model_id": "tencent/HunyuanImage-3.0-Instruct-Distil",
                "Hardware": "H100",
                "date": "20261121-111228",
                "completed": 96,
                "failed": 0,
                "request_throughput": 0.2537612332556933,
                "image_throughput": 0.2537612332556933,
                "mean_denoise_step_latency_ms": 489.79824452665827,
                "mean_peak_memory_mb": 55502.333333333336,
                "mean_e2el_ms": 58353.102476180844,
                "video_throughput": 0,
                "audio_throughput": 0,
                "mean_ttft_ms": 0,
                "duration": 378.30837582377717,
                "baseline": {"H100": {"mean_e2el_ms": 58151.52815403417}},
            },
            "result_test_video.json": {
                "test_name": "video_case",
                "model_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
                "Hardware": "H100",
                "date": "20261121-110754",
                "completed": 10,
                "failed": 0,
                "request_throughput": 0.20646337642470264,
                "video_throughput": 0.28388714258396613,
                "mean_video_rtf": 2.835084208846092,
                "mean_video_generation_ms": 3898.240787163377,
                "mean_peak_memory_mb": 43325.8,
                "mean_e2el_ms": 4842.333703674376,
                "image_throughput": 0,
                "mean_denoise_step_latency_ms": 0,
                "duration": 48.434740209952,
            },
            "result_test_lingbot_video.json": {
                "test_name": "lingbot_video_case",
                "model_id": "robbyant/lingbot-video-dense-1.3b",
                "Hardware": "H100",
                "date": "20261121-105423",
                "completed": 3,
                "failed": 0,
                "video_throughput": 0.1869749021583839,
                "mean_video_rtf": 0.5563927193482716,
                "mean_video_generation_ms": 208.64726975560188,
                "mean_e2el_ms": 2005.1397066563368,
                "duration": 6.016850320622325,
            },
            "result_test_realtime.json": {
                "test_name": "lingbot_realtime_case",
                "model_id": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
                "Hardware": "910B",
                "date": "20261121-110710",
                "completed": 3,
                "failed": 0,
                "request_throughput": 0.008241,
                "mean_e2el_ms": 121338.58955775698,
                "mean_video_ttfc_ms": 4012.9338813324766,
                "mean_video_fps": 3.9312533627855495,
                "mean_video_rtf": 3.0524618088324367,
                "mean_video_steady_interval_ms": 3021.8125894816235,
                "min_video_fps": 3.93045403578819,
                "duration": 364.0168080367148,
                "baseline": {"A2": {"mean_video_ttfc_ms": 6332.3904, "request_throughput": 0.0071}},
            },
            "result_test_duplex.json": {
                "test_name": "duplex_case",
                "model_id": "openbmb/MiniCPM-o-4_5",
                "Hardware": "H100",
                "date": "20261121-110036",
                "completed": 66,
                "failed": 0,
                "request_throughput": 0.16086710984868532,
                "audio_throughput": 1.8726881490748775,
                "mean_ttft_ms": 565.9387038975227,
                "mean_audio_rtf": 0.8660640406504065,
                "mean_duplex_response_ttft_ms": 565.939,
                "mean_duplex_stall_ratio": 0.341108,
                "p99_duplex_stall_ratio": 4.602397,
                "duration": 410.27653236314654,
                "input_lens": [1, 2, 3],
                "ttfts": [1.0, 2.0],
            },
            "diffusion_result_test_hunyuan.json": [
                {
                    "test_name": "diffusion_case",
                    "Hardware": "910",
                    "timestamp": "20261121-105919",
                    "completed": 10,
                    "failed": 0,
                    "server_params": {"model": "tencent/HunyuanImage-3.0-Instruct"},
                    "throughput_qps": 0.18737164038779,
                    "latency_mean": 5.336798547685612,
                    "latency_p99": 5.43519172689761,
                    "peak_memory_mb_mean": 47214.0,
                    "duration": 53.369869524030946,
                    "baseline": {
                        "H200": {"throughput_qps": 0.465},
                        "A3": {"throughput_qps": 0.1855},
                    },
                }
            ],
            "result_test_no_metrics.json": {
                "test_name": "empty_perf",
                "model_id": "NoPerf",
                "Hardware": "H100",
                "completed": 0,
                "failed": 0,
                "random_input_len": 128,
                "num_prompts": 4,
                "audio_throughput": 0,
                "image_throughput": 0,
                "video_throughput": 0,
            },
            "benchmark_results_future.json": {
                "test_name": "future_perf",
                "model_id": "FutureModel",
                "Hardware": "H100",
                "date": "20261121-130000",
                "completed": 2,
                "failed": 0,
                "custom_step_latency_ms": 12.5,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload in fixtures.items():
                (root / name).write_text(json.dumps(payload))
            md = render_perf_accuracy_summary_section(
                log_dirs=[("Buildkite", root)],
                report_date="2026-11-21",
                buildkite_sources=[],
            )
        perf = md[: md.index("## Accuracy Summary")]
        sections: dict[str, str] = {}
        for part in perf.split("\n### "):
            if part.startswith("### "):
                part = part[4:]
            title, _, body = part.partition("\n")
            sections[title.strip()] = body

        self.assertNotIn("Serving / omni", perf)
        text = sections["Text serving"]
        self.assertIn("text_serving_case", text)
        self.assertIn("1.250", text)
        self.assertIn("6.500", text)
        self.assertIn("mean_itl_ms", text)
        self.assertNotIn("audio_throughput", text)
        self.assertNotIn("random_input_len", perf)

        omni = sections["Omni text and audio"]
        self.assertIn("mean_tpot_ms", omni)
        self.assertIn("5.971", omni)
        self.assertIn("mean_audio_rtf", omni)
        self.assertIn("0.1318", omni)
        self.assertIn("p99_audio_ttfp_ms", omni)
        self.assertIn("0.9562", omni)  # H100 baseline, not the A3 0.5098 profile
        self.assertNotIn("0.5098", omni)

        audio = sections["Audio / TTS"]
        self.assertIn("22.387", audio)
        self.assertIn("1716.16", audio)
        self.assertIn("2564.48", audio)
        self.assertIn("0.7299", audio)
        self.assertNotIn("mean_tpot_ms", audio)
        self.assertNotIn("image_throughput", audio)

        image = sections["Image generation"]
        self.assertIn("0.2538", image)
        self.assertIn("489.80", image)
        self.assertIn("55502.33", image)
        self.assertIn("58353.10", image)
        self.assertIn("baseline_mean_e2el_ms", image)
        self.assertNotIn("mean_ttft_ms", image)
        self.assertNotIn("video_throughput", image)
        self.assertNotIn("audio_throughput", image)

        video = sections["Video generation"]
        self.assertIn("video_case", video)
        self.assertIn("lingbot_video_case", video)
        self.assertIn("2.835", video)
        self.assertIn("3898.24", video)
        self.assertIn("208.65", video)
        self.assertNotIn("image_throughput", video)
        self.assertNotIn("mean_video_ttfc_ms", video)
        self.assertNotIn("mean_denoise_step_latency_ms", video)

        realtime = sections["Lingbot-World realtime"]
        self.assertIn("4012.93", realtime)
        self.assertIn("3.931", realtime)
        self.assertIn("3021.81", realtime)
        self.assertIn("min_video_fps", realtime)
        self.assertNotIn("6332.39", realtime)  # A2 baseline does not match Hardware 910B
        self.assertNotIn("mean_video_generation_ms", realtime)

        duplex = sections["Omni duplex eval"]
        self.assertIn("0.3411", duplex)
        self.assertIn("4.602", duplex)
        self.assertIn("565.94", duplex)
        self.assertNotIn("input_lens", duplex)

        diffusion = sections["Diffusion"]
        self.assertIn("diffusion_case", diffusion)
        self.assertIn("0.1874", diffusion)
        self.assertIn("5.337", diffusion)
        self.assertIn("47214.00", diffusion)
        self.assertNotIn("0.4650", diffusion)  # H200 baseline, hardware is 910
        self.assertNotIn("request_throughput", diffusion)

        self.assertIn("12.500", sections["Performance (custom_step_latency_ms)"])
        self.assertIn("*No performance metrics in `result_test_no_metrics.json`.*", perf)
        self.assertNotIn("empty_perf", perf)
        self.assertNotIn("NoPerf", perf)
        self.assertEqual(perf.count("result_test_"), 1)


class AccuracySchemaTests(unittest.TestCase):
    def test_accuracy_schemas_render_as_separate_tables(self) -> None:
        import json
        import tempfile

        fixtures = {
            "result_acc_video_similarity.json": {
                "model_id": "MiniMaxAI/MiniMax-H3",
                "date": "20261121-111246",
                "Hardware": "H100",
                "ssim": 0.98057,
                "psnr": 42.644149,
                "label": "minimax_h3_ref2va_official_reference",
                "dataset": "video_similarity",
                "latency_ms": 12.5,
                "ssim_threshold": 0.93,
            },
            "result_acc_image_similarity.json": {
                "model_id": "Qwen/Qwen-Image-2.1",
                "date": "20261121-104736",
                "Hardware": "H100",
                "ssim": 0.9999962449073792,
                "psnr": "inf",
                "alpha_mae": 0.0,
                "case": "t2i",
                "dataset": "image_similarity",
                "calibration_only": False,
                "psnr_threshold": 40.0,
            },
            "result_acc_daily_omni.json": {
                "model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "date": "20261121-111453",
                "Hardware": "910",
                "dataset": "daily_omni",
                "request_throughput": 1.008391718851022,
                "daily_omni_accuracy": 0.722639933166249,
                "daily_omni_correct": 865,
                "daily_omni_evaluated": 1197,
                "daily_omni_per_task_accuracy": {"Reasoning": 0.8342857142857143},
                "daily_omni_per_task": {"Reasoning": {"correct": 146, "total": 175}},
                "daily_omni_per_task_accuracy_github_style": {"Reasoning": 0.8342857142857143},
                "daily_omni_per_category_accuracy": {"Entertainment": 0.7281},
                "daily_omni_per_category": {"Entertainment": {"correct": 158, "total": 217}},
                "daily_omni_per_duration_accuracy": {"30s": 0.7295},
                "daily_omni_per_duration": {"30s": {"correct": 472, "total": 647}},
                "baseline": {"910": {"min_daily_omni_accuracy": 0.69}},
            },
            "result_acc_seed_tts.json": {
                "model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "dataset": "seed_tts",
                "Hardware": "910",
                "date": "20261121-113528",
                "request_throughput": 9.9,
                "seed_tts_content_metric": "wer",
                "seed_tts_content_error_mean": 0.03697131681155254,
                "seed_tts_content_evaluated": 1088,
                "seed_tts_sim_mean": 0.8223071049022324,
                "seed_tts_sim_evaluated": 1088,
                "seed_tts_utmos_mean": None,
                "seed_tts_utmos_evaluated": 0,
                "baseline": {
                    "910": {
                        "max_seed_tts_mean_wer": 0.05,
                        "min_seed_tts_mean_sim": 0.75,
                    }
                },
            },
            "result_acc_omniinteract.json": {
                "test_name": "test_qwen3_omni_duplex_omniinteract",
                "name": "1q1a_math",
                "dataset": "omniinteract",
                "Hardware": "H100",
                "date": "20261121-114041",
                "accuracy": {
                    "status": "ok",
                    "IA_QTF1": 0.31006848305035795,
                    "Precision": 0.3152055758318858,
                    "Recall": 0.3050961496020795,
                    "Global_TP": 28.538120364103985,
                    "evaluated": 51,
                    "total": 60,
                    "summary": {"IA_QTF1": 0.01},
                },
                "baseline": {"min_IA_QTF1": 0.2},
            },
            "result_acc_duplex.json": {
                "test_name": "test_minicpmo_4_5_omni_duplex_eval",
                "dataset": "omni_duplex_eval",
                "Hardware": "H100",
                "date": "20261121-110036",
                "accuracy": {
                    "status": "completed",
                    "evaluated": 66,
                    "total": 66,
                    "rtd": {
                        "mean_content_score": 1.8166666666666667,
                        "mean_avg_temporal_score": 1.8683333333333332,
                        "by_task": {
                            "RTD_OCR": {
                                "mean_content_score": 1.8,
                                "mean_avg_temporal_score": 2.2,
                            }
                        },
                    },
                    "pr": {
                        "mean_all_success": 0.3333333333333333,
                        "by_task": {"correction": 0.8461538461538461},
                    },
                },
                "baseline": {
                    "min_rtd_mean_content_score": 1.8,
                    "min_rtd_mean_temporal_score": 1.4,
                    "min_pr_mean_all_success": 0.28,
                },
            },
            "result_acc_vbench.json": {
                "model_id": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
                "dataset": "lingbot_world_vbench",
                "Hardware": "A2",
                "date": "20261121-113207",
                "boardwalk-42_subject_consistency": 0.9539171869263929,
                "boardwalk-42_dynamic_degree": 0.0,
                "street-42_subject_consistency": 0.9109,
                "measurement_report": {"suite": "vbench", "passed": True},
            },
            "result_acc_wbench.json": {
                "model_id": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
                "dataset": "lingbot_world_wbench",
                "Hardware": "H100",
                "date": "20261121-112705",
                "boardwalk-42_aesthetic_quality": 0.680510188738505,
                "boardwalk-42_hpsv3_quality": 0.8669641784752158,
            },
            "result_acc_precision.json": {
                "model_id": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
                "dataset": "lingbot_world_precision",
                "Hardware": "H100",
                "date": "20261121-112132",
                "output_decoded_sample_0": -0.78515625,
                "measurement_report": {
                    "suite": "precision",
                    "passed": True,
                    "sample_comparison": {
                        "sample_count": 64,
                        "max_abs": {"output/decoded": 0.0},
                    },
                    "tolerance": {"atol": 0.04, "rtol": 0.0},
                },
                "baseline": {"H100": {"output_decoded_sample_0": -0.78515625}},
            },
            "result_acc_no_metrics.json": {
                "model_id": "NoScores",
                "dataset": "plain",
                "request_throughput": 3.5,
            },
            "result_acc_future.json": {
                "model_id": "FutureBench",
                "dataset": "future_bench",
                "accuracy": {"wer": 0.12, "sim": 0.8},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload in fixtures.items():
                (root / name).write_text(json.dumps(payload))
            md = render_perf_accuracy_summary_section(
                log_dirs=[("Buildkite", root)],
                report_date="2026-11-21",
                buildkite_sources=[],
            )
        acc = md[md.index("## Accuracy Summary") :]
        sections = {}
        for part in acc.split("\n### "):
            if part.startswith("### "):
                part = part[4:]
            title, _, body = part.partition("\n")
            sections[title.strip()] = body

        similarity = sections["Image and video similarity"]
        self.assertIn("0.9806", similarity)
        self.assertIn("42.644", similarity)
        self.assertIn("inf", similarity)
        self.assertIn("0.0000", similarity)
        self.assertIn("t2i", similarity)
        self.assertNotIn("IA_QTF1", similarity)
        self.assertNotIn("latency_ms", acc)
        self.assertNotIn("ssim_threshold", acc)
        self.assertNotIn("psnr_threshold", acc)
        self.assertNotIn("calibration_only", acc)
        self.assertNotIn("request_throughput", md)
        self.assertNotIn("1.0084", md)

        daily = sections["Daily Omni"]
        self.assertIn("0.7226", daily)
        self.assertIn("865", daily)
        self.assertIn("0.6900", daily)
        self.assertIn("| Task |", sections["Daily Omni by task"])
        self.assertIn("Reasoning", sections["Daily Omni by task"])
        self.assertIn("0.8343", sections["Daily Omni by task"])
        self.assertNotIn("github style", sections["Daily Omni by task"])
        self.assertIn("| Category |", sections["Daily Omni by category"])
        self.assertIn("Entertainment", sections["Daily Omni by category"])
        self.assertIn("| Duration |", sections["Daily Omni by duration"])
        self.assertIn("30s", sections["Daily Omni by duration"])

        seed = sections["Seed-TTS"]
        self.assertIn("wer", seed)
        self.assertIn("0.0370", seed)
        self.assertIn("0.8223", seed)
        self.assertIn("0.0500", seed)
        self.assertIn("0.7500", seed)
        self.assertNotIn("utmos", seed)

        interact = sections["OmniInteract"]
        self.assertIn("0.3101", interact)
        self.assertIn("0.2000", interact)
        self.assertIn("1q1a_math", interact)
        self.assertNotIn("0.0100", interact)  # top-level score wins over summary

        duplex = sections["Omni duplex eval"]
        self.assertIn("1.817", duplex)
        self.assertIn("1.868", duplex)
        self.assertIn("0.3333", duplex)
        self.assertIn("1.800", duplex)
        self.assertIn("1.400", duplex)
        self.assertIn("0.2800", duplex)
        self.assertIn("RTD_OCR", sections["Omni duplex RTD by task"])
        self.assertIn("2.200", sections["Omni duplex RTD by task"])
        self.assertIn("correction", sections["Omni duplex PR by task"])
        self.assertIn("0.8462", sections["Omni duplex PR by task"])

        vbench = sections["Lingbot-World VBench"]
        wbench = sections["Lingbot-World WBench"]
        self.assertIn("subject_consistency", vbench)
        self.assertIn("boardwalk-42", vbench)
        self.assertIn("0.9539", vbench)
        self.assertIn("0.0000", vbench)
        self.assertNotIn("hpsv3_quality", vbench)
        self.assertIn("hpsv3_quality", wbench)
        self.assertIn("0.8670", wbench)
        self.assertNotIn("subject_consistency", wbench)
        self.assertNotIn("| passed |", vbench)

        precision = sections["Lingbot-World precision"]
        self.assertIn("| passed |", precision)
        self.assertIn("true", precision)
        self.assertIn("max_abs_output_decoded", precision)
        self.assertIn("0.0000", precision)
        self.assertIn("0.0400", precision)
        self.assertNotIn("output_decoded_sample", acc)

        future_titles = [title for title in sections if title.startswith("Accuracy (")]
        self.assertEqual(future_titles, ["Accuracy (sim, wer)"])
        self.assertIn("0.1200", sections["Accuracy (sim, wer)"])
        self.assertIn("0.8000", sections["Accuracy (sim, wer)"])
        self.assertNotIn("wer", interact)

        self.assertIn("*No accuracy metrics in `result_acc_no_metrics.json`.*", acc)
        self.assertEqual(acc.count("result_acc_"), 1)
        self.assertNotIn("NoScores", acc)


class PreviewWiringTests(unittest.TestCase):
    def test_preview_html_includes_chapter_and_evidence(self) -> None:
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
        self.assertNotIn('id="release-export-pdf"', html)
        self.assertNotIn("Export PDF", html)
        self.assertNotIn("window.print()", html)


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

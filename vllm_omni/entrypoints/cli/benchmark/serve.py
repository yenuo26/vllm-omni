import argparse

from vllm.benchmarks.serve import add_cli_args

from vllm_omni.benchmarks.serve import main
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


class OmniBenchmarkServingSubcommand(OmniBenchmarkSubcommandBase):
    """The `serve` subcommand for vllm bench."""

    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)
        dest_mapping = {
            "--percentile-metrics": "percentile_metrics",
            "--random-mm-limit-mm-per-prompt": "random_mm_limit_mm_per_prompt",
            "--random-mm-bucket-config": "random_mm_bucket_config",
        }

        for arg_name, dest_name in dest_mapping.items():
            for action in parser._actions:
                if arg_name in action.option_strings or action.dest == dest_name:
                    if arg_name == "percentile-metrics":
                        action.help = (
                            "Comma-separated list of selected metrics to report percentiles.\n"
                            "This argument specifies the metrics to report percentiles.\n"
                            'Allowed metric names are "ttft", "tpot", "itl", "e2el", '
                            '"audio_ttfp", "audio_rtf".'
                        )
                    elif arg_name == "random-mm-limit-mm-per-prompt":
                        action.help = (
                            "Per-modality hard caps for items attached per request, e.g.\n"
                            '\'{"image": 3, "video": 0, "audio": 1}\'. The sampled per-request item\n'
                            "count is clamped to the sum of these limits. When a modality\n"
                            "reaches its cap, its buckets are excluded and probabilities are\n"
                            "renormalized."
                        )
                    elif arg_name == "random-mm-bucket-config":
                        action.help = (
                            "The bucket config is a dictionary mapping a multimodal item\n"
                            "sampling configuration to a probability.\n"
                            "Currently allows for 3 modalities: audio, images and videos.\n"
                            "An bucket key is a tuple of (height, width, num_frames).\n"
                            "The value is the probability of sampling that specific item.\n"
                            "Example:\n"
                            "  --random-mm-bucket-config "
                            '"{(256, 256, 1): 0.5, (720, 1280, 16): 0.4, (0, 1, 5): 0.10}"\n'
                            "First item: images with resolution 256x256 w.p. 0.5\n"
                            "Second item: videos with resolution 720x1280 and 16 frames\n"
                            "Third item: audios with 1s duration and 5 channels w.p. 0.1\n"
                            "OBS.: If the probabilities do not sum to 1, they are normalized."
                        )
                    break

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

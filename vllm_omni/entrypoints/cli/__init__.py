"""CLI helpers for vLLM-Omni entrypoints."""

from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

from .serve import OmniServeCommand

__all__ = ["OmniServeCommand", "OmniBenchmarkServingSubcommand"]

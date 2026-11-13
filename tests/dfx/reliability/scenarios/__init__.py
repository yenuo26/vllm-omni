# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault scenarios. Each class owns one fault's steps; model tests call ``run``."""

from tests.dfx.reliability.scenarios.launch import FullCrashAtLaunch, GpuOomAtLaunch
from tests.dfx.reliability.scenarios.oom import (
    GpuOomConcurrent,
    GpuOomErrorContract,
    GpuOomLarge,
    GpuOomRecovery,
    GpuOomUnderLoad,
)
from tests.dfx.reliability.scenarios.process_kill import (
    FullCrashAtExit,
    PartialCrashAtExit,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
)
from tests.dfx.reliability.scenarios.replica import (
    ReplicaCrashAtExit,
    ReplicaCrashIdle,
    ReplicaCrashWithLoad,
)

__all__ = [
    "FullCrashAtExit",
    "FullCrashAtLaunch",
    "GpuOomAtLaunch",
    "GpuOomConcurrent",
    "GpuOomErrorContract",
    "GpuOomLarge",
    "GpuOomRecovery",
    "GpuOomUnderLoad",
    "PartialCrashAtExit",
    "PartialStageIsolation",
    "ProcessKillServeRootNoLoad",
    "ProcessKillServeRootWithLoad",
    "ProcessKillTreeNoLoad",
    "ProcessKillTreeWithLoad",
    "ProcessKillWorkerNoLoad",
    "ProcessKillWorkerWithLoad",
    "ReplicaCrashAtExit",
    "ReplicaCrashIdle",
    "ReplicaCrashWithLoad",
]

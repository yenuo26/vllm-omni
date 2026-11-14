# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault scenarios. Each class owns one fault's steps; model tests call ``run``."""

from tests.dfx.reliability.scenarios.launch import FullCrashAtLaunch, GpuOomAtLaunch, PartialCrashAtLaunch
from tests.dfx.reliability.scenarios.oom import (
    GpuOomConcurrent,
    GpuOomErrorContract,
    GpuOomLarge,
    GpuOomRecovery,
    GpuOomUnderLoad,
)
from tests.dfx.reliability.scenarios.process_kill import (
    GracefulShutdown,
    PartialCrashAtExit,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
    ReplicaGracefulShutdown,
)
from tests.dfx.reliability.scenarios.replica import (
    ReplicaCrashAtExit,
    ReplicaCrashAtLaunch,
    ReplicaCrashIdle,
    ReplicaCrashWithLoad,
)

__all__ = [
    "FullCrashAtLaunch",
    "GracefulShutdown",
    "ReplicaGracefulShutdown",
    "GpuOomAtLaunch",
    "GpuOomConcurrent",
    "GpuOomErrorContract",
    "GpuOomLarge",
    "GpuOomRecovery",
    "GpuOomUnderLoad",
    "PartialCrashAtExit",
    "PartialCrashAtLaunch",
    "PartialStageIsolation",
    "ProcessKillServeRootNoLoad",
    "ProcessKillServeRootWithLoad",
    "ProcessKillTreeNoLoad",
    "ProcessKillTreeWithLoad",
    "ProcessKillWorkerNoLoad",
    "ProcessKillWorkerWithLoad",
    "ReplicaCrashAtExit",
    "ReplicaCrashAtLaunch",
    "ReplicaCrashIdle",
    "ReplicaCrashWithLoad",
]

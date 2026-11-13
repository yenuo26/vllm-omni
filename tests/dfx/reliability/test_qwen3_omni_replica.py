# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni stage-replica reliability.

Uses the existing 4-GPU CI deploy: one thinker, and three replicas each of
stage 1 and stage 2. A test kills one stage-1 replica. The other replicas
still accept a request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dfx.reliability.helpers import scenario_params
from tests.dfx.reliability.scenarios import ReplicaCrashAtExit, ReplicaCrashIdle, ReplicaCrashWithLoad
from tests.dfx.reliability.test_qwen3_omni import Qwen3Omni
from tests.helpers.mark import hardware_marks
from tests.helpers.stage_config import get_deploy_config_path

_DEPLOY = Path(get_deploy_config_path("ci/qwen3_omni_moe_multi_replicas_4gpu.yaml"))


class Qwen3OmniReplica(Qwen3Omni):
    num_cards = 4
    deploy_dir = _DEPLOY.parent
    replica_scenarios = [
        {
            "test_name": "qwen3_omni_reliability_replica",
            "server_params": {
                "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "stage_config_name": _DEPLOY.name,
            },
        }
    ]
    # devices "1,2,3": replica 1 is the second card.
    bad_replica_pattern = "StageEngineCoreProc_stage1_replica1"
    bad_replica_devices = "2"


MODEL = Qwen3OmniReplica()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, ReplicaCrashIdle)
def test_replica_crash_idle(omni_server_function, online_client_function) -> None:
    """Kill one idle stage-1 replica. Another replica still accepts a request."""
    ReplicaCrashIdle().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ReplicaCrashWithLoad)
def test_replica_crash_with_load(omni_server_function, online_client_function) -> None:
    """Kill one stage-1 replica under load. In-flight work fails, and another replica still serves."""
    ReplicaCrashWithLoad().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ReplicaCrashAtExit)
def test_replica_crash_at_exit(omni_server_function, online_client_function) -> None:
    """After one stage-1 replica dies, shutdown finishes or escalates, and a new server can start."""
    ReplicaCrashAtExit().run(MODEL, omni_server_function, online_client_function)

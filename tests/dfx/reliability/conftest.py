# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pytest fixtures for reliability tests.

Model tests call a fault scenario directly, so there is no case-lookup fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.dfx.reliability.helpers import FaultInjector


@pytest.fixture
def fault_injector(request: pytest.FixtureRequest) -> FaultInjector:
    """Indirect only: ``request.param`` must be a ``FaultInjector`` callable."""
    return request.param


@pytest.fixture
def omni_server_after_fault(omni_server: Any, fault_injector: FaultInjector):
    """After ``omni_server`` is up, run ``fault_injector(omni_server)``, then yield the server."""
    fault_injector(omni_server)
    yield omni_server


@pytest.fixture
def omni_server_after_fault_function(omni_server_function: Any, fault_injector: FaultInjector):
    """Inject fault after function-scoped server startup, then yield server."""
    fault_injector(omni_server_function)
    yield omni_server_function


@pytest.fixture(scope="class")
def omni_server_shared(request: pytest.FixtureRequest, run_level: str):
    """Class-scoped server for OOM cases that leave the API process up."""
    from tests.dfx.reliability.helpers import SharedServer
    from tests.helpers.fixtures.runtime import omni_fixture_lock
    from tests.helpers.runtime import iter_omni_server

    generator = iter_omni_server(request, run_level, omni_fixture_lock)
    server = next(generator)
    slot = SharedServer(server)
    try:
        yield slot
    finally:
        if slot.server is not server:
            slot.server.__exit__(None, None, None)
        try:
            next(generator)
        except StopIteration:
            pass

"""Tests for blop.bayesian.kernels module.

Tests the public API of LatentKernel - a Matérn kernel with learned affine
transformation using SO(N) parameterization for orthogonal rotations.
"""

import pytest
from bluesky.run_engine import RunEngine

from blop.ax.agent import Agent
from blop.ax.dof import RangeDOF
from blop.ax.generation import get_generation_strategy
from blop.ax.objective import Objective
from blop.evaluation.test_functions import TestFunctionEvaluation

from ..conftest import MovableSignal
from ..test_plans import _collect_optimize_events


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_optimize(RE):

    dofs = [
        RangeDOF(actuator=MovableSignal("x1"), bounds=(-2.0, 2.0), parameter_type="float"),
        RangeDOF(actuator=MovableSignal("x2"), bounds=(-2.0, 2.0), parameter_type="float"),
    ]

    objectives = [
        Objective(name="fitness", minimize=False),
    ]

    agent = Agent(
        sensors=[],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=TestFunctionEvaluation(),
        name="test",
        description="Optimization on a test function",
        experiment_type="demo",
        generation_strategy=get_generation_strategy("beamline"),
    )

    callback, events = _collect_optimize_events()
    RE.subscribe(callback)
    try:
        RE(agent.optimize(20))
    except RuntimeError:
        ...
    finally:
        RE.unsubscribe(callback)

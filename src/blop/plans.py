from collections.abc import Generator, Mapping, Sequence
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.protocols import Movable
from bluesky.run_engine import Msg
from ophyd import Signal  # type: ignore[import-untyped]

from .dofs import DOF


def list_scan_with_delay(*args: Any, delay: float = 0, **kwargs: Any) -> Generator[Msg, None, str]:
    "Accepts all the normal 'scan' parameters, plus an optional delay."

    def one_nd_step_with_delay(
        detectors: Sequence[Signal], step: Mapping[Movable, Any], pos_cache: Mapping[Movable, Any]
    ) -> Generator[Msg, None, None]:
        "This is a copy of bluesky.plan_stubs.one_nd_step with a sleep added."
        motors = step.keys()
        yield from bps.move_per_step(step, pos_cache)
        yield from bps.sleep(delay)
        yield from bps.trigger_and_read(list(detectors) + list(motors))

    kwargs.setdefault("per_step", one_nd_step_with_delay)
    uid = yield from bp.list_scan(*args, **kwargs)
    return uid


def default_acquisition_plan(
    dofs: Sequence[DOF], inputs: Mapping[str, Sequence[Any]], dets: Sequence[Signal], **kwargs: Any
) -> Generator[Msg, None, str]:
    """
    Parameters
    ----------
    x : list of DOFs or DOFList
        A list of DOFs
    inputs: dict
        A dict of a list of inputs per dof, keyed by dof.name
    dets: list
        A list of detectors to trigger
    """
    delay = kwargs.get("delay", 0)
    args = []
    for dof in dofs:
        args.append(dof.device)
        args.append(inputs[dof.name])

    uid = yield from list_scan_with_delay(dets, *args, delay=delay)
    return uid

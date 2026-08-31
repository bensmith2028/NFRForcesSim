"""Rocker kinematics -- converts rod motion into spring/damper motion.

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
`kinematics.rod_motion_ratio` gives wheel-to-ROD, which is NOT the ratio that
sets wheel rate. On NFR26 both corners run a bellcrank between the rod and the
spring/damper, and its leverage is substantial:

    front   rod arm 67.6 mm, spring arm 91.2 mm
    rear    rod arm 46.6 mm, spring arm 74.5 mm

Wheel rate scales with the SQUARE of the damper motion ratio, so using the rod
ratio would put spring and damper rates out by a factor of roughly 1.5-3.

FORMULATION
-----------
The rocker is a rigid body with one rotational degree of freedom about a fixed
axis. The sheets give:

    Rocker_Frame    a point on the pivot axis
    Rocker_Axis     a second point defining the axis DIRECTION
                    (the sheets annotate this "*perp to rocker plane")
    IB_Push/IB_Pull rod attachment, on the rocker
    Rocker_Spring   spring/damper attachment, on the rocker
    Spring_Chassis  the other end of the spring/damper, chassis-fixed

Given the upright pose, the rod's outboard end is known, and the rod has fixed
length. That closes a scalar problem in the rocker angle:

    f(theta) = |rod_outboard - IB_rod(theta)| - L_rod = 0

Once theta is known the damper length follows directly. One scalar root-find
per operating point.

PUSHROD VS PULLROD VS DIRECT-ACTING
-----------------------------------
These differ only in the SIGN of the resulting motion ratio and in which way
the rocker turns -- not in the equations. `solve_rocker_angle` handles all
three, and a direct-acting corner (no rocker) short-circuits to the rod length
itself. That is what makes them interchangeable per the year-to-year
generality requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import brentq

from .geometry import Actuator, CornerGeometry
from .kinematics import CornerState, rod_outboard_point, rodrigues

__all__ = [
    "RockerState", "solve_rocker_angle", "rocker_state",
    "damper_motion_ratio", "wheel_rate",
]


@dataclass
class RockerState:
    angle_rad: float
    damper_length: float
    rod_attach: np.ndarray
    spring_attach: np.ndarray
    converged: bool
    reason: str = ""
    """Populated when `converged` is False. Distinguishes a numerical failure
    from a GEOMETRIC BIND -- the rod being unable to close on the rocker circle
    at all -- which is a real property of the linkage, not a solver problem."""


def _axis_unit(act: Actuator) -> np.ndarray:
    v = np.asarray(act.rocker_axis, dtype=float) - np.asarray(act.rocker_pivot, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError(
            "Rocker_Axis coincides with Rocker_Frame, so the pivot axis "
            "direction is undefined. These must be two distinct points on the "
            "axis."
        )
    return v / n


def _rotate(pts: np.ndarray, pivot: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    R = rodrigues(axis * theta)
    pts = np.atleast_2d(pts)
    out = (R @ (pts - pivot).T).T + pivot
    return out[0] if out.shape[0] == 1 else out


def solve_rocker_angle(act: Actuator, rod_outboard: np.ndarray,
                       rod_length: float, bracket_rad: float = 1.2) -> RockerState:
    """Find the rocker angle that keeps the rod at its design length.

    Args:
        act: the actuator, with rocker geometry populated.
        rod_outboard: current position of the rod's upright-side end.
        rod_length: design (constant) rod length.
        bracket_rad: half-width of the search bracket. Rockers do not travel
            far; +/-1.2 rad is generous.
    """
    if not act.has_rocker:
        raise ValueError("actuator has no rocker; use the rod length directly")

    pivot = np.asarray(act.rocker_pivot, dtype=float)
    axis = _axis_unit(act)
    rod_attach0 = np.asarray(act.inboard, dtype=float)
    spring0 = np.asarray(act.rocker_spring, dtype=float)
    spring_chassis = np.asarray(act.spring_chassis, dtype=float)

    def f(theta: float) -> float:
        p = _rotate(rod_attach0, pivot, axis, theta)
        return float(np.linalg.norm(rod_outboard - p) - rod_length)

    # Bracket the root by scanning. The function is smooth and has two roots in
    # general (the rocker can close on the rod from either side); we take the
    # one nearest zero, which is the assembled configuration. Scanning rather
    # than trusting a fixed bracket is what stops the solver from flipping to
    # the mirrored assembly mid-sweep.
    thetas = np.linspace(-bracket_rad, bracket_rad, 241)
    vals = np.array([f(t) for t in thetas])
    sign_change = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]

    if len(sign_change) == 0:
        # Distinguish a bind from a numerical miss by checking reachability
        # directly: the rod attach point traces a circle about the pivot axis,
        # so the achievable rod length spans [d_min, d_max] over that circle.
        # If L_rod lies outside that span, no rocker angle can ever close the
        # linkage -- the suspension has run out of travel.
        scan = np.linspace(-np.pi, np.pi, 721)
        reach = np.array([np.linalg.norm(rod_outboard - _rotate(rod_attach0, pivot, axis, t))
                          for t in scan])
        lo, hi = float(reach.min()), float(reach.max())
        if not (lo <= rod_length <= hi):
            side = "too short" if rod_length < lo else "too long"
            reason = (
                f"geometric bind: rod is {side}. Rod length {rod_length*1e3:.2f} mm "
                f"but the rocker attach circle can only be reached at "
                f"{lo*1e3:.2f}-{hi*1e3:.2f} mm from the rod's outboard end at this "
                f"wheel position. The linkage cannot assemble here -- this is a "
                f"travel limit of the geometry, not a solver failure."
            )
        else:
            reason = (
                f"no sign change found within +/-{bracket_rad:.2f} rad although "
                f"the rod length {rod_length*1e3:.2f} mm is within the reachable "
                f"span {lo*1e3:.2f}-{hi*1e3:.2f} mm. Widen `bracket_rad`."
            )
        return RockerState(float("nan"), float("nan"),
                           rod_attach0, spring0, converged=False, reason=reason)

    best = sign_change[np.argmin(np.abs(thetas[sign_change]))]
    theta = brentq(f, thetas[best], thetas[best + 1], xtol=1e-14)

    rod_attach = _rotate(rod_attach0, pivot, axis, theta)
    spring_attach = _rotate(spring0, pivot, axis, theta)
    return RockerState(
        angle_rad=float(theta),
        damper_length=float(np.linalg.norm(spring_attach - spring_chassis)),
        rod_attach=rod_attach, spring_attach=spring_attach, converged=True,
    )


def rocker_state(corner: CornerGeometry, state: CornerState) -> RockerState:
    """Solve the rocker for an already-solved upright pose."""
    act = corner.actuator
    if act is None:
        raise ValueError(f"{corner.name}: no actuator defined")

    # Carried by the UPPER WISHBONE, not the upright -- see
    # kinematics.rod_outboard_point. Using the upright pose here
    # overstated the damper motion ratio by ~15%.
    rod_outboard = rod_outboard_point(corner, state.pose)

    if not act.has_rocker:
        # Direct-acting: the "damper length" is the rod length itself.
        d = float(np.linalg.norm(rod_outboard - act.inboard))
        return RockerState(0.0, d, act.inboard, act.inboard, converged=True)

    return solve_rocker_angle(act, rod_outboard, act.length)


def damper_motion_ratio(corner: CornerGeometry, states: List[CornerState]) -> np.ndarray:
    """d(damper length) / d(wheel travel), as a curve over travel.

    THIS is the ratio that sets wheel rate and damping at the wheel. Returned
    as a curve because it varies with travel for any real rocker -- treating it
    as a constant is a known source of wheel-rate error (docs/07 SS2).

    Sign convention: returned positive for a corner whose damper compresses in
    bump, which is the usual case for both pushrod and pullrod. The sign is
    what distinguishes the two layouts; the magnitude is what matters for rate.
    """
    z = np.array([s.wheel_centre[2] for s in states])
    solved = [rocker_state(corner, s) for s in states]

    failed = [(zi, r) for zi, r in zip(z, solved) if not r.converged]
    if failed:
        z0 = z[len(z) // 2]
        detail = failed[0][1].reason
        raise ValueError(
            f"{corner.name}: rocker unsolvable at {len(failed)} of {len(z)} travel "
            f"points, first at {(failed[0][0] - z0) * 1e3:+.1f} mm from the swept "
            f"mid-point.\n  {detail}\n"
            f"Sweep a narrower travel range, or check the rocker hardpoints."
        )

    return np.gradient(np.array([r.damper_length for r in solved]), z)


def travel_limits(corner: CornerGeometry, states: List[CornerState]) -> tuple:
    """(min, max) wheel travel over which the linkage actually assembles.

    Returns absolute wheel-centre heights. Useful for reporting a geometry's
    usable travel rather than discovering the limit as a solver failure.
    """
    ok = [s.wheel_centre[2] for s in states if rocker_state(corner, s).converged]
    if not ok:
        raise ValueError(f"{corner.name}: rocker does not solve anywhere in this sweep")
    return float(min(ok)), float(max(ok))


def wheel_rate(spring_rate_N_per_m: float, motion_ratio: float) -> float:
    """Wheel rate from spring rate and DAMPER motion ratio.

        k_wheel = k_spring * MR^2

    The square is why the rod-vs-damper distinction matters so much: a 1.35x
    error in MR becomes a 1.8x error in wheel rate.
    """
    return float(spring_rate_N_per_m * motion_ratio ** 2)

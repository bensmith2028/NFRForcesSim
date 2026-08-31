"""Vehicle-level load cases -> per-corner contact-patch loads.

THE MISSING LINK
----------------
`suspension.forces` extracts member forces from a `CornerLoads`, and `T1` knows
the per-corner vertical loads at a given acceleration -- but nothing joined
them. Every force extraction in this repo so far has been a hand-built
`CornerLoads(Fz=..., Fy=...)` at a call site, which is exactly how a structural
case ends up quietly inconsistent with the vehicle model it is supposed to
come from.

This module is that join: give it an acceleration state, get the four corners.

THE CASES
---------
A structural load case is (a_x, a_y) plus the hardware state. The four
steady-state ones mirror the team's existing spreadsheet:

    braking             a_x = -1.85 g,  a_y =  0
    outside cornering   a_x = -0.24 g,  a_y = +1.59 g
    inside cornering    a_x = -0.24 g,  a_y = -1.59 g
    acceleration        a_x = +1.40 g,  a_y =  0

"Outside" and "inside" are not separate physics -- they are the SAME corner at
mirrored lateral acceleration. In ISO 8855 positive a_y is a LEFT turn, which
loads the right-hand tires, so a right-hand corner is OUTSIDE at +a_y and
INSIDE at -a_y. Modelling them as one case evaluated at +/- a_y (rather than two
hand-entered load sets) is what guarantees they stay consistent with each other.

The BUMP case does not belong here: it is a transient with no steady-state
acceleration, and it comes from `vehicle.quarter_car`. See `bump_corner_loads`.

HOW LATERAL FORCE IS DISTRIBUTED, AND WHY IT IS NOT FROM THE TIRE MODEL
-----------------------------------------------------------------------
Per-corner Fy is assigned in proportion to that corner's vertical load:

    Fy_i = Fz_i * (m * a_y) / sum(Fz)

i.e. every tire runs the same friction utilisation. That is the standard
assumption for a STRUCTURAL case and it is deliberate here, for two reasons:

  1. It is independent of the tire model, which is still carrying a placeholder
     longitudinal fit. A load case that moves when the tire fit is re-run is a
     bad basis for sizing a part.
  2. It is conservative in the right place. The real tire, with load
     sensitivity, puts slightly LESS force on the heavily-loaded outside
     corner than equal-utilisation predicts -- so this over-estimates the
     outside corner, which is the one that sizes the part.

If you want the tire model's actual distribution, run `solve_steady_state` and
build `CornerLoads` from its per-corner `Fy` -- the interface is the same. That
is a different question (what the car does) from this one (what the part must
survive).

ALIGNING MOMENT, AND WHY IT IS TAKEN AT ITS PEAK OVER SLIP ANGLE
-----------------------------------------------------------------
`Mz` used to be left at zero here. It is the load that sizes the TIE ROD
(docs/04 item 2), and leaving it out under-reported that member by roughly 3x
-- 455 N against ~1.4 kN on the front outside corner. `tire.aligning` now
supplies it.

It is set from `MF52Aligning.peak_mz`, the largest |Mz| over slip angle at that
corner's vertical load, signed to oppose Fy. That is an ENVELOPE choice and it
needs stating, because the peak Mz and the peak Fy do NOT occur together:
pneumatic trail collapses as the tire saturates, so |Mz| peaks near 3-5 deg of
slip where Fy is only about 70% of its own peak.

Pairing limit Fy with peak Mz is therefore conservative -- by roughly 1/0.7 on
the Fy contribution at the instant Mz is worst. It is the same trade this
module already makes for Fy itself (equal utilisation, deliberately generous on
the loaded outside corner), and it is the right one for a structural case: the
alternative, evaluating Mz at the limit slip angle, would report almost the
only point on the sweep where the aligning moment has already decayed to
nothing, and would size the tie rod from it.

Straight-line cases get Mz = 0, because a tire at zero slip angle generates
none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..core.units import G
from ..suspension.forces import CornerLoads, wheel_reaction_torque
from ..tire.aligning import MF52Aligning
from .loadtransfer import VehicleParams, load_transfer

__all__ = ["LoadCase", "STANDARD_CASES", "corner_loads_for_case",
           "bump_corner_loads"]

CORNERS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class LoadCase:
    """A steady-state structural case."""

    name: str
    a_x_g: float
    """Longitudinal, positive = accelerating."""
    a_y_g: float
    """Lateral, positive = LEFT turn (ISO), which loads the RIGHT-hand tires."""
    speed_m_s: float = 0.0
    """Only used for aero. Zero means no downforce -- conservative for the
    linkage (less vertical load) but NOT for the wheel/bearing, so set it
    where downforce matters."""

    def report(self) -> str:
        return (f"{self.name}: a_x {self.a_x_g:+.2f} g, a_y {self.a_y_g:+.2f} g"
                + (f", V {self.speed_m_s:.0f} m/s" if self.speed_m_s else ""))


STANDARD_CASES = (
    LoadCase("braking", a_x_g=-1.85, a_y_g=0.0),
    LoadCase("outside_cornering", a_x_g=-0.24, a_y_g=1.59),
    LoadCase("inside_cornering", a_x_g=-0.24, a_y_g=-1.59),
    LoadCase("acceleration", a_x_g=1.40, a_y_g=0.0),
)
"""Matches the team's `Suspension Component Forces NFR26` spreadsheet."""


def corner_loads_for_case(
        p: VehicleParams, case: LoadCase, *,
        brake_bias_front: float = 0.67,
        drive_fraction_front: float = 0.5,
        loaded_radius: Dict[str, float] | float = 0.198,
        caliper_on_upright: bool = True,
        drive_reacts_on_upright: bool = True,
        unsprung_mass: Optional[Dict[str, float]] = None,
        aligning: Optional[MF52Aligning] = None,
        static_camber: Optional[Dict[str, float]] = None,
) -> Dict[str, CornerLoads]:
    """Per-corner contact-patch loads for one steady-state case.

    Returns {"FL": CornerLoads, "FR": ..., "RL": ..., "RR": ...}.

    Longitudinal force is split by brake bias when decelerating and by torque
    split when accelerating -- NOT by the same number for both, which is a
    tempting simplification and wrong: a car can brake 67% front and drive 50%
    front, and the resulting link loads differ.

    Brake and drive torque are set from that corner's Fx via
    `wheel_reaction_torque`, so the caller only has to say WHERE the couple
    reacts (`caliper_on_upright`, `drive_reacts_on_upright`) and the extractor
    handles the free-body consequence.

    Args:
        aligning: tire aligning-moment model. Defaults to the fitted
            coefficients for the tire this car runs, with conicity/plysteer
            removed and at `lambda_mz = 1.0` (raw belt -- conservative; see
            `MF52Aligning`). Pass
            `MF52Aligning(lambda_mz=<tires.scaling.mu_y>)` for the on-track
            value, or `MF52Aligning(lambda_mz=0.0)` to reproduce the old
            zero-Mz behaviour.
        static_camber: per-corner inclination in radians, for the Mz camber
            terms. Defaults to zero at every corner, which costs about 5% on
            peak Mz at this car's -1 deg -- small next to the fit's own +/-20%,
            so it is a convenience default rather than a modelling claim.
            `solvers.steady_state.symmetric_camber` builds the correct dict.
    """
    a_x = case.a_x_g * G
    a_y = case.a_y_g * G
    lt = load_transfer(p, a_y, a_x, case.speed_m_s)

    total_Fz = sum(lt.Fz.values())
    accelerating = a_x > 0.0

    # Longitudinal force per AXLE, then split evenly left/right. Symmetric
    # left/right is an assumption: it is exact for braking, and for drive it
    # holds unless torque vectoring is in play (which would be a T2.5 study,
    # and would ADD asymmetry rather than change the peak).
    Fx_total = p.mass * a_x
    if accelerating:
        front_share = drive_fraction_front
    else:
        front_share = brake_bias_front
    Fx_axle = {"F": Fx_total * front_share, "R": Fx_total * (1.0 - front_share)}

    if isinstance(loaded_radius, (int, float)):
        loaded_radius = {k: float(loaded_radius) for k in CORNERS}
    unsprung_mass = unsprung_mass or {k: 0.0 for k in CORNERS}
    # Conicity/plysteer zeroed, matching what `build_tire` does to the lateral
    # model and for the same reason: one tire's measured asymmetry applied to
    # both sides of a symmetric car makes mirrored corners disagree. On this fit
    # the raw shifts are worth 12% between +3 and -3 deg, so `inside_cornering`
    # and `outside_cornering` -- which are meant to be one case at +/- a_y --
    # would stop being mirror images of each other.
    aligning = (MF52Aligning().without_conicity_and_plysteer()
                if aligning is None else aligning)
    static_camber = static_camber or {k: 0.0 for k in CORNERS}

    out: Dict[str, CornerLoads] = {}
    for k in CORNERS:
        Fz = lt.Fz[k]
        # Equal friction utilisation -- see the module docstring.
        Fy = (Fz / total_Fz) * p.mass * a_y if total_Fz > 1e-9 else 0.0
        Fx = 0.5 * Fx_axle[k[0]]
        torque = wheel_reaction_torque(Fx, loaded_radius[k])

        # Aligning moment at its peak over slip angle, signed to OPPOSE Fy --
        # it is a restoring moment. Zero in a straight line, where the tire
        # runs no slip angle and generates none. See the module docstring for
        # why the peak rather than the value at the limit.
        if abs(Fy) > 1e-9:
            Mz = -np.sign(Fy) * abs(float(
                aligning.peak_mz(Fz, static_camber[k])))
        else:
            Mz = 0.0

        out[k] = CornerLoads(
            Fx=Fx, Fy=Fy, Fz=Fz, Mz=Mz,
            brake_torque=0.0 if accelerating else torque,
            drive_torque=torque if accelerating else 0.0,
            caliper_on_upright=caliper_on_upright,
            drive_reacts_on_upright=drive_reacts_on_upright,
            unsprung_mass=unsprung_mass[k],
        )
    return out


def bump_corner_loads(bump_result, *, corner: str = "FR",
                      caliper_on_upright: bool = True,
                      drive_reacts_on_upright: bool = True) -> CornerLoads:
    """`CornerLoads` for a bump, from a `quarter_car.BumpResult`.

    Thin wrapper over `BumpResult.to_corner_loads` so all five cases are
    reachable through one module. The bump is a SINGLE-CORNER, single-wheel
    event: a quarter car has no roll or pitch, so there is no meaningful
    "other three corners" to return. Combine it with a steady-state case by
    passing the steady loads through as keyword arguments if you want
    bump-while-cornering.
    """
    return bump_result.to_corner_loads(
        caliper_on_upright=caliper_on_upright,
        drive_reacts_on_upright=drive_reacts_on_upright)

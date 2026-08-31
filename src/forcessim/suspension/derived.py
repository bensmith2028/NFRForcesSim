"""Kinematic quantities DERIVED FROM HARDPOINTS, never hand-entered.

WHY THIS MODULE EXISTS
----------------------
Three numbers that are outputs of the suspension geometry were being carried in
`configs/nfr26_baseline.yaml` as inputs:

    dampers.motion_ratio_at_ride.front / .rear
    kinematics_acceptance.front.roll_centre_height_m / rear...
    camber_gain.front_per_rad / .rear_per_rad

Every one of them goes stale the moment a hardpoint moves, and nothing said so.
That is not hypothetical: `motion_ratio_at_ride` sat at 1.1567 / 1.1583 -- the
output of a T0 rocker solve that carried the push/pullrod pickup on the upright
instead of the upper wishbone -- and propagated silently into wheel rate, roll
stiffness, LLTD and every T1/T2 result until the kinematics bug was found
independently. The stored number outlived the code that produced it, which is
the failure mode this module removes.

THE RULE
--------
If it can be computed from the hardpoints, it is computed from the hardpoints.
A value still present in the YAML becomes an ACCEPTANCE TARGET, checked against
the derived value on load (`check_against_config`), not an input. A stale entry
then fails loudly instead of quietly steering the whole model.

WHAT IS AND IS NOT DERIVED HERE
-------------------------------
Derived: damper motion ratio at ride, kinematic roll centre height, camber gain
per radian of body roll.

Also derived: anti-dive, anti-lift and anti-squat. These are NOT pure geometry
-- they combine the side-view swing arm with brake bias, CG height, wheelbase
and torque split -- so they are computed by `anti_percent` from whichever of
those you supply, rather than stored anywhere. Change the brake bias and the
anti-dive follows; move a hardpoint and it follows too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .geometry import CornerGeometry
from .kinematics import roll_centre_height, side_view_swing_arm, sweep_heave
from .rocker import damper_motion_ratio

__all__ = ["DerivedKinematics", "AxleKinematics", "AntiGeometry",
           "camber_gain_per_rad_roll", "derive_axle", "derive_both_axles",
           "anti_percent", "derive_anti"]


@dataclass(frozen=True)
class AxleKinematics:
    """Everything one axle's hardpoints determine."""

    motion_ratio: float
    """Wheel-to-DAMPER displacement ratio at ride height."""
    roll_centre_height_m: float
    camber_gain_per_rad_roll: float
    """|d(camber)/d(body roll)|, chassis-relative. See the function below."""
    svsa_angle_from_patch_rad: float
    """Side-view swing-arm angle measured from the CONTACT PATCH -- the
    reference when the brake/drive couple reacts at the UPRIGHT."""
    svsa_angle_from_wheel_centre_rad: float
    """...and from the WHEEL CENTRE, the reference when it reacts at the
    CHASSIS (inboard brake, inboard motor + halfshafts)."""
    svsa_length_from_patch_m: float


@dataclass(frozen=True)
class DerivedKinematics:
    front: AxleKinematics
    rear: AxleKinematics

    def as_dict(self) -> Dict[str, float]:
        return {
            "dampers.motion_ratio_at_ride.front": self.front.motion_ratio,
            "dampers.motion_ratio_at_ride.rear": self.rear.motion_ratio,
            "kinematics_acceptance.front.roll_centre_height_m":
                self.front.roll_centre_height_m,
            "kinematics_acceptance.rear.roll_centre_height_m":
                self.rear.roll_centre_height_m,
            "camber_gain.front_per_rad": self.front.camber_gain_per_rad_roll,
            "camber_gain.rear_per_rad": self.rear.camber_gain_per_rad_roll,
        }


def camber_gain_per_rad_roll(corner: CornerGeometry, track_m: float,
                             dz: float = 0.010) -> float:
    """|d(inclination) / d(body roll)| for one corner, per radian.

    THE QUANTITY, because it is easy to compute a different one by accident.
    In body roll of phi the two wheels of an axle move OPPOSITE vertically, by
    +/- (track/2) * phi for small phi. So the camber the geometry generates per
    radian of roll is the heave camber gain scaled by the half-track:

        d(camber)/d(roll) = d(camber)/dz * (track / 2)

    This is the CHASSIS-RELATIVE kinematic gain. It does NOT include the
    chassis's own roll, which `solvers.steady_state._cambers` adds separately
    via the per-corner sign in `symmetric_camber_gain`. Returning the
    ground-relative figure here would double-count that term.

    `sweep_heave` is the right primitive despite moving both wheels together:
    a single corner's camber-vs-z curve is a property of that corner alone, and
    roll simply evaluates it at +z on one side and -z on the other. What would
    be wrong is reading a MOTION RATIO off a heave sweep and calling it a roll
    motion ratio -- that one does differ, because an anti-roll bar loads only
    in roll. There is no ARB on this car (`arb_present: false` both ends), but
    the distinction is why this function is separate rather than an alias.

    Verified against the hand-carried config values it replaces: 0.3445 vs
    0.345 front and 0.4765 vs 0.477 rear, i.e. 0.1% -- close enough to confirm
    the construction and far closer than those [ESTIMATED] numbers deserved.

    Args:
        dz: half-width of the central difference in wheel travel, m. 10 mm sits
            inside the ~11 mm design bump travel, so the gain is evaluated over
            the range the car actually uses rather than extrapolated.
    """
    z0 = float(corner.wheel_centre[2])
    states = sweep_heave(corner, np.array([-dz, 0.0, dz]))
    if not all(s.converged for s in states):
        raise ValueError(
            f"{corner.name}: the pose solver did not converge across the "
            f"+/-{dz * 1000:.0f} mm camber-gain sweep, so the gain would be "
            f"read off a bad pose. Check the hardpoints.")
    d_camber = np.radians(states[2].camber_deg - states[0].camber_deg)
    return abs(d_camber / (2.0 * dz)) * (track_m / 2.0)


def derive_axle(corner: CornerGeometry, track_m: float,
                travel: Optional[np.ndarray] = None) -> AxleKinematics:
    """All three derived quantities for one axle, from its hardpoints alone."""
    if corner.wheel_centre is None:
        raise ValueError(
            f"{corner.name}: wheel_centre is required to derive kinematics "
            f"(it sets the ride-height operating point).")
    z0 = float(corner.wheel_centre[2])

    if travel is None:
        # Symmetric about ride and wide enough for a clean central difference,
        # but well inside the rear linkage bind at +23 mm.
        travel = np.linspace(-0.010, 0.010, 5)

    states = sweep_heave(corner, travel)
    if not all(s.converged for s in states):
        raise ValueError(f"{corner.name}: pose solver failed across the "
                         f"motion-ratio sweep; refusing to report a ratio.")
    mr = np.abs(damper_motion_ratio(corner, states))
    mid = len(travel) // 2

    contact_patch = np.array([corner.wheel_centre[0], corner.wheel_centre[1], 0.0])
    rc = roll_centre_height(corner, z0, contact_patch)

    wheel_centre = np.asarray(corner.wheel_centre, dtype=float)
    len_patch, ang_patch = side_view_swing_arm(corner, z0, contact_patch)
    _, ang_wc = side_view_swing_arm(corner, z0, wheel_centre)

    return AxleKinematics(
        motion_ratio=float(mr[mid]),
        roll_centre_height_m=float(rc),
        camber_gain_per_rad_roll=camber_gain_per_rad_roll(corner, track_m),
        svsa_angle_from_patch_rad=float(ang_patch),
        svsa_angle_from_wheel_centre_rad=float(ang_wc),
        svsa_length_from_patch_m=float(len_patch),
    )


def derive_both_axles(front: CornerGeometry, rear: CornerGeometry,
                      track_front_m: float, track_rear_m: float
                      ) -> DerivedKinematics:
    return DerivedKinematics(front=derive_axle(front, track_front_m),
                             rear=derive_axle(rear, track_rear_m))


# ---------------------------------------------------------------------------
# Anti-dive / anti-lift / anti-squat
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AntiGeometry:
    """Longitudinal load-transfer reacted through the LINKS, as a percentage.

    100% means the geometry carries the whole of that axle's share of
    longitudinal load transfer and the springs see none of it (no dive/squat at
    all). 0% means the springs take all of it. Negative means the geometry
    works the wrong way -- pro-dive or pro-squat.
    """

    anti_dive_front_pct: float
    anti_lift_rear_pct: float
    anti_squat_front_pct: float
    anti_squat_rear_pct: float

    def report(self) -> str:
        return (f"braking : anti-dive front {self.anti_dive_front_pct:6.1f}%   "
                f"anti-lift rear {self.anti_lift_rear_pct:6.1f}%\n"
                f"driving : anti-squat front {self.anti_squat_front_pct:6.1f}%  "
                f"rear {self.anti_squat_rear_pct:6.1f}%")


def anti_percent(svsa_angle_rad: float, force_fraction: float,
                 wheelbase_m: float, cg_height_m: float) -> float:
    """Percentage of longitudinal load transfer taken by the linkage.

    DERIVATION, so the factors are auditable rather than folded into a constant.
    Under an acceleration `a`, total longitudinal load transfer is

        dW = m * a * h / L

    The axle in question carries a longitudinal force `F = k * m * a`, where
    `k` is that axle's share (brake bias, or torque split). Fed into the links
    along a side-view line rising at `theta`, that force reacts a VERTICAL
    component `F * tan(theta)` through the linkage rather than the spring. So

        anti = F * tan(theta) / dW = k * tan(theta) * L / h

    which is independent of `a` -- as it must be, since anti-dive is a property
    of the car, not of how hard you brake.

    `svsa_angle_rad` MUST be measured from the right reference point: contact
    patch when the couple reacts at the upright, wheel centre when it reacts at
    the chassis. `kinematics.side_view_swing_arm` takes that point as an
    argument precisely so the choice is explicit at the call site.
    """
    if cg_height_m <= 0.0:
        raise ValueError("cg_height_m must be positive")
    return 100.0 * force_fraction * np.tan(svsa_angle_rad) * wheelbase_m / cg_height_m


def derive_anti(derived: DerivedKinematics, wheelbase_m: float, cg_height_m: float,
                brake_bias_front: float,
                drive_fraction_front: float,
                brake_reacts_on_upright: bool = True,
                drive_reacts_on_upright: bool = True) -> AntiGeometry:
    """Combine the side-view geometry with the hardware that feeds it force.

    Every argument here is something you set elsewhere -- brake bias in the
    config, CG height in the config, torque split by the powertrain layout --
    so this recomputes whenever any of them moves. Nothing is cached against a
    stale copy.
    """
    def ang(axle, on_upright):
        a = getattr(derived, axle)
        return (a.svsa_angle_from_patch_rad if on_upright
                else a.svsa_angle_from_wheel_centre_rad)

    return AntiGeometry(
        anti_dive_front_pct=anti_percent(
            ang("front", brake_reacts_on_upright), brake_bias_front,
            wheelbase_m, cg_height_m),
        anti_lift_rear_pct=anti_percent(
            ang("rear", brake_reacts_on_upright), 1.0 - brake_bias_front,
            wheelbase_m, cg_height_m),
        anti_squat_front_pct=anti_percent(
            ang("front", drive_reacts_on_upright), drive_fraction_front,
            wheelbase_m, cg_height_m),
        anti_squat_rear_pct=anti_percent(
            ang("rear", drive_reacts_on_upright), 1.0 - drive_fraction_front,
            wheelbase_m, cg_height_m),
    )

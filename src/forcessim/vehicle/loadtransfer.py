"""Load transfer, including chassis torsional compliance.

Implements docs/03 SS3 and SS4. The chassis is modelled as a torsional spring
between the front and rear suspension planes, which makes LLTD an OUTPUT of
springs, roll centres, tire rates, track and CG height -- never an input.

WHAT GOES THROUGH THE CHASSIS AND WHAT DOES NOT
-----------------------------------------------
Only the ELASTIC path twists the chassis. Geometric load transfer (through the
links, set by roll centre height) and unsprung load transfer react at their own
axle and do not. Folding them into the torsion solve overstates the chassis
effect substantially, since geometric transfer is a large fraction of the total
at typical FSAE roll centre heights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.units import G

__all__ = ["VehicleParams", "AeroParams", "LoadTransferResult", "load_transfer"]


@dataclass
class AeroParams:
    ClA: float = 0.0
    CdA: float = 0.0
    balance_front: float = 0.5
    cp_height_m: float = 0.0
    air_density: float = 1.20

    def downforce(self, V: float) -> float:
        return 0.5 * self.air_density * V * V * self.ClA

    def drag(self, V: float) -> float:
        return 0.5 * self.air_density * V * V * self.CdA


@dataclass
class VehicleParams:
    """Everything load transfer needs. SI throughout."""

    mass: float
    cg_height: float
    weight_dist_front: float
    wheelbase: float
    track_front: float
    track_rear: float

    unsprung_front: float
    unsprung_rear: float
    unsprung_cg_height: float

    roll_centre_front: float
    roll_centre_rear: float

    roll_stiffness_front: float          # N*m/rad, at the wheel, tire in series
    roll_stiffness_rear: float

    chassis_torsional_stiffness: Optional[float] = None
    """N*m/rad. None means rigid."""

    aero: AeroParams = field(default_factory=AeroParams)

    # ---- derived ----
    @property
    def a(self) -> float:
        """CG to front axle."""
        return self.wheelbase * (1.0 - self.weight_dist_front)

    @property
    def b(self) -> float:
        """CG to rear axle."""
        return self.wheelbase * self.weight_dist_front

    @property
    def unsprung_total(self) -> float:
        return 2.0 * (self.unsprung_front + self.unsprung_rear)

    @property
    def sprung_mass(self) -> float:
        return self.mass - self.unsprung_total

    @property
    def sprung_cg_height(self) -> float:
        """Derived, never entered independently -- deriving it is what keeps
        sprung and unsprung bookkeeping consistent."""
        return ((self.mass * self.cg_height
                 - self.unsprung_total * self.unsprung_cg_height)
                / self.sprung_mass)

    @property
    def roll_axis_height_at_cg(self) -> float:
        """Roll axis height interpolated to the CG's longitudinal station."""
        f = self.weight_dist_front
        return self.roll_centre_rear + (self.roll_centre_front - self.roll_centre_rear) * f

    @property
    def roll_moment_arm(self) -> float:
        """Perpendicular distance from sprung CG to the roll axis."""
        return self.sprung_cg_height - self.roll_axis_height_at_cg


@dataclass
class LoadTransferResult:
    Fz: Dict[str, float]
    lltd: float
    delta_front: float
    delta_rear: float
    roll_angle_front: float
    roll_angle_rear: float
    chassis_twist: float
    breakdown: Dict[str, float]
    lifted: Tuple[str, ...] = ()
    """Corners carrying zero load. Non-empty means lateral load transfer on
    that axle has SATURATED, so any result at this operating point is on the
    edge of what this model can represent -- see `_axle_split`."""

    @property
    def any_wheel_lifted(self) -> bool:
        return bool(self.lifted)

    def report(self) -> str:
        lines = [f"{'corner':<6}{'Fz N':>10}"]
        for k in ("FL", "FR", "RL", "RR"):
            lines.append(f"{k:<6}{self.Fz[k]:10.1f}")
        if self.lifted:
            lines.append(f">>> WHEEL LIFT: {', '.join(self.lifted)} at zero load "
                         f"-- lateral transfer is saturated on that axle. <<<")
        lines += [
            f"LLTD {self.lltd:.4f} front",
            f"transfer  front {self.delta_front:8.1f} N   rear {self.delta_rear:8.1f} N",
            f"roll      front {np.degrees(self.roll_angle_front):6.3f} deg  "
            f"rear {np.degrees(self.roll_angle_rear):6.3f} deg  "
            f"twist {np.degrees(self.chassis_twist):6.3f} deg",
            "contributions to front transfer:",
        ]
        for k, v in self.breakdown.items():
            lines.append(f"  {k:<12}{v:9.1f} N")
        return "\n".join(lines)


def _axle_split(axle_total: float, d_lat: float) -> tuple:
    """Split one axle's vertical load into (left, right), conserving the total.

    >>> DO NOT REPLACE THIS WITH `max(Fz, 0.0)` ON EACH WHEEL. <<<

    Past wheel lift the naive clamp is not conservative, it is CREATIVE: it
    zeroes the inside wheel without giving its share to the outside one, so the
    outside keeps `axle/2 + d_lat` with `d_lat > axle/2` and the axle ends up
    carrying more than it was given. On the baseline config the GGV's own
    high-speed envelope runs past lift, and the old clamp invented 420 N at
    33 m/s -- 9% of true total vertical load, straight into the grip budget.
    Worse, because grip grows with Fz, capability kept RISING with lateral
    demand past lift, so the bisection in `max_trimmed_lateral` converged on a
    fictitious operating point instead of failing visibly.

    The physical statement: once the inside wheel leaves the ground, lateral
    load transfer on that axle SATURATES. The outside wheel carries the whole
    axle load and no more; further roll moment goes into rolling the car over,
    which is outside this model. `LoadTransferResult.lifted` reports it so the
    saturation is visible rather than silent.
    """
    left = axle_total / 2.0 - d_lat
    right = axle_total / 2.0 + d_lat
    if axle_total <= 0.0:
        # The whole axle is unloaded (extreme longitudinal transfer). Nothing
        # to distribute; a tire cannot pull down on the road.
        return 0.0, 0.0
    if left < 0.0:
        return 0.0, axle_total
    if right < 0.0:
        return axle_total, 0.0
    return left, right


def load_transfer(p: VehicleParams, a_y: float, a_x: float = 0.0,
                  V: float = 0.0) -> LoadTransferResult:
    """Per-corner vertical loads.

    Args:
        p: vehicle parameters.
        a_y: lateral acceleration, m/s^2. Positive = leftward (ISO +y), which
             is a LEFT turn, loading the RIGHT-hand tires.
        a_x: longitudinal acceleration, m/s^2. Positive = accelerating.
        V: speed, m/s, for aero.

    Returns:
        LoadTransferResult with per-corner Fz and the LLTD it implies.
    """
    W = p.mass * G
    down = p.aero.downforce(V)

    # ---- static + aero, before any transfer ----
    Fz_front_axle = W * p.weight_dist_front + down * p.aero.balance_front
    Fz_rear_axle = W * (1.0 - p.weight_dist_front) + down * (1.0 - p.aero.balance_front)

    # ---- longitudinal transfer ----
    d_long = p.mass * a_x * p.cg_height / p.wheelbase
    Fz_front_axle -= d_long
    Fz_rear_axle += d_long

    # ---- lateral: sprung, elastic path through the chassis ----
    ms = p.sprung_mass
    H = p.roll_moment_arm
    M_total = ms * a_y * H
    M_f = M_total * p.weight_dist_front
    M_r = M_total * (1.0 - p.weight_dist_front)

    Kf, Kr = p.roll_stiffness_front, p.roll_stiffness_rear
    Kc = p.chassis_torsional_stiffness

    if Kc is None or not np.isfinite(Kc):
        phi = M_total / (Kf + Kr)
        phi_f = phi_r = phi
    else:
        # Closed form for  [[Kf+Kc, -Kc], [-Kc, Kr+Kc]] @ [phi_f, phi_r] = [M_f, M_r].
        # Algebraically identical to np.linalg.solve on the 2x2, but this is
        # the innermost loop of the GGV sweep (called ~200k times per envelope)
        # and the LAPACK call costs more than the arithmetic by an order of
        # magnitude. det = Kf*Kr + Kc*(Kf + Kr) > 0 whenever all three
        # stiffnesses are positive, so it cannot go singular on real input.
        det = Kf * Kr + Kc * (Kf + Kr)
        if abs(det) < 1e-30:
            raise ValueError(
                "roll/chassis stiffness matrix is singular -- at least one of "
                "roll_stiffness_front/rear or chassis_torsional_stiffness must "
                "be non-zero")
        phi_f = ((Kr + Kc) * M_f + Kc * M_r) / det
        phi_r = (Kc * M_f + (Kf + Kc) * M_r) / det

    elastic_f = Kf * phi_f / p.track_front
    elastic_r = Kr * phi_r / p.track_rear

    # ---- lateral: geometric, straight through the links (bypasses chassis) ----
    ms_f = ms * p.weight_dist_front
    ms_r = ms * (1.0 - p.weight_dist_front)
    geometric_f = ms_f * a_y * p.roll_centre_front / p.track_front
    geometric_r = ms_r * a_y * p.roll_centre_rear / p.track_rear

    # ---- lateral: unsprung ----
    unsprung_f = 2.0 * p.unsprung_front * a_y * p.unsprung_cg_height / p.track_front
    unsprung_r = 2.0 * p.unsprung_rear * a_y * p.unsprung_cg_height / p.track_rear

    # ---- lateral: aero ----
    # ZERO, AND THAT IS THE PHYSICS, NOT A SIMPLIFICATION.
    #
    # Downforce is a FORCE applied to the body, not a mass. It has no lateral
    # acceleration reaction, so acting on the centreline it exerts no roll
    # moment and produces no lateral load transfer. Its whole contribution is
    # the extra vertical load already added to the axles above.
    #
    # This previously read
    #     aero_f = down * balance_front * (a_y / G) * cp_height / track_front
    # which converts downforce into an equivalent MASS (down/G) and gives that
    # phantom mass lateral inertia at CP height. On the baseline config that
    # term reached 32% of front lateral transfer at 25 m/s and made LLTD drift
    # 0.522 -> 0.502 with speed. In this model LLTD is provably speed-
    # INDEPENDENT -- elastic, geometric and unsprung transfer are all functions
    # of a_y alone -- so every bit of that drift was an artefact, and it is
    # pinned as an invariant by
    # test_lltd_is_independent_of_speed_because_downforce_carries_no_roll_moment.
    #
    # The real aero-in-roll effect (downforce and balance changing with roll
    # and ride height) is a roll-dependent aero MAP, not this formula. It is
    # not modelled; adding it means giving AeroParams a roll sensitivity, and
    # it would then enter through the elastic path like any other roll moment.
    aero_f = 0.0
    aero_r = 0.0

    d_lat_f = elastic_f + geometric_f + unsprung_f + aero_f
    d_lat_r = elastic_r + geometric_r + unsprung_r + aero_r

    total = d_lat_f + d_lat_r
    lltd = float(d_lat_f / total) if abs(total) > 1e-9 else 0.5

    # Positive a_y is a left turn, so load moves to the RIGHT-hand tires.
    left_f, right_f = _axle_split(Fz_front_axle, d_lat_f)
    left_r, right_r = _axle_split(Fz_rear_axle, d_lat_r)
    Fz = {"FL": left_f, "FR": right_f, "RL": left_r, "RR": right_r}
    lifted = tuple(k for k in ("FL", "FR", "RL", "RR") if Fz[k] <= 0.0)

    return LoadTransferResult(
        Fz=Fz, lltd=lltd, delta_front=d_lat_f, delta_rear=d_lat_r,
        roll_angle_front=float(phi_f), roll_angle_rear=float(phi_r),
        chassis_twist=float(phi_f - phi_r),
        lifted=lifted,
        breakdown={"elastic": elastic_f, "geometric": geometric_f,
                   "unsprung": unsprung_f, "aero": aero_f},
    )


def setup_authority(p: VehicleParams, a_y: float, d_roll_stiffness: float = 100.0) -> float:
    """Fraction of an intended ARB change that survives chassis compliance.

    eta = (dLLTD/dK_front)_actual / (dLLTD/dK_front)_rigid

    Converts an abstract structural target into a concrete one: "we need K_c
    above X to retain 90% of our ARB authority." See docs/03 SS4.4.
    """
    import copy

    def slope(chassis_k):
        q = copy.copy(p)
        q.chassis_torsional_stiffness = chassis_k
        lo = load_transfer(q, a_y).lltd
        q2 = copy.copy(q)
        q2.roll_stiffness_front = p.roll_stiffness_front + d_roll_stiffness
        return (load_transfer(q2, a_y).lltd - lo) / d_roll_stiffness

    rigid = slope(None)
    if abs(rigid) < 1e-15:
        return 1.0
    return float(slope(p.chassis_torsional_stiffness) / rigid)

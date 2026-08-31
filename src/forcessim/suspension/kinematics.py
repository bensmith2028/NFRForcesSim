"""T0 -- suspension kinematics solver.

Solves the upright pose for a prescribed wheel vertical position and rack
travel, then derives camber, toe, motion ratio and the instant axis.

FORMULATION
-----------
The upright is a rigid body: 6 unknowns (translation + rotation vector). Five
link-length constraints plus one driving input close the system exactly, so the
corner has one degree of freedom, as a suspension should.

    residual[0..4] = |p_outboard(pose) - p_inboard| - L_link
    residual[5]    = wheel_centre_z(pose) - target_z

This is topology-agnostic by construction -- it never asks whether the corner
is a double wishbone or a five-link, only for its list of links. See
geometry.py.

Steering enters by translating the inboard end of any link flagged `steered`,
which is exactly what a rack does. Ackermann is therefore an OUTPUT of the
geometry, not a parameter.

VALIDATION POSTURE
------------------
The NFR26 geometry sheets carry OptimumK-derived values, but the team has
flagged that some may be stale. So they are NOT used as pass/fail acceptance
tests. Validation rests on:

  1. Closed-form answers on synthetic geometry (see tests).
  2. Internal consistency -- link lengths must be invariant through travel.
  3. The subset of sheet values that depend only on hardpoints with no
     construction ambiguity (caster, kingpin, scrub, mechanical trail), which
     reproduce to five significant figures and are therefore demonstrably
     current.

Swing-arm and roll-centre figures from the sheet are reported alongside ours as
information, never asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .geometry import CornerGeometry

__all__ = ["Pose", "CornerState", "solve_pose", "sweep_heave", "rodrigues",
           "side_view_instant_centre", "side_view_swing_arm",
           "wishbone_links", "upper_wishbone_links", "rod_pickup_arm",
           "rod_outboard_point"]


def rodrigues(rv: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector (axis * angle)."""
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclass(frozen=True)
class Pose:
    """Rigid-body pose of the upright relative to its design position."""

    translation: np.ndarray
    rotation_vector: np.ndarray
    reference: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return rodrigues(self.rotation_vector)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        out = (self.matrix @ (pts - self.reference).T).T + self.reference + self.translation
        return out[0] if out.shape[0] == 1 else out


@dataclass
class CornerState:
    """Solved kinematic state at one (wheel_z, rack) operating point."""

    wheel_z: float
    rack_travel: float
    pose: Pose
    wheel_centre: np.ndarray
    spindle_axis: np.ndarray
    camber_rad: float
    toe_rad: float
    rod_length: float
    residual_norm: float
    converged: bool

    @property
    def camber_deg(self) -> float:
        return float(np.degrees(self.camber_rad))

    @property
    def toe_deg(self) -> float:
        return float(np.degrees(self.toe_rad))


def _nominal_spindle_axis(camber_rad: float, toe_rad: float, left: bool) -> np.ndarray:
    """Wheel rotation axis at design position, ISO frame, unit length.

    Built from the design camber and toe rather than assumed to be lateral,
    because the whole point of tracking camber is that the axis is not lateral.

    Sign convention, pinned by tests:
        the axis nominally points OUTBOARD (+y for a left corner, -y for right);
        positive camber leans the top of the wheel outboard;
        positive toe points the wheel's front edge inboard (toe-in).

    CAMBER AND TOE ARE PER-SIDE CONVENTIONS, SO BOTH ANGLES MIRROR.
    -------------------------------------------------------------
    Both are defined relative to the car's centreline -- "outboard" and
    "inboard" mean opposite things in +y on the two sides -- so a left corner
    is the MIRROR IMAGE of a right one, not a copy. That is why both rotation
    angles carry `-s` and not just the starting axis.

    An earlier version applied the rotations by `+camber` and `+toe` on both
    sides and mirrored only the axis direction. That is an ISO-frame
    inclination/steer pair, not a camber/toe pair, and it is right on the
    RIGHT corner and inverted on the left. Measured on mirrored front
    geometry, camber gain in bump came out -32.89 deg/m on FR against
    +33.35 deg/m on FL, where mirrored corners must gain camber the SAME way;
    at 15 mm of bump FL reported -0.50 deg where the true camber was -1.49 deg.
    Static toe was worse than a sign error: feeding the geometry sheet's
    front -0.5 deg into both sides gave each axle 0.5 deg/wheel of STEER
    instead of symmetric toe-out, and inverted every left-corner bump-steer
    curve.

    It survived because `_camber_toe_from_axis` is the exact inverse of this
    function, so the round-trip test passed either way, and because no test in
    the repo built a left corner. Now pinned by
    test_camber_and_toe_mirror_between_left_and_right.

    Expanding the composition below:

        n_x =  sin(toe) * cos(camber)          (same on both sides)
        n_y =  s * cos(toe) * cos(camber)
        n_z = -sin(camber)                     (same on both sides)
    """
    s = 1.0 if left else -1.0
    n = np.array([0.0, s, 0.0])
    # camber: rotate about +x (forward), mirrored on the left
    c, sc = np.cos(-s * camber_rad), np.sin(-s * camber_rad)
    n = np.array([n[0], c * n[1] - sc * n[2], sc * n[1] + c * n[2]])
    # toe: rotate about +z (up), mirrored on the left
    ct, st = np.cos(-s * toe_rad), np.sin(-s * toe_rad)
    n = np.array([ct * n[0] - st * n[1], st * n[0] + ct * n[1], n[2]])
    return n / np.linalg.norm(n)


def _camber_toe_from_axis(axis: np.ndarray, left: bool) -> Tuple[float, float]:
    """Inverse of `_nominal_spindle_axis`. Must round-trip exactly.

    Deriving the terms rather than guessing them. The forward map builds

        n = Rz(-s * toe) @ Rx(-s * camber) @ [0, s, 0]

    which expands to

        n_x =  sin(toe) * cos(camber)
        n_y =  s * cos(toe) * cos(camber)
        n_z = -sin(camber)

    so, since cos(camber) > 0 for any physical camber,

        camber = arcsin(-n_z)
        toe    = atan2(n_x, s * n_y)

    Note NEITHER `s` sits where it used to. Camber has none at all: the axis
    direction already carries the side, and `n_z` measures the same physical
    lean on both. Toe has one, inside the atan2 -- it cancels the `s` in `n_y`
    so the ratio is tan(toe) on both sides.

    History worth keeping, because both errors are easy to reintroduce and
    neither is visible in a round-trip test (this function is the exact
    inverse of the forward map, so the round trip passes under any consistent
    pair of conventions):

      - an early version used +atan2(s*n_x, s*n_y) for toe and returned
        +0.5 deg for an input of -0.5 deg, inverting every toe reading;
      - the version before this one used arcsin(s*n_z) and
        -atan2(s*n_x, s*n_y), which is self-consistent and correct on a RIGHT
        corner but mirrored on a left one. See `_nominal_spindle_axis` for the
        measured consequences.

    Pinned by test_camber_toe_round_trip AND
    test_camber_and_toe_mirror_between_left_and_right -- the round trip alone
    is not enough.
    """
    s = 1.0 if left else -1.0
    n = axis / np.linalg.norm(axis)
    camber = float(np.arcsin(np.clip(-n[2], -1.0, 1.0)))
    toe = float(np.arctan2(n[0], s * n[1]))
    return camber, toe


def solve_pose(corner: CornerGeometry, wheel_z: float, rack_travel: float = 0.0,
               x0: Optional[np.ndarray] = None,
               tol: float = 1e-10) -> Tuple[Pose, float, bool]:
    """Solve the upright pose for a target wheel-centre height and rack travel.

    Args:
        corner: geometry, with `wheel_centre` populated.
        wheel_z: target absolute wheel-centre height, m (ISO z, up).
        rack_travel: lateral rack displacement, m. Positive is +y (left).
        x0: warm start; strongly recommended when sweeping.

    Returns:
        (pose, residual_norm, converged)
    """
    if corner.wheel_centre is None:
        raise ValueError(
            f"{corner.name}: wheel_centre is not set. The driving constraint is "
            f"the wheel-centre height, so the solver cannot run without it. "
            f"Derive it from track width and loaded radius (see config "
            f"'wheels.centre_m')."
        )

    ref = np.asarray(corner.wheel_centre, dtype=float)
    lengths = corner.link_lengths
    outboard = np.array([l.outboard for l in corner.links])
    inboard = np.array([l.inboard for l in corner.links])

    # A rack displaces the inboard end of every steered link.
    inboard = inboard.copy()
    for i, link in enumerate(corner.links):
        if link.steered:
            inboard[i] = inboard[i] + np.array([0.0, rack_travel, 0.0])

    def residual(x: np.ndarray) -> np.ndarray:
        pose = Pose(x[:3], x[3:], ref)
        moved = pose.apply(outboard)
        r = np.linalg.norm(moved - inboard, axis=1) - lengths
        wc = pose.apply(ref)
        return np.concatenate([r, [wc[2] - wheel_z]])

    if x0 is None:
        x0 = np.zeros(6)
        x0[2] = wheel_z - ref[2]

    # method="trf", NOT "lm".
    #
    # MINPACK's Levenberg-Marquardt sizes its initial trust region from
    # ||x0||. Warm-starting a sweep hands it the previous solution, and at zero
    # wheel travel that solution is EXACTLY the zero vector (the design pose).
    # The trust region then has zero radius and lm can never take a step -- it
    # returns x0 unchanged, silently, with a large residual. Every subsequent
    # step inherits the stuck state.
    #
    # Symptom when this was live: droop solved perfectly, bump froze at the
    # design pose, because the sweep passed through zero travel on the way up.
    # trf has no such degeneracy and converged in 4 evaluations where lm took 47.
    sol = least_squares(residual, x0, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    rn = float(np.linalg.norm(sol.fun))

    # A warm start can also land in the mirrored assembly (upright reflected
    # through the link plane), which satisfies every length constraint and is
    # physically wrong. Retry cold rather than return a bad root quietly.
    if rn >= tol:
        cold = np.zeros(6)
        cold[2] = wheel_z - ref[2]
        retry = least_squares(residual, cold, method="trf",
                              xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.linalg.norm(retry.fun) < rn:
            sol, rn = retry, float(np.linalg.norm(retry.fun))

    return Pose(sol.x[:3], sol.x[3:], ref), rn, rn < tol


#: Substring that names each arm in a `Link`'s auto-generated
#: "<inboard>-><outboard>" name. The point names come from the topology tables
#: in geometry.py, which have used `IB_Upp*` / `IB_Low*` since the first sheet.
_ARM_TAG = {"upper_wishbone": "Upp", "lower_wishbone": "Low"}


def wishbone_links(corner: CornerGeometry, which: str) -> Optional[List]:
    """The two unsteered links of one arm, if they form a real wishbone.

    `which` is "upper_wishbone" or "lower_wishbone" -- the same strings
    `Actuator.rod_pickup` uses, so a declared pickup can be looked up directly
    with no second naming scheme to keep in step.

    Returns None unless that arm is exactly two unsteered links SHARING an
    outboard point. Sharing is the load-bearing part of the test: the rear
    five-link's two lower links run to different uprights points, so they are
    not a wishbone, have no single ball joint, and cannot carry a rod load the
    way this construction assumes.
    """
    tag = _ARM_TAG.get(which)
    if tag is None:
        return None
    arm = [l for l in corner.links if tag in l.name and not l.steered]
    if len(arm) != 2:
        return None
    if not np.allclose(arm[0].outboard, arm[1].outboard, atol=1e-9):
        return None
    return arm


def upper_wishbone_links(corner: CornerGeometry) -> Optional[List]:
    """The upper arm, i.e. `wishbone_links(corner, "upper_wishbone")`.

    Kept as a named helper because "is there an identifiable upper wishbone"
    is asked in several places; it no longer implies anything about where the
    rod picks up. That question is answered by `corner.actuator.rod_pickup`
    alone -- see `rod_outboard_point`.
    """
    return wishbone_links(corner, "upper_wishbone")


def rod_pickup_arm(corner: CornerGeometry) -> Optional[List]:
    """The arm the rod picks up on, or None for an upright-mounted rod.

    Raises when the declared arm is not a wishbone in this topology, rather
    than quietly falling back to another construction -- that fallback is
    exactly the failure this field was added to remove.
    """
    pickup = corner.actuator.rod_pickup
    if pickup == "upright":
        return None
    arm = wishbone_links(corner, pickup)
    if arm is None:
        raise ValueError(
            f"{corner.name}: rod_pickup={pickup!r}, but this topology has no "
            f"such wishbone -- an arm must be exactly two unsteered links "
            f"sharing one outboard ball joint. A five-link's split lower links "
            f"are not a wishbone and cannot carry a rod this way; declare "
            f"rod_pickup: upright, or model the rod on the arm that is one.")
    return arm


def rod_outboard_point(corner: CornerGeometry, pose: Pose) -> np.ndarray:
    """Current position of the push/pullrod's outboard pickup.

    >>> WHICH BODY CARRIES IT IS DECLARED, NOT INFERRED. <<<

    `corner.actuator.rod_pickup` says which body the outboard end bolts to.
    This function used to INFER it -- upper wishbone whenever one could be
    identified, upright otherwise -- which silently mis-modelled the ordinary
    layout of an upright-mounted rod on a car that also has an upper wishbone.
    Measured on the NFR26 front corner at 25 mm of bump, the two constructions
    put the pickup 4.05 mm apart in z, and the rod length 2.7 mm apart
    (0.36697 m arm-carried against 0.36429 m upright-carried).

    NFR26 declares `upper_wishbone` at both ends, and the hardpoints agree: the
    pickup sits 61 mm (front) / 78 mm (rear) from the upper ball joint and
    24 mm / 21 mm out of the wishbone plane -- a bracket standing off the arm --
    at 81% of the ball joint's perpendicular distance from the arm's inboard
    axis.

    THE CONSTRUCTION -- ARM-AGNOSTIC
    --------------------------------
    A wishbone with two chassis pickups and one ball joint has exactly ONE
    degree of freedom: rotation about the line joining its inboard points. The
    pose solver already gives the ball joint's new position, and that fixes the
    rotation angle. So, for EITHER arm:

        1. take the arm's inboard axis a through p1, p2;
        2. find the angle that carries the ball joint from its design position
           to the solved one, measured perpendicular to a;
        3. apply that same rotation to the rod pickup.

    Step 2 is exact rather than fitted -- the ball joint's perpendicular
    distance from the axis is invariant, so one angle reproduces it. Nothing in
    it is specific to the upper arm: it needs that arm's two inboard points and
    its shared outboard ball joint, and no more.

    For `rod_pickup: upright` the pickup is a point on the upright and simply
    rides the upright pose, through all six of its degrees of freedom.
    """
    rod0 = np.asarray(corner.actuator.outboard, dtype=float)
    arm = rod_pickup_arm(corner)
    if arm is None:
        return pose.apply(rod0)

    p1 = np.asarray(arm[0].inboard, dtype=float)
    p2 = np.asarray(arm[1].inboard, dtype=float)
    a = p2 - p1
    n = np.linalg.norm(a)
    if n < 1e-12:                       # coincident pickups: no axis to rotate about
        return pose.apply(rod0)
    a = a / n

    ball0 = np.asarray(arm[0].outboard, dtype=float)
    ball = pose.apply(ball0)

    def perp(v):
        return v - np.dot(v, a) * a

    v0, v1 = perp(ball0 - p1), perp(ball - p1)
    if np.linalg.norm(v0) < 1e-12 or np.linalg.norm(v1) < 1e-12:
        return pose.apply(rod0)

    theta = float(np.arctan2(np.dot(a, np.cross(v0, v1)), np.dot(v0, v1)))
    return p1 + rodrigues(a * theta) @ (rod0 - p1)


def corner_state(corner: CornerGeometry, wheel_z: float, rack_travel: float = 0.0,
                 left: bool = False, x0: Optional[np.ndarray] = None) -> CornerState:
    """Solve and derive camber, toe, rod length at one operating point."""
    pose, rn, ok = solve_pose(corner, wheel_z, rack_travel, x0)

    axis0 = corner.spindle_axis
    if axis0 is None:
        axis0 = _nominal_spindle_axis(0.0, 0.0, left)
    axis = pose.matrix @ np.asarray(axis0, dtype=float)

    camber, toe = _camber_toe_from_axis(axis, left)
    wc = pose.apply(corner.wheel_centre)

    rod_len = float("nan")
    if corner.actuator is not None:
        rod_out = rod_outboard_point(corner, pose)
        rod_len = float(np.linalg.norm(rod_out - corner.actuator.inboard))

    return CornerState(wheel_z=wheel_z, rack_travel=rack_travel, pose=pose,
                       wheel_centre=wc, spindle_axis=axis,
                       camber_rad=camber, toe_rad=toe, rod_length=rod_len,
                       residual_norm=rn, converged=ok)


def sweep_heave(corner: CornerGeometry, travel: np.ndarray,
                rack_travel: float = 0.0, left: bool = False) -> List[CornerState]:
    """Sweep wheel vertical travel, warm-starting each step from the last.

    Warm-starting matters: a cold Newton start at large travel can converge to
    the mirrored assembly (the upright flipped through the link plane), which is
    a valid root of the length constraints and completely wrong physically.
    """
    z0 = float(corner.wheel_centre[2])
    states: List[CornerState] = []
    x0 = None
    for dz in travel:
        st = corner_state(corner, z0 + float(dz), rack_travel, left, x0)
        x0 = np.concatenate([st.pose.translation, st.pose.rotation_vector])
        states.append(st)
    return states


def rod_motion_ratio(states: List[CornerState]) -> np.ndarray:
    """d(rod length) / d(wheel travel), as a function of travel.

    >>> THIS IS NOT THE SPRING/DAMPER MOTION RATIO. <<<

    It is the ratio from wheel to PUSH/PULLROD only. On NFR26 both corners run
    a rocker, which applies a further leverage between rod and damper, so the
    damper motion ratio is this value times the rocker ratio. Computed here for
    the real geometry: 0.710 front, 0.495 rear -- whereas the team reports
    wheel-to-damper ratios "close to 1.0" at both ends, which is consistent
    with a rocker ratio above one making up the difference. (These read
    0.815 / 0.573 while the rod pickup was wrongly carried by the upright
    pose; see `rod_outboard_point`.)

    Wheel rate scales with the SQUARE of the damper motion ratio, so conflating
    the two would put spring and damper rates out by a factor of ~1.5-3.
    `damper_motion_ratio` (rocker solve) is the next piece of work; until it
    exists, do not use this value for wheel-rate or damping calculations.

    Returned as a CURVE, not a constant -- it varies with travel for any real
    rocker geometry (docs/07 SS2). Measured variation here: front 0.787 to
    0.847 over +/-30 mm (progressive), rear 0.577 to 0.569 (near-constant).
    """
    z = np.array([s.wheel_centre[2] for s in states])
    L = np.array([s.rod_length for s in states])
    return np.gradient(L, z)


# Deprecated alias kept deliberately short-lived; remove once the rocker solve
# lands and the real damper motion ratio is available.
motion_ratio = rod_motion_ratio


# ---------------------------------------------------------------------------
# Instant axis, front-view instant centre, roll centre
# ---------------------------------------------------------------------------
#
# EARLIER APPROACH WAS WRONG. A first version sliced each control-arm PLANE at
# the wheel-centre plane and intersected the resulting lines. That is only
# valid when the inboard pivot axes are parallel to the vehicle x-axis, which
# is true at neither end of this car, and it produced FVSA/SVSA/roll-centre
# figures 5-10% off.
#
# The correct construction takes the upright's actual instantaneous motion.
# With five constraints the upright has one degree of freedom, so its motion is
# a SCREW: a rotation about some axis in space, possibly with translation along
# it. Everything else -- front-view instant centre, roll centre, swing-arm
# lengths -- follows from that axis.
#
# This is also topology-generic. It never asks how the corner is built, only
# how it moves, so it works unchanged for the double wishbone and the
# five-link.

@dataclass(frozen=True)
class VelocityField:
    """Instantaneous motion of the upright, per unit wheel travel.

        v(p) = v0 + omega x (p - p0)
    """

    omega: np.ndarray
    p0: np.ndarray
    v0: np.ndarray

    def at(self, p) -> np.ndarray:
        return self.v0 + np.cross(self.omega, np.asarray(p, dtype=float) - self.p0)


@dataclass(frozen=True)
class InstantAxis:
    """Instantaneous screw axis of the upright relative to the chassis."""

    point: np.ndarray
    direction: np.ndarray
    pitch: float
    omega_magnitude: float


def velocity_field(corner: CornerGeometry, wheel_z: float,
                   rack_travel: float = 0.0, eps: float = 1e-5) -> VelocityField:
    """Upright velocity field per unit wheel travel, by central difference.

    Numerical rather than analytic on purpose: it depends only on the pose
    solver, so it cannot drift out of agreement with the kinematics the way a
    parallel closed-form derivation would.
    """
    probes = np.array([l.outboard for l in corner.links]
                      + [np.asarray(corner.wheel_centre, dtype=float)])

    p_lo, _, ok_lo = solve_pose(corner, wheel_z - eps, rack_travel)
    p_hi, _, ok_hi = solve_pose(corner, wheel_z + eps, rack_travel)
    if not (ok_lo and ok_hi):
        raise ValueError(
            f"{corner.name}: pose did not converge either side of z={wheel_z:.4f} "
            f"while differencing for the velocity field."
        )

    lo, hi = p_lo.apply(probes), p_hi.apply(probes)
    vel = (hi - lo) / (2.0 * eps)

    # v - v0 = omega x (p - p0). Least squares over all probes eliminates v0.
    p0, v0 = lo[-1], vel[-1]
    A, b = [], []
    for p, v in zip(lo[:-1], vel[:-1]):
        d = p - p0
        A.append(np.array([[0.0, d[2], -d[1]],
                           [-d[2], 0.0, d[0]],
                           [d[1], -d[0], 0.0]]))
        b.append(v - v0)
    omega, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    return VelocityField(omega=omega, p0=p0, v0=v0)


def instant_axis(corner: CornerGeometry, wheel_z: float,
                 rack_travel: float = 0.0, eps: float = 1e-5) -> InstantAxis:
    """Instantaneous screw axis. Reported for diagnostics.

    NOTE: this is NOT what the front-view instant centre is built from -- see
    `front_view_instant_centre`.
    """
    vf = velocity_field(corner, wheel_z, rack_travel, eps)
    w2 = float(vf.omega @ vf.omega)
    if w2 < 1e-12:
        return InstantAxis(np.full(3, np.inf), np.array([1.0, 0.0, 0.0]),
                           float("inf"), 0.0)
    return InstantAxis(point=vf.p0 + np.cross(vf.omega, vf.v0) / w2,
                       direction=vf.omega / np.sqrt(w2),
                       pitch=float(vf.v0 @ vf.omega) / w2,
                       omega_magnitude=float(np.sqrt(w2)))


def front_view_instant_centre(vf: VelocityField) -> np.ndarray:
    """Front-view instant centre, (y, z) in metres.

    WHY NOT THE 3D AXIS. An earlier version intersected the instantaneous screw
    axis with the transverse plane through the contact patch. That is wrong,
    and badly so on this car -- it produced a front roll centre of 269 mm.

    The front view is a PROJECTION. Projected into the y-z plane the upright's
    motion is a planar rigid motion whose rotation rate is the x-component of
    omega alone; the y and z components describe caster change and steer, which
    are invisible in the front view. The front-view instant centre is where the
    PROJECTED velocity vanishes:

        v_y = v0_y - omega_x * (z - z0) = 0   ->  z = z0 + v0_y / omega_x
        v_z = v0_z + omega_x * (y - y0) = 0   ->  y = y0 - v0_z / omega_x

    Sanity check on NFR26 front: omega_x = -0.562 rad/m from the measured camber
    gain, so 1/omega_x = 1779 mm, and the instant centre lands 1779 mm inboard
    of the contact patch -- matching the sheet's FVSA length of 1778.7 mm.

    The full 3D axis is still meaningful (see `instant_axis`), just not for
    this.
    """
    wx = float(vf.omega[0])
    if abs(wx) < 1e-9:
        return np.array([np.inf, vf.p0[2]])   # no front-view rotation
    return np.array([vf.p0[1] - vf.v0[2] / wx,
                     vf.p0[2] + vf.v0[1] / wx])


def side_view_instant_centre(vf: VelocityField) -> np.ndarray:
    """Side-view instant centre, (x, z) in metres.

    Exact analogue of `front_view_instant_centre`, and the same warning
    applies: this is a PROJECTION, not an intersection of the 3D screw axis
    with a plane. Projected into the x-z plane the upright's motion is a planar
    rigid motion whose rotation rate is the y-component of omega alone; the x
    and z components describe camber and steer change, invisible in side view.
    The side-view instant centre is where the PROJECTED velocity vanishes:

        v_x = v0_x + omega_y*(z - z0) = 0  ->  z = z0 - v0_x / omega_y
        v_z = v0_z - omega_y*(x - x0) = 0  ->  x = x0 + v0_z / omega_y

    This is what anti-dive, anti-lift and anti-squat are built on.
    """
    wy = float(vf.omega[1])
    if abs(wy) < 1e-9:
        return np.array([np.inf, vf.p0[2]])   # no side-view rotation
    return np.array([vf.p0[0] + vf.v0[2] / wy,
                     vf.p0[2] - vf.v0[0] / wy])


def side_view_swing_arm(corner: CornerGeometry, wheel_z: float,
                        reference: np.ndarray,
                        rack_travel: float = 0.0) -> Tuple[float, float]:
    """(length, angle_rad) of the side-view swing arm from `reference`.

    `reference` is the point through which longitudinal force is fed into the
    linkage, and WHICH POINT THAT IS DEPENDS ON THE HARDWARE:

      * reaction at the UPRIGHT (outboard brake caliper, hub motor) -> the
        contact patch, because the brake/drive couple is internal to the
        upright free body and the only external longitudinal force on it
        arrives at the ground;
      * reaction at the CHASSIS (inboard brake, inboard motor + halfshafts)
        -> the wheel centre, because the couple crosses the cut and only the
        bearing force passes through the links.

    This is the same free-body distinction `forces.CornerLoads` documents, and
    it is why NFR26's sheet carries two anti-squat figures (7.61% with hub
    motors, 10.28% without). Positive angle = instant centre above `reference`.
    """
    vf = velocity_field(corner, wheel_z, rack_travel)
    x_ic, z_ic = side_view_instant_centre(vf)
    if not np.isfinite(x_ic):
        return float("inf"), 0.0
    dx = x_ic - float(reference[0])
    dz = z_ic - float(reference[2])
    return float(abs(dx)), float(np.arctan2(dz, abs(dx)))


def roll_centre_height(corner: CornerGeometry, wheel_z: float,
                       contact_patch: np.ndarray,
                       rack_travel: float = 0.0) -> float:
    """Kinematic roll centre height above ground, metres.

    The roll centre lies on the line from the contact patch through the
    front-view instant centre, taken at the vehicle centreline (y = 0).

    Closed-form check in the tests: parallel equal-length arms put the instant
    centre at infinity, the line horizontal, and the roll centre at ground.
    """
    vf = velocity_field(corner, wheel_z, rack_travel)
    y_ic, z_ic = front_view_instant_centre(vf)
    y_cp, z_cp = float(contact_patch[1]), float(contact_patch[2])

    if not np.isfinite(y_ic):
        return z_cp

    dy = y_ic - y_cp
    if abs(dy) < 1e-9:
        return float("inf")
    return float(z_cp + (0.0 - y_cp) * (z_ic - z_cp) / dy)


def front_view_swing_arm(corner: CornerGeometry, wheel_z: float,
                         contact_patch: np.ndarray,
                         rack_travel: float = 0.0) -> Tuple[float, float]:
    """(length, angle_rad) of the front-view swing arm, from the contact patch."""
    vf = velocity_field(corner, wheel_z, rack_travel)
    y_ic, z_ic = front_view_instant_centre(vf)
    if not np.isfinite(y_ic):
        return float("inf"), 0.0
    dy = y_ic - float(contact_patch[1])
    dz = z_ic - float(contact_patch[2])
    return float(abs(dy)), float(np.arctan2(dz, abs(dy)))

"""Link and bearing force extraction -- goal G1.

A POST-PROCESSOR, NOT A SOLVER TIER
-----------------------------------
This consumes a converged corner state from ANY tier (steady-state, QSS lap,
or transient) plus the loads at the contact patch, and returns axial forces in
every suspension member. Written once, used everywhere. See docs/00 SS2.

THE 6x6
-------
Free body: upright + wheel + brake + hub assembly, cut at the suspension links.
Every link is a two-force member (spherical joints both ends, massless), which
is a good model for FSAE tube-and-rod-end suspension.

    unknowns : 5 link axial forces + 1 push/pullrod force   = 6
    equations: 3 force + 3 moment equilibrium                = 6

        [  u_1     ...  u_6   ] [f_1]     [F_ext]
        [ r_1xu_1  ... r_6xu_6] [...]  = -[M_ext]
                                [f_6]

Link unit vectors are taken at the CURRENT deflected attitude, not the design
position -- that is why this needs a solved pose rather than raw hardpoints.

cond(A) is returned as a free diagnostic. A high value means the geometry is
near-singular, which is real design feedback worth surfacing.

DRIVE TORQUE REACTION -- ARCHITECTURE DEPENDENT
-----------------------------------------------
NFR26 runs quad hub motors, so the motor stator is mounted to the UPRIGHT and
the full drive torque reacts through the suspension links. With inboard drive
(motor/diff on the chassis, halfshafts out) it does not. Getting this backwards
under-designs the uprights and links on a hub-motor car.

The team's own geometry sheet carries two anti-squat figures for exactly this
reason (7.61% with hub motors vs 10.28% without), so the distinction is real
and already understood on the vehicle side. See docs/04 SSB.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .geometry import CornerGeometry
from .kinematics import CornerState, rod_outboard_point, rod_pickup_arm

__all__ = ["CornerLoads", "LinkForces", "WishboneMountedRodForces",
           "solve_link_forces", "solve_wishbone_mounted_rod", "bearing_loads",
           "wheel_reaction_torque"]


def wheel_reaction_torque(Fx: float, loaded_radius: float) -> float:
    """Torque about the spindle axis that the brake or motor must apply.

        T = -Fx * r_loaded

    Use this rather than writing the expression at the call site. The sign is
    the same relation for braking and driving -- braking (Fx < 0) needs a
    positive torque, driving (Fx > 0) a negative one -- and hand-writing it
    per case is an easy way to flip one of them. Doing exactly that made a
    hub-motor load case look milder than inboard drive, which is backwards.

    Valid for steady state (no wheel angular acceleration). In the transient
    model, add I_wheel * omega_dot.
    """
    return -Fx * loaded_radius


@dataclass
class CornerLoads:
    """External loads acting on one corner assembly. SI, ISO 8855.

    Contact-patch forces are what the TIRE exerts ON THE CAR.
    """

    Fx: float = 0.0
    Fy: float = 0.0
    Fz: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0

    brake_torque: float = 0.0
    """Braking torque applied to the wheel about the spindle axis.

    Whether this loads the links depends ENTIRELY on where the caliper is --
    see `caliper_on_upright` and the free-body note on that field."""

    drive_torque: float = 0.0
    """Drive torque applied to the wheel about the spindle axis."""

    caliper_on_upright: bool = True
    """NFR26: True.

    THE FREE BODY, because the intuitive answer is the wrong way round:

      * Caliper ON THE UPRIGHT (outboard). The caliper and the rotor are BOTH
        inside the cut, so the brake couple is INTERNAL and cancels. Nothing is
        added -- and the links therefore carry the FULL contact-patch moment,
        including its Fx * r_loaded component about the spindle axis.

      * Caliper INBOARD. The rotor sits on the chassis side of the cut, so
        braking torque arrives through a halfshaft as an EXTERNAL moment on the
        assembly. In steady braking it very nearly cancels the contact-patch
        spindle moment, so the links carry LESS.

    So upright-mounted brakes produce the HIGHER link loads. An earlier version
    of this module applied the correction to the wrong case and reported
    inboard calipers as the more severe condition -- exactly backwards, and it
    would have under-designed NFR26's front links, which are the
    upright-mounted case."""

    drive_reacts_on_upright: bool = True
    """NFR26: True (quad hub motors -- stator bolted to the upright).

    Identical free-body logic to the brake. A hub motor's couple is internal to
    the cut, so the links carry the full contact-patch moment. Inboard drive
    (chassis motor/diff + halfshafts) puts the torque across the cut and
    relieves the links."""

    unsprung_mass: float = 0.0
    unsprung_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Inertial term. Small in steady state; NOT small over a kerb, which is
    where peak link loads actually occur (docs/04 SSB.2)."""

    gravity: float = 9.80665


@dataclass
class WishboneMountedRodForces:
    """Result when the push/pullrod picks up on a wishbone, not the upright.

    NFR26 is this case at BOTH ends, on the UPPER arm. It is not the textbook
    6x6, because a wishbone carrying a rod load is no longer a two-force member
    -- it carries BENDING, and its outboard ball-joint reaction is a general 3D
    force rather than an axial one.

    That distinction is the whole point of modelling it properly: the structures
    team needs to know the loaded wishbone is a bending member, and a model that
    reports two clean axial forces in it would say otherwise.

    The roles SWAP for a lower-wishbone-mounted rod: the lower arm bends and the
    upper links become the clean two-force pair. `loaded_arm` says which it is,
    so nothing downstream has to assume.
    """

    ball_joint_force: np.ndarray
    """3D force the loaded wishbone exerts on the upright, N."""
    free_link_forces: np.ndarray
    """Axial forces in the links that remain two-force members -- the OTHER
    arm's two links. Named `lower_link_*` until the pickup became configurable;
    for `lower_wishbone` these are the UPPER links, so the old name was a
    statement about NFR26 rather than about the model."""
    free_link_names: List[str]
    tie_rod_force: float
    rod_force: float
    """Axial force in the push/pullrod, from wishbone moment balance."""
    condition_number: float
    residual: float
    loaded_arm: str = "upper_wishbone"
    """Which arm carries the rod, from `Actuator.rod_pickup`."""

    def report(self) -> str:
        arm = self.loaded_arm.replace("_", " ")
        lines = [
            f"{arm} ball joint force  {np.round(self.ball_joint_force, 1)} N "
            f"(|F| = {np.linalg.norm(self.ball_joint_force):.1f})",
            f"  -> {arm} carries BENDING; not a two-force member",
        ]
        for n, f in zip(self.free_link_names, self.free_link_forces):
            lines.append(f"{n:<28} {f:10.1f} N  {'tension' if f > 0 else 'compression'}")
        lines.append(f"{'tie rod':<28} {self.tie_rod_force:10.1f} N")
        lines.append(f"{'push/pullrod':<28} {self.rod_force:10.1f} N")
        lines.append(f"cond(A) = {self.condition_number:.1f}, residual = {self.residual:.2e} N")
        return "\n".join(lines)


@dataclass
class LinkForces:
    """Axial force in each member. Positive = TENSION."""

    names: List[str]
    forces: np.ndarray
    condition_number: float
    residual: float
    reference: np.ndarray

    def __getitem__(self, name: str) -> float:
        return float(self.forces[self.names.index(name)])

    def as_dict(self) -> Dict[str, float]:
        return {n: float(f) for n, f in zip(self.names, self.forces)}

    @property
    def max_tension(self) -> float:
        return float(np.max(self.forces))

    @property
    def max_compression(self) -> float:
        return float(np.min(self.forces))

    def report(self) -> str:
        lines = [f"{'member':<28} {'force N':>10}  state"]
        for n, f in zip(self.names, self.forces):
            lines.append(f"{n:<28} {f:10.1f}  {'tension' if f > 0 else 'compression'}")
        lines.append(f"cond(A) = {self.condition_number:.1f}, residual = {self.residual:.2e} N")
        return "\n".join(lines)


def _spindle_axis(corner: CornerGeometry, state: CornerState) -> np.ndarray:
    a = state.spindle_axis
    return np.asarray(a, dtype=float) / np.linalg.norm(a)


def _spin_axis_vehicle_frame(axis: np.ndarray) -> np.ndarray:
    """Wheel spin axis oriented consistently in the VEHICLE frame (toward +y).

    >>> DO NOT USE THE RAW OUTBOARD SPINDLE AXIS FOR A TORQUE. <<<

    `_spindle_axis` points OUTBOARD, so it flips sign between the two sides of
    the car. `wheel_reaction_torque` returns a side-INDEPENDENT scalar, because
    the couple the contact patch applies about the wheel centre is the same
    vector on both sides: for 1.8 g of braking on this car it is [0, +498.4, 0]
    at FR and [0, +498.4, 0] at FL, measured. Multiplying that side-independent
    scalar by a side-flipping axis therefore cancels the couple on one side and
    DOUBLES it on the other.

    Measured before this helper existed, inboard caliper, 1.8 g braking:
        FR  net spindle moment    +0.09 N.m   peak link force   2474 N  (right)
        FL  net spindle moment  +996.67 N.m   peak link force  11296 N  (wrong)
    -- 4.6x the right corner, and 1.6x the genuinely severe upright-mounted
    case, i.e. the correction made the load worse instead of relieving it.

    Orienting to +y rather than keying off a `left` flag keeps this a property
    of the geometry, which is what it is; `CornerState` does not carry the side.
    A wheel spin axis is within a few degrees of lateral by construction, so
    the test on `axis[1]` cannot be marginal -- but it is asserted rather than
    assumed.
    """
    if abs(axis[1]) < 0.5:
        raise ValueError(
            f"spindle axis {np.round(axis, 4)} is more than 60 deg from "
            f"lateral; it cannot be a wheel spin axis, so orienting it in the "
            f"vehicle frame is not meaningful. Check the corner's "
            f"spindle_axis / camber / toe inputs.")
    return axis if axis[1] >= 0.0 else -axis


def solve_link_forces(corner: CornerGeometry, state: CornerState,
                      loads: CornerLoads,
                      contact_patch: Optional[np.ndarray] = None,
                      rod_on_upright: bool = True) -> LinkForces:
    """Solve the 6x6 upright equilibrium.

    Args:
        corner: geometry (design hardpoints).
        state: solved pose from the kinematics solver.
        loads: external loads at the contact patch.
        contact_patch: application point. Defaults to directly below the
            current wheel centre at z = 0, which is exact for zero camber and
            good to a few mm otherwise.
        rod_on_upright: True if the push/pullrod picks up on the upright. Set
            False when it mounts to a control arm -- the rod force then does
            not appear in the upright free body and the system is no longer
            6x6. Deliberately an ARGUMENT rather than a read of
            `corner.actuator.rod_pickup`, so this solver can still be exercised
            against a corner declared the other way; production callers
            (`component_forces.extract_components`) dispatch on the
            declaration.

    Returns:
        LinkForces, positive in tension.
    """
    if corner.actuator is None:
        raise ValueError(f"{corner.name}: no actuator; the 6th unknown is undefined")
    if not rod_on_upright:
        raise ValueError(
            "rod is mounted on a wishbone, not the upright -- use "
            "solve_wishbone_mounted_rod() instead. The upright 6x6 is the wrong "
            "free body for that layout (NFR26 is the wishbone-mounted case at "
            "both ends)."
        )

    pose = state.pose
    ref = np.asarray(state.wheel_centre, dtype=float)

    # Geometry at the CURRENT attitude.
    members = list(corner.links)
    outboard = [pose.apply(l.outboard) for l in members]
    inboard = [l.inboard for l in members]
    names = [l.name for l in members]

    rod_out = pose.apply(corner.actuator.outboard)
    outboard.append(rod_out)
    inboard.append(np.asarray(corner.actuator.inboard, dtype=float))
    names.append(f"{corner.actuator.kind}")

    A = np.zeros((6, 6))
    for j, (ob, ib) in enumerate(zip(outboard, inboard)):
        u = ob - ib
        u = u / np.linalg.norm(u)
        A[:3, j] = u
        A[3:, j] = np.cross(ob - ref, u)

    # ---- external force ----
    if contact_patch is None:
        contact_patch = np.array([ref[0], ref[1], 0.0])

    F_ext = np.array([loads.Fx, loads.Fy, loads.Fz], dtype=float)

    # Unsprung weight and inertia. d'Alembert: the inertial term opposes the
    # acceleration, so it enters the equilibrium as -m*a.
    F_ext = F_ext + np.array([0.0, 0.0, -loads.unsprung_mass * loads.gravity])
    F_ext = F_ext - loads.unsprung_mass * np.asarray(loads.unsprung_accel, dtype=float)

    # ---- external moment about the reference ----
    M_ext = np.cross(contact_patch - ref, np.array([loads.Fx, loads.Fy, loads.Fz]))
    M_ext = M_ext + np.array([loads.Mx, loads.My, loads.Mz])

    # Brake / drive torque enters ONLY when it crosses the cut, i.e. when the
    # caliper or motor is INBOARD and a halfshaft carries the torque. When
    # either is mounted on the upright the couple is internal to the free body
    # and contributes nothing -- leaving the links to carry the full
    # contact-patch moment. See CornerLoads.caliper_on_upright.
    # SUBTRACT along a vehicle-frame-consistent spin axis. When the caliper or
    # motor is inboard the couple is reacted at the chassis, so it never passes
    # through the upright and must be REMOVED from its free body. See
    # _spin_axis_vehicle_frame for why the raw outboard axis is wrong here.
    spin = _spin_axis_vehicle_frame(_spindle_axis(corner, state))
    if not loads.caliper_on_upright:
        M_ext = M_ext - spin * loads.brake_torque
    if not loads.drive_reacts_on_upright:
        M_ext = M_ext - spin * loads.drive_torque

    rhs = -np.concatenate([F_ext, M_ext])
    f_compression = np.linalg.solve(A, rhs)

    # THE SOLVE IS COMPRESSION-POSITIVE. NEGATE ONCE, HERE.
    #
    # `u` points inboard -> outboard, so a member in TENSION pulls the upright
    # back toward the chassis and exerts `-f_t * u` on it. Equilibrium is then
    # `sum(f_t * u) = +F_ext`, whereas this system solves `A f = -F_ext` --
    # so what comes out of `solve` is `-f_t`, i.e. compression-positive.
    #
    # Verified numerically rather than argued: reading the raw solution as
    # tension leaves a 4.6 kN force residual and 725 N.m of moment on the
    # upright free body; reading it as compression closes to 1e-13.
    #
    # Everything downstream -- the `LinkForces` docstring, `max_tension`,
    # `max_compression`, and the tension/compression labels in `report()` --
    # states the tension-positive convention, so the flip belongs here rather
    # than in five places that each have to remember it. Pinned by
    # test_link_force_sign_is_tension_positive.
    f = -f_compression

    return LinkForces(names=names, forces=f,
                      condition_number=float(np.linalg.cond(A)),
                      residual=float(np.linalg.norm(A @ f_compression - rhs)),
                      reference=ref)


def bearing_loads(corner: CornerGeometry, state: CornerState, loads: CornerLoads,
                  inner_offset: float, outer_offset: float,
                  contact_patch: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Radial and axial load on each wheel bearing row.

    Cuts between hub and upright. The contact-patch force acting at the loaded
    radius produces a moment about the spindle axis line, and that moment is
    what splits load between the rows -- it can put the inner row in the
    OPPOSITE radial direction to the outer.

    Args:
        inner_offset, outer_offset: signed positions of the bearing centre
            planes along the spindle axis, relative to the wheel centre.

    Note: predicted magnitudes scale roughly as 1/(bearing spacing), so they
    are only as good as those offsets. NFR26's are currently ESTIMATED at
    +/-35 mm -- see config `bearings`.
    """
    if abs(outer_offset - inner_offset) < 1e-6:
        raise ValueError("bearing rows are coincident; spacing must be non-zero")

    ref = np.asarray(state.wheel_centre, dtype=float)
    axis = _spindle_axis(corner, state)
    if contact_patch is None:
        contact_patch = np.array([ref[0], ref[1], 0.0])

    F = np.array([loads.Fx, loads.Fy, loads.Fz], dtype=float)
    M = np.cross(contact_patch - ref, F) + np.array([loads.Mx, loads.My, loads.Mz])

    # Split into components along and perpendicular to the spindle axis.
    F_axial = float(F @ axis)
    F_radial = F - F_axial * axis
    M_radial = M - float(M @ axis) * axis      # the axial part is reacted by brake/drive

    # Moment balance about each row in turn.
    #
    # The two rows carry `R_inner + R_outer = F_radial` and must also react the
    # radial moment: `d_i x R_inner + d_o x R_outer = +M_radial`, with the row
    # positions `d = offset * axis`. Solving that pair gives
    #
    #     R_outer = -(axis x M_radial) / span  -  (inner_offset / span) * F_radial
    #
    # The moment term was previously POSITIVE, which balances -M_radial instead
    # of +M_radial and leaves a residual of exactly 2*|M_radial| (measured:
    # 649.9 N.m for 1650 N lateral at the loaded radius). With the symmetric
    # +/-35 mm offsets in the config that is indistinguishable from swapping
    # the two rows, so bearing life was being computed for the wrong row; with
    # asymmetric offsets it is not a swap and both magnitudes were wrong.
    # Pinned by test_bearing_rows_react_the_radial_moment.
    span = outer_offset - inner_offset
    R_outer = (-np.cross(axis, M_radial) / span) + F_radial * (-inner_offset / span)
    R_inner = F_radial - R_outer

    return {
        "inner_radial": R_inner,
        "outer_radial": R_outer,
        "inner_radial_mag": float(np.linalg.norm(R_inner)),
        "outer_radial_mag": float(np.linalg.norm(R_outer)),
        "axial": F_axial,
        "axis": axis,
    }


def solve_wishbone_mounted_rod(
        corner: CornerGeometry, state: CornerState, loads: CornerLoads,
        loaded_link_names: Optional[List[str]] = None,
        contact_patch: Optional[np.ndarray] = None) -> WishboneMountedRodForces:
    """Force extraction when the push/pullrod picks up on a WISHBONE.

    Which wishbone comes from `corner.actuator.rod_pickup`; pass
    `loaded_link_names` to override. NFR26 is the upper-arm case at both ends,
    so this -- not `solve_link_forces` -- is the production path for that car.
    An `upright` pickup is not this free body at all: use `solve_link_forces`.

    NOTHING BELOW IS SPECIFIC TO THE UPPER ARM. The derivation needs the loaded
    arm's two inboard points and its shared outboard ball joint, and no more,
    so a lower-wishbone-mounted rod (common on pullrod cars) runs through the
    same two free bodies with the roles swapped.

    WHY THE 6x6 DOES NOT APPLY
    --------------------------
    Decomposing an A-arm into two axial links is exact ONLY while the arm is a
    two-force member: spherical outboard joint, two chassis pickups, and no
    other load. Hanging a pushrod off the arm breaks that. The arm then carries
    bending, and its ball-joint reaction on the upright is a general 3D force.

    Reporting an arm like that as two clean axial forces would tell the
    structures team it is a strut when it is actually a bending member.

    TWO FREE BODIES
    ---------------
    1. UPRIGHT. Unknowns are the 3D ball-joint force of the LOADED arm (3), the
       other arm's two virtual-link axial forces (2, still valid -- nothing
       hangs off it), and the tie rod (1). Six unknowns, six equilibrium
       equations.

    2. THE LOADED WISHBONE. Moments about its inboard axis. The two chassis pickups
       cannot react moment about the line joining them, so that single scalar
       equation yields the rod force directly:

           [(r_ball - r_axis) x (-F_ball)] . a
         + f_rod * [(r_rod - r_axis) x u_rod] . a  =  0
    """
    if corner.actuator is None:
        raise ValueError(f"{corner.name}: no actuator defined")

    pose = state.pose
    ref = np.asarray(state.wheel_centre, dtype=float)

    loaded_arm = corner.actuator.rod_pickup
    if loaded_link_names is None:
        arm = rod_pickup_arm(corner)
        if arm is None:
            raise ValueError(
                f"{corner.name}: rod_pickup='upright', so no wishbone carries "
                f"the rod and this two-body idealization does not apply. Every "
                f"member is a two-force member -- use solve_link_forces().")
        loaded_link_names = [l.name for l in arm]
    else:
        # An explicit override says which arm to treat as loaded, so report
        # that rather than the declaration it is standing in for.
        loaded_arm = ("lower_wishbone" if all("Low" in n for n in loaded_link_names)
                      else "upper_wishbone")
    loaded = [l for l in corner.links if l.name in loaded_link_names]
    free = [l for l in corner.links if l.name not in loaded_link_names and not l.steered]
    tie = [l for l in corner.links if l.steered]

    if len(loaded) != 2 or len(free) != 2 or len(tie) != 1:
        raise ValueError(
            f"{corner.name}: expected 2 loaded-arm links, 2 free links and 1 tie "
            f"rod, got {len(loaded)}/{len(free)}/{len(tie)}. This routine assumes "
            f"the loaded wishbone shares one outboard ball joint; a five-link "
            f"with split lower arms is fine as the FREE pair, but the LOADED arm "
            f"must be a wishbone."
        )

    ball = pose.apply(loaded[0].outboard)
    if not np.allclose(ball, pose.apply(loaded[1].outboard), atol=1e-9):
        raise ValueError(
            f"{corner.name}: the two loaded links do not share an outboard "
            f"point, so they are not a wishbone and cannot carry a rod load "
            f"this way."
        )

    # ---- body 1: upright ----
    A = np.zeros((6, 6))
    A[:3, 0:3] = np.eye(3)
    rb = ball - ref
    A[3:, 0:3] = np.array([[0.0, -rb[2], rb[1]],
                           [rb[2], 0.0, -rb[0]],
                           [-rb[1], rb[0], 0.0]])
    for j, link in enumerate(free + tie):
        ob = pose.apply(link.outboard)
        u = ob - link.inboard
        u = u / np.linalg.norm(u)
        A[:3, 3 + j] = u
        A[3:, 3 + j] = np.cross(ob - ref, u)

    if contact_patch is None:
        contact_patch = np.array([ref[0], ref[1], 0.0])

    F_ext = np.array([loads.Fx, loads.Fy, loads.Fz], dtype=float)
    F_ext = F_ext + np.array([0.0, 0.0, -loads.unsprung_mass * loads.gravity])
    F_ext = F_ext - loads.unsprung_mass * np.asarray(loads.unsprung_accel, dtype=float)

    M_ext = np.cross(contact_patch - ref, np.array([loads.Fx, loads.Fy, loads.Fz]))
    M_ext = M_ext + np.array([loads.Mx, loads.My, loads.Mz])

    # Same correction as solve_link_forces -- see the note there.
    spin = _spin_axis_vehicle_frame(_spindle_axis(corner, state))
    if not loads.caliper_on_upright:
        M_ext = M_ext - spin * loads.brake_torque
    if not loads.drive_reacts_on_upright:
        M_ext = M_ext - spin * loads.drive_torque

    rhs = -np.concatenate([F_ext, M_ext])
    x = np.linalg.solve(A, rhs)
    F_ball = x[:3]

    # ---- body 2: the loaded wishbone, moments about its inboard axis ----
    p1, p2 = loaded[0].inboard, loaded[1].inboard
    a = p2 - p1
    a = a / np.linalg.norm(a)

    # The pickup rides the WISHBONE, so its position -- and therefore the rod's
    # line of action and lever arm about the arm's inboard axis -- follow the
    # arm's single rotational DOF, not the upright's six.
    rod_out = rod_outboard_point(corner, pose)
    u_rod = rod_out - corner.actuator.inboard
    u_rod = u_rod / np.linalg.norm(u_rod)

    lever = float(np.dot(np.cross(rod_out - p1, u_rod), a))
    if abs(lever) < 1e-9:
        raise ValueError(
            f"{corner.name}: the rod line passes through the wishbone's inboard "
            f"axis, so it exerts no moment about it and its force is "
            f"indeterminate. Check the rod hardpoints."
        )
    moment_from_ball = float(np.dot(np.cross(ball - p1, -F_ball), a))
    f_rod = -moment_from_ball / lever

    # Same compression-positive solve as `solve_link_forces`, so the same
    # single negation applies -- see the long note there.
    #
    # `F_ball` is NOT negated: it is a genuine force VECTOR on the upright, not
    # a signed axial force, and `A[:3, 0:3] = I` already gives it in the right
    # sense (the moment balance on the wishbone below consumes `-F_ball` as the
    # reaction, which is only correct with that sign).
    #
    # `f_rod` IS negated: `u_rod` points from the rod's inboard end to the
    # wishbone, so a rod in COMPRESSION pushes the wishbone along `+u_rod`,
    # making the raw `f_rod` compression-positive like the links. On this car
    # that flips the front pushrod from a reported +816.7 N "tension" to
    # -816.7 N compression -- which is what a pushrod does, and the difference
    # between sizing it as a tie and sizing it as a strut against buckling.
    return WishboneMountedRodForces(
        ball_joint_force=F_ball,
        free_link_forces=-x[3:5],
        free_link_names=[l.name for l in free],
        tie_rod_force=float(-x[5]),
        rod_force=-f_rod,
        condition_number=float(np.linalg.cond(A)),
        residual=float(np.linalg.norm(A @ x - rhs)),
        loaded_arm=loaded_arm,
    )

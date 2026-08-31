"""Per-component 3D forces at named attachment points.

WHY THIS EXISTS ALONGSIDE `forces.py`
--------------------------------------
`forces.solve_link_forces` returns AXIAL SCALARS -- one number per two-force
member. That is the right answer for sizing a rod end or checking a tube for
buckling, and it is what the 6x6 solves for.

It is NOT what a structures team can put into FEA. They need, for each
component in turn, the 3D force VECTOR applied at each of its attachment
points, in a consistent frame, with a sign convention that says which body the
force acts ON. That is what this module produces, and it matches the layout of
the team's `Suspension Component Forces NFR26` spreadsheet:

    Uprights    OB, IB, BC, UBJ, RL, FL, TR
    Wishbones   OBJ, PR, RBJ, FBJ  (upper, a fixed body carrying bending)
                RL, FL             (lower, still two-force members)
    Rods        OP, IP  (outboard / inboard point)
    Tie rods    OP
    Rockers     RD, SP, PV  (rod, spring/damper, pivot)
    Hubs        CP, OB, IB, BR  (contact patch, bearing rows, brake rotor)

SIGN CONVENTION -- STATED ONCE, HERE
-------------------------------------
Every vector returned is the force **applied TO the named component** by
whatever attaches at that point. So the upright's "UBJ" entry is the force the
upper ball joint exerts ON THE UPRIGHT, and the upper wishbone's "OBJ" entry is
its Newton's-third-law pair -- equal and opposite. `check_action_reaction`
asserts exactly that, because a whole-vehicle force set whose pairs do not
cancel is worse than no force set at all.

Frame is ISO 8855 (x forward, y left, z up), SI newtons. The spreadsheet is in
lbf with "x = forward, y = axial, z = up", which is the same frame; use
`to_lbf` at the reporting boundary only.

`table()` and `report()` take a `frame=` argument that rewrites the components
in another triad -- "solidworks" being x left, y up, z forward -- for pasting
into CAD. That is a relabelling applied at the same boundary as `to_lbf`, never
mid-calculation: everything stored in `components` is ISO, always. See
`core.frames.ReportFrame`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from ..core.frames import get_report_frame, to_report_frame
from ..core.units import LBF_TO_N
from .forces import (CornerLoads, WishboneMountedRodForces, _spin_axis_vehicle_frame,
                     _spindle_axis, solve_link_forces, solve_wishbone_mounted_rod)
from .geometry import CornerGeometry
from .kinematics import CornerState, rod_outboard_point, upper_wishbone_links
from .placeholders import (Provenance, resolve_bearing_offsets,
                           resolve_caliper_mounts, resolve_thrust_row)
from .rocker import rocker_state

__all__ = ["ComponentForces", "extract_components", "to_lbf",
           "check_action_reaction", "check_equilibrium", "caliper_force",
           "hub_bearing_reactions"]


def to_lbf(v):
    """Newtons -> pounds-force, for reporting only."""
    return np.asarray(v, dtype=float) / LBF_TO_N


@dataclass
class ComponentForces:
    """Force vectors on each component, keyed by attachment-point label.

    `components["upright"]["UBJ"]` is a 3-vector in newtons, ISO 8855, being
    the force applied TO the upright at the upper ball joint.
    """

    components: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    case_name: str = ""
    corner: str = ""

    points: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    """Application point of each force, same keys as `components`. Needed to
    check MOMENT equilibrium -- a force set can have every action/reaction pair
    cancel and still be wrong if the forces are at the wrong places."""

    hinge_axes: Dict[str, tuple] = field(default_factory=dict)
    """{body: (axis, mode)} -- which moment components are NOT checked, because
    nothing in the model has to balance them.

    The two modes are genuinely opposite, and conflating them would either hide
    a real error or flag a non-error:

      "along"  exempt the component ALONG `axis`. The upper wishbone's two
               chassis pickups lie on a line and cannot react moment about it,
               so how the pair splits that component is a modelling choice.

      "perp"   exempt everything PERPENDICULAR to `axis`. The rocker pivot is a
               bearing: it freely reacts moment across its axis but not about
               it, so only the about-axis balance -- spring against rod -- is
               a physical statement.
    """

    couples: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    """PURE moments on a body, N.m -- drive torque from a hub motor, or brake
    torque arriving through a halfshaft. These have no line of action, so they
    cannot be folded into `components` as a force at a point, and a moment
    balance that ignores them will not close."""

    provenance: Provenance = field(default_factory=Provenance)
    """Which inputs were placeholders. Carried alongside the numbers rather
    than printed once at import, so it survives into the export."""

    def net(self, component: str, point: str) -> float:
        return float(np.linalg.norm(self.components[component][point]))

    def table(self, component: str, lbf: bool = True,
              frame=None) -> Dict[str, Dict[str, float]]:
        """{point: {Fx, Fy, Fz, Net}} in lbf (spreadsheet units) or N.

        `frame` names the axis triad the components are WRITTEN in -- `None` or
        `"iso8855"` for the internal frame, `"solidworks"` for the CAD one. It
        is a relabelling of the same vector: `Net` is a magnitude and does not
        move, which is what makes it safe for `ForceMatrix.envelope` to pick a
        winner in one frame and report it in another.
        """
        conv = to_lbf if lbf else (lambda v: np.asarray(v, dtype=float))
        frame = get_report_frame(frame)
        out = {}
        for point, vec in self.components[component].items():
            v = conv(vec)
            net = float(np.linalg.norm(v))
            v = to_report_frame(v, frame)
            out[point] = {"Fx": float(v[0]), "Fy": float(v[1]),
                          "Fz": float(v[2]), "Net": net}
        return out

    def report(self, lbf: bool = True, frame=None) -> str:
        unit = "lbf" if lbf else "N"
        frame = get_report_frame(frame)
        lines = [f"{self.corner} / {self.case_name}   forces in {unit}, "
                 f"{frame.header}"]
        for comp in self.components:
            lines.append(f"  [{comp}]")
            lines.append(f"    {'point':<6}{'Fx':>10}{'Fy':>10}{'Fz':>10}{'Net':>10}")
            for point, v in self.table(comp, lbf=lbf, frame=frame).items():
                lines.append(f"    {point:<6}{v['Fx']:10.1f}{v['Fy']:10.1f}"
                             f"{v['Fz']:10.1f}{v['Net']:10.1f}")
        return "\n".join(lines)


def _axial_vector(inboard, outboard, f_tension: float) -> np.ndarray:
    """Force a two-force member applies to the body at its OUTBOARD end.

    `f_tension` is tension-positive (the convention `LinkForces` reports). A
    member in tension PULLS the outboard body toward the inboard end, i.e.
    along -u where u points inboard -> outboard.
    """
    u = np.asarray(outboard, dtype=float) - np.asarray(inboard, dtype=float)
    u = u / np.linalg.norm(u)
    return -f_tension * u


def _bending_arm(p1: np.ndarray, p2: np.ndarray, ball: np.ndarray,
                 obj: np.ndarray, rod_out: np.ndarray, pr: np.ndarray
                 ) -> Dict[str, np.ndarray]:
    """OBJ / PR / FBJ / RBJ for the arm the rod picks up on.

    An arm carrying a rod is NOT a two-force member, so its two chassis pickups
    do not take axial loads along their own legs -- they take whatever is left
    after the ball joint and the rod, and the split between them is fixed by
    moment balance about the line joining them, less the component about that
    line, which they cannot react at all.

    Arm-agnostic: the caller passes the arm's own inboard points and ball
    joint, so this serves an upper- or a lower-mounted rod unchanged.
    """
    body = {"OBJ": obj, "PR": pr}
    resid_F = -(obj + pr)
    resid_M = -(np.cross(ball - p1, obj) + np.cross(rod_out - p1, pr))
    d = p2 - p1
    L2 = float(d @ d)
    # F2 satisfies d x F2 = resid_M (component perpendicular to d), plus an
    # even share of the axial part, which the pair cannot distinguish.
    #
    # THE SIGN. `d x (-(d x M)/L2)` expands to `M - d(d.M)/L2`, which is the
    # perpendicular part of M; the positive form gives exactly minus that. The
    # positive form was used here originally, which flipped BOTH pickup forces
    # and left the arm with an unbalanced moment of 2*|resid_M| -- measured at
    # 493 N.m on the front arm under braking, against a real pickup load of
    # roughly 1 kN. Both pickups were reported with the wrong sign, so the pair
    # still summed correctly and `check_action_reaction` could not see it.
    # Pinned by test_upper_wishbone_pickups_balance_moment.
    F2 = -np.cross(d, resid_M) / L2 + 0.5 * (float(resid_F @ d) / L2) * d
    F1 = resid_F - F2
    # Name them by position: the more REARWARD pickup is the rear ball joint.
    if p1[0] >= p2[0]:
        body["FBJ"], body["RBJ"] = F1, F2
    else:
        body["RBJ"], body["FBJ"] = F1, F2
    return body


def _two_force_arm(links, axials, pose, ball: np.ndarray) -> Dict[str, np.ndarray]:
    """OBJ / FBJ / RBJ for an arm that carries NO rod.

    With nothing hanging off it, each leg is a genuine two-force member, so the
    chassis pickups take exactly the leg's own axial load -- no moment-balance
    split and no modelling choice, and the body then closes in moment as well as
    force with NO hinge exemption. (Three forces: the ball joint takes
    `-(F1+F2)` and each `F_i` lies along its own leg through the ball joint, so
    every moment term about a pickup vanishes identically.)

    Its two legs share one outboard point -- that is what makes it an arm -- so
    the upright sees a single 3D force there, the same shape of entry a bending
    arm gives it. That is why `UBJ` means the same thing in every pickup mode.
    """
    body: Dict[str, np.ndarray] = {}
    # -f*u for a leg in tension f. This one vector is BOTH the force the leg
    # puts on the upright at the ball joint and the force the chassis puts back
    # on the arm at that leg's pickup -- a two-force member pulls its two ends
    # toward each other, and its own reactions are the same magnitude at both.
    axial_vecs = [_axial_vector(l.inboard, pose.apply(l.outboard), f)
                  for l, f in zip(links, axials)]
    body["OBJ"] = -(axial_vecs[0] + axial_vecs[1])
    named = sorted(zip(links, axial_vecs), key=lambda lf: float(lf[0].inboard[0]))
    (rear_link, f_rear), (front_link, f_front) = named
    body["FBJ"], body["RBJ"] = f_front, f_rear
    return body


def _lower_arm_points(lower, lower_pairs, pose, ball_lower, rod_out
                      ) -> Dict[str, np.ndarray]:
    """Application points for the lower arm, in whichever shape it is reported.

    Bending (rod picks up on it): one shared ball joint plus two chassis
    pickups, like the upper arm. Otherwise: two independent two-force links,
    each with an outboard and an inboard end -- which is the only shape that
    also serves the rear five-link, whose lower links share nothing.
    """
    if ball_lower is None:
        return {
            **{label: pose.apply(link.outboard) for label, link, _ in lower_pairs},
            **{label + "_IB": np.asarray(link.inboard, float)
               for label, link, _ in lower_pairs},
        }
    q1 = np.asarray(lower[0].inboard, dtype=float)
    q2 = np.asarray(lower[1].inboard, dtype=float)
    return {"OBJ": pose.apply(lower[0].outboard), "PR": rod_out,
            "FBJ": q1 if q1[0] >= q2[0] else q2,
            "RBJ": q2 if q1[0] >= q2[0] else q1}


def caliper_force(spin_axis: np.ndarray, centroid: np.ndarray,
                  wheel_centre: np.ndarray, torque: float,
                  rotor_effective_radius: float) -> np.ndarray:
    """Force the CALIPER applies to the UPRIGHT, at the pad centroid.

    THE FREE BODY, derived rather than asserted, because the sign is easy to
    invert and an inverted brake reaction adds load where it should relieve it:

      * Pads apply friction `F_r` to the ROTOR at radius `r_eff`, and that must
        supply the wheel's braking couple: `(p - wc) x F_r = -T * n`, where `T`
        is `wheel_reaction_torque` (positive under braking) and `n` the
        vehicle-frame spin axis.
      * Reaction on the CALIPER from the pads is `-F_r`.
      * The caliper is a free body in equilibrium, so its mounts pull `+F_r`
        from the upright -- and by Newton's third law the caliper pushes
        `-F_r` onto the upright.

    With `p - wc = r_eff * r_hat` and `F_r = f * (n x r_hat)`, the identity
    `r_hat x (n x r_hat) = n` gives `f = -T / r_eff` directly, so the returned
    force is `+(T / r_eff) * (n x r_hat)`.

    `rotor_effective_radius` MUST be the centroid's own radial distance from the
    wheel centre -- `placeholders.resolve_caliper_mounts` returns it as
    `r_eff` for exactly this reason. Pass a config value that disagrees with the
    geometry and the reproduced couple comes out as `T * r_geom / r_config`,
    i.e. the wrong braking torque, and the corner stops closing.

    WHAT THE MOUNT BOLTS DO AND DO NOT DECIDE
    ------------------------------------------
    Applied at `centroid` this reproduces the net force AND the couple exactly,
    which is why equilibrium never depends on where the mount BOLTS are -- only
    the upright's internal stress distribution does.

    The centroid's AXIAL position is a different matter, and it does change the
    answer. Only its radial part is used to build the force above (the pad force
    is tangential wherever the rotor sits along the axis), but the caller takes
    the moment about the wheel centre at the full 3D point, and with
    `p - wc = r_eff*r_hat + dy*n`:

        (r_eff*r_hat + dy*n) x F  =  T*n  -  (T / r_eff) * dy * r_hat

    The brake couple `T*n` is there either way; the second term is a RADIAL
    moment that appears only when the rotor centreplane is offset from the
    wheel-centre plane. It is real -- ~23 N.m at 913 N of pad force and 25 mm of
    offset -- and the bearing rows split it into a few hundred newtons apiece.
    So a centroid flattened into the wheel-centre plane understates OB/IB;
    see `placeholders.PAD_CENTRE_POINT_NAMES`.
    """
    n = np.asarray(spin_axis, dtype=float)
    n = n / np.linalg.norm(n)
    r = np.asarray(centroid, dtype=float) - np.asarray(wheel_centre, dtype=float)
    r_perp = r - float(r @ n) * n
    mag = np.linalg.norm(r_perp)
    if mag < 1e-9:
        raise ValueError("caliper centroid lies on the spindle axis")
    r_hat = r_perp / mag
    t_hat = np.cross(n, r_hat)
    return (torque / rotor_effective_radius) * t_hat


def hub_bearing_reactions(axis: np.ndarray, inner_offset: float,
                          outer_offset: float, ext_force: np.ndarray,
                          ext_moment: np.ndarray, thrust_row: str = "outer"
                          ) -> Dict[str, np.ndarray]:
    """Force the HUB applies to the UPRIGHT through each bearing row.

    WHY NOT JUST USE `forces.bearing_loads`
    ---------------------------------------
    That function answers a narrower question -- the radial load on each row
    from the contact patch alone -- and deliberately drops both the axial force
    and any other force crossing the cut, because it exists to feed bearing
    life calculations. Building the hub's free body from it leaves the caliper
    reaction and the thrust load unaccounted for, and the body does not close.

    This solves the whole thing: `ext_force` and `ext_moment` (about the wheel
    centre) are everything acting on the hub EXCEPT the bearings, so

        A_in + A_out                       =  ext_force
        d_in x A_in  +  d_out x A_out      =  ext_moment

    with `d = offset * axis`. Two point forces on a common axis cannot react
    moment ABOUT that axis, so the axial component of the second equation is
    not an equation at all -- it is the statement that brake or drive torque
    balances the contact patch, and it is checked, not solved.

    That leaves five equations for six unknowns. The missing one is the axial
    thrust split, which is a property of the HARDWARE (which row is located),
    not of the loading -- hence `thrust_row`.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    span = outer_offset - inner_offset
    if abs(span) < 1e-6:
        raise ValueError("bearing rows are coincident; spacing must be non-zero")

    F = np.asarray(ext_force, dtype=float)
    M = np.asarray(ext_moment, dtype=float)
    F_axial = float(F @ n)
    F_rad = F - F_axial * n
    M_rad = M - float(M @ n) * n

    A_out = -np.cross(n, M_rad) / span - (inner_offset / span) * F_rad
    A_in = F_rad - A_out

    if thrust_row == "inner":
        A_in = A_in + F_axial * n
    elif thrust_row == "split":
        A_in = A_in + 0.5 * F_axial * n
        A_out = A_out + 0.5 * F_axial * n
    else:
        A_out = A_out + F_axial * n

    return {"inner": A_in, "outer": A_out,
            "unbalanced_axial_moment": float(M @ n)}


def extract_components(corner: CornerGeometry, state: CornerState,
                       loads: CornerLoads,
                       inner_bearing_offset: float = -0.035,
                       outer_bearing_offset: float = 0.035,
                       case_name: str = "", solved:
                       Optional[WishboneMountedRodForces] = None,
                       rotor_effective_radius: float = 0.077216,
                       axially_located: str = "UNCONFIRMED",
                       hardpoints: Optional[Dict[str, np.ndarray]] = None,
                       spring_force: Optional[float] = None,
                       ) -> ComponentForces:
    """All component force vectors for one corner in one load case.

    THE BREAKDOWN DEPENDS ON `corner.actuator.rod_pickup`, because which
    members are two-force members does:

      `upper_wishbone`  (NFR26, both axles) upper arm bends and hands the
            upright a 3D `UBJ`; lower links stay two-force and are reported per
            link as `RL`/`FL` with their `_IB` chassis ends.
      `lower_wishbone`  the roles swap. The lower arm bends and hands the
            upright a 3D `LBJ`; the upper pair becomes two clean two-force
            links, still summed into one `UBJ` because they share a ball joint.
      `upright`  nothing hangs off an arm, so the plain 6x6 applies, no arm is
            a bending member, and the rod appears in the UPRIGHT's own free
            body as `PR`.

    Args:
        hardpoints: raw point dict, consulted for the brake pad friction
            centroid (`BPad_Center`), the caliper mount bolts
            (`BC_Upper`/`BC_Lower`) and the bearing seats
            (`IB_Bearing`/`OB_Bearing`). `BPad_Center` contributes all three of
            its coordinates -- radius, clock and AXIAL offset -- read in the
            design frame and re-applied at this case's pose, and its radius,
            not `rotor_effective_radius`, then sets the pad force, so the couple
            it reproduces is the commanded one. With only the bolts, the
            centroid goes on `rotor_effective_radius` in their direction but in
            the wheel-centre plane, which drops the radial moment the rotor's
            axial offset carries. Omit the lot and the mounts are synthesized
            and the seats fall back to the `inner_/outer_bearing_offset`
            arguments -- see `placeholders`.
        inner_bearing_offset: fallback inboard seat position along the spindle
            axis, used only when `hardpoints` carries no bearing points.
        outer_bearing_offset: the same, outboard.
        spring_force: overrides the rocker's spring load. Normally left None,
            which derives it from rocker moment balance; supply it only to
            check against a measured spring installation.
    """
    prov = Provenance()
    pose = state.pose

    # ---- spindle torque, resolved BEFORE anything is solved ----
    #
    # THE TORQUE THE WHEEL ACTUALLY NEEDS, not the nominal -Fx * r_loaded.
    #
    # `wheel_reaction_torque` assumes a lateral spin axis and a moment arm of
    # exactly the loaded radius. With camber and toe the axis is tilted, so
    # LATERAL force contributes to the spindle moment too -- 7.9% of nominal on
    # this car's front corner under cornering.
    #
    # This has to be settled first, because the link solver removes an inboard
    # caliper's couple from the upright and the hub applies that same couple.
    # If the two use different numbers the upright is left with the difference:
    # 3.4 N.m, measured, with an inboard caliper. Same torque, one source.
    wc = np.asarray(state.wheel_centre, dtype=float)
    axis_out = _spindle_axis(corner, state)
    spin = _spin_axis_vehicle_frame(axis_out)
    cp = (np.asarray(corner.contact_patch, dtype=float)
          if corner.contact_patch is not None
          else np.array([wc[0], wc[1], 0.0]))

    F_cp = np.array([loads.Fx, loads.Fy, loads.Fz], dtype=float)
    M_cp = np.cross(cp - wc, F_cp) + np.array([loads.Mx, loads.My, loads.Mz])
    T_required = float(M_cp @ spin)

    braking = abs(loads.brake_torque) >= abs(loads.drive_torque)
    T_nominal = loads.brake_torque + loads.drive_torque
    if abs(T_nominal) > 1e-9:
        gap = abs(T_required - T_nominal) / abs(T_nominal)
        if gap > 0.02:
            prov.add("spindle_torque", f"{T_required:.1f} N.m vs nominal "
                                       f"{T_nominal:.1f} N.m ({gap:.1%})",
                     "Torque taken from the contact-patch moment about the "
                     "ACTUAL spin axis, which includes camber/toe and the "
                     "lateral-force term that -Fx*r_loaded omits.")
    effective = replace(loads,
                        brake_torque=T_required if braking else 0.0,
                        drive_torque=0.0 if braking else T_required)

    upper = upper_wishbone_links(corner)
    if upper is None:
        raise ValueError(
            f"{corner.name}: no identifiable upper wishbone, so the "
            f"component breakdown below does not apply.")

    lower = [l for l in corner.links if l.name not in
             [u.name for u in upper] and not l.steered]
    tie = [l for l in corner.links if l.steered][0]

    # ---- THE DECLARED PICKUP CHOOSES THE FREE BODY ----
    #
    # This is a structural-topology question, not a parameter (docs/11 SS5):
    # which members are two-force members changes with it, so the three cases
    # are three different solves, not one solve with a flag.
    #
    #   upper_wishbone  upper arm bends, lower links stay two-force  (NFR26)
    #   lower_wishbone  the same picture with the roles SWAPPED
    #   upright         nothing hangs off an arm, so the plain 6x6 applies and
    #                   NO arm is a bending member
    pickup = corner.actuator.rod_pickup
    ball_upper = ball_lower = None
    upper_axial = lower_axial = None

    if pickup == "upright":
        if solved is not None:
            raise ValueError(
                f"{corner.name}: `solved` carries a WishboneMountedRodForces, "
                f"but rod_pickup='upright' -- that is a different free body. "
                f"Drop the argument and let this solve the 6x6.")
        lf = solve_link_forces(corner, state, effective)
        axial = lf.as_dict()
        f_rod = axial[corner.actuator.kind]
        f_tie = axial[tie.name]
        upper_axial = [axial[l.name] for l in upper]
        lower_axial = [axial[l.name] for l in lower]
    else:
        r = solved if solved is not None else solve_wishbone_mounted_rod(
            corner, state, effective)
        f_rod, f_tie = r.rod_force, r.tie_rod_force
        if pickup == "upper_wishbone":
            ball_upper = np.asarray(r.ball_joint_force, float)
            lower_axial = list(r.free_link_forces)
        else:
            ball_lower = np.asarray(r.ball_joint_force, float)
            upper_axial = list(r.free_link_forces)

    ball = pose.apply(upper[0].outboard)
    rod_out = rod_outboard_point(corner, pose)
    rod_in = np.asarray(corner.actuator.inboard, dtype=float)

    # ---- upright ----
    # The wishbone reacts on the upright with -F_ball (F_ball as solved is the
    # force the WISHBONE exerts on the UPRIGHT, so it is already the right
    # sense for the upright's own free body).
    # Lower links labelled by longitudinal position -- see the lower-arm block.
    lower_pairs = ()
    if lower_axial is not None:
        lower_named = sorted(zip(lower, lower_axial),
                             key=lambda lf_: float(lf_[0].inboard[0]))
        (rear_link, f_rear), (front_link, f_front) = lower_named
        lower_pairs = (("RL", rear_link, f_rear), ("FL", front_link, f_front))

    upright: Dict[str, np.ndarray] = {}
    if ball_upper is not None:
        upright["UBJ"] = ball_upper
    else:
        # A two-force upper arm still meets the upright at ONE point, so the
        # upright sees one 3D force there -- see `_two_force_arm`.
        upright["UBJ"] = -_two_force_arm(upper, upper_axial, pose, ball)["OBJ"]
    if ball_lower is not None:
        # LBJ, not RL/FL: a bending lower arm hands the upright a single
        # general 3D force at its shared ball joint, and reporting it as two
        # axial numbers would say "strut" about a member that is bending.
        upright["LBJ"] = ball_lower
    for label, link, f in lower_pairs:
        upright[label] = _axial_vector(link.inboard, pose.apply(link.outboard), f)
    upright["TR"] = _axial_vector(tie.inboard, pose.apply(tie.outboard), f_tie)
    if pickup == "upright":
        # The rod loads the upright directly -- it is in ITS free body, and in
        # no arm's. Absent here, the upright would not close.
        upright["PR"] = _axial_vector(rod_in, rod_out, f_rod)

    # ---- hub: contact patch in, bearings and brake out ----
    #
    # Solved BEFORE the upright's OB/IB entries because those are just this
    # body's reactions. Deriving both from one free body is what makes the two
    # agree; the previous version took them from `bearing_loads`, which omits
    # axial load and the caliper reaction, so the upright could not close.
    #
    # TWO FRAMES GO IN, ONE COMES OUT. `wc`/`spin` are POSED and the returned
    # centroid is in that frame; the STATIC pair is what the sheet's own
    # `BPad_Center` is decomposed against, because its coordinates are
    # design-position geometry like every other hardpoint. Handing the posed
    # wheel centre to a design-position point would subtract this case's bump
    # and steer travel out of the rotor's axial offset -- invisible while only
    # a unit direction survived that subtraction, first-order now that the
    # axial coordinate is kept. Same reasoning, and the same static arguments,
    # as `resolve_bearing_offsets` below.
    #
    # `spin`, note, is the +y-oriented axis, NOT the outboard `axis_out` the
    # bearing call takes: the pad offset's sign is stated against ISO left.
    static_wc = corner.wheel_centre if corner.wheel_centre is not None else wc
    static_spin = (_spin_axis_vehicle_frame(
        np.asarray(corner.spindle_axis, dtype=float)
        / float(np.linalg.norm(corner.spindle_axis)))
        if corner.spindle_axis is not None else spin)
    mounts = resolve_caliper_mounts(hardpoints, wc, spin,
                                    rotor_effective_radius, prov=prov,
                                    static_wheel_centre=static_wc,
                                    static_spin_axis=static_spin)
    thrust_row = resolve_thrust_row(axially_located, prov=prov)

    # Seat positions come from the sheet when it carries them, and the config
    # estimate only when it does not. Projected in the STATIC frame -- the
    # sheet's points are design-position geometry, while `wc`/`axis_out` here
    # have been moved by the pose. The offsets themselves are body-fixed, so
    # once measured they are the same at any pose.
    inner_bearing_offset, outer_bearing_offset = resolve_bearing_offsets(
        hardpoints,
        corner.wheel_centre if corner.wheel_centre is not None else wc,
        corner.spindle_axis if corner.spindle_axis is not None else axis_out,
        inner_bearing_offset, outer_bearing_offset, prov=prov)

    hub: Dict[str, np.ndarray] = {"CP": F_cp}
    hub_points: Dict[str, np.ndarray] = {"CP": cp}

    # The rotor is part of this body only when the caliper is outboard. With
    # inboard brakes the rotor sits on the chassis side of the cut and the
    # couple arrives through the halfshaft as a pure moment instead.
    ext_F = F_cp.copy()
    ext_M = M_cp.copy()
    couples: Dict[str, Dict[str, np.ndarray]] = {"hub": {}, "upright": {}}

    # THE TIRE'S OWN MOMENTS ARE A PURE COUPLE AND MUST BE DECLARED AS ONE.
    #
    # Mz (aligning), Mx (overturning) and My (rolling resistance) have no line
    # of action, so they cannot ride along with `hub["CP"]` as a force at a
    # point. They are already inside `M_cp`, hence inside `ext_M`, so the
    # bearing rows do react them -- but `check_equilibrium` rebuilds each body's
    # moment from its forces and their application points, and would see this
    # one as missing.
    #
    # Left undeclared, the hub came up with an unbalanced moment of exactly
    # |Mz| -- 54.96 N.m on the front outside-cornering corner, i.e. the whole
    # aligning moment. That is what appeared the moment `load_cases` started
    # populating Mz, and it is the correct failure: the body genuinely was not
    # closing. Pinned by test_hub_closes_with_tire_aligning_moment.
    M_tire = np.array([loads.Mx, loads.My, loads.Mz], dtype=float)
    if np.any(M_tire):
        couples["hub"]["CP"] = M_tire

    if braking and loads.caliper_on_upright:
        # Outboard caliper: the pads are a real force pair at a real radius, so
        # the rotor sees a FORCE, not an abstract couple. Applying it at the pad
        # centroid reproduces the couple exactly, so equilibrium never depends
        # on where the mount bolts sit.
        #
        # THE DIVISOR IS `mounts["r_eff"]`, NOT THE CONFIG RADIUS. They agree
        # whenever the centroid was constructed at the config value, and differ
        # when the sheet supplies a real `BPad_Center` at its own radius -- in
        # which case the config value would reproduce a couple of
        # `T * r_sheet / r_config` and the hub would not close. The geometry the
        # moment arm is taken at has to be the geometry the force is sized from.
        F_pad_on_rotor = -caliper_force(spin, mounts["centroid"], wc,
                                        T_required, mounts["r_eff"])
        hub["BR"] = F_pad_on_rotor
        hub_points["BR"] = mounts["centroid"]
        ext_F = ext_F + F_pad_on_rotor
        ext_M = ext_M + np.cross(mounts["centroid"] - wc, F_pad_on_rotor)
        upright["BC"] = -F_pad_on_rotor
    else:
        # Everything else reaches the hub as a PURE couple: an inboard caliper
        # or a halfshaft has no line of action at this cut. Where the other end
        # of that couple lands -- upright or chassis -- is what the two
        # `*_on_upright` flags decide, and it is the whole reason a hub motor
        # loads the links differently from an inboard diff.
        hub["BR"] = np.zeros(3)
        hub_points["BR"] = wc
        couples["hub"]["BR"] = -spin * T_required
        ext_M = ext_M - spin * T_required
        upright["BC"] = np.zeros(3)
        on_upright = (loads.drive_reacts_on_upright if not braking
                      else loads.caliper_on_upright)
        if on_upright:
            couples["upright"]["BC"] = spin * T_required

    br = hub_bearing_reactions(axis_out, inner_bearing_offset,
                               outer_bearing_offset, ext_F, ext_M, thrust_row)
    hub["OB"] = -br["outer"]
    hub["IB"] = -br["inner"]
    hub_points["OB"] = wc + outer_bearing_offset * axis_out
    hub_points["IB"] = wc + inner_bearing_offset * axis_out

    upright["OB"] = br["outer"]
    upright["IB"] = br["inner"]

    # ---- unsprung weight and d'Alembert inertia ----
    #
    # The link solver already applies this to the upright's free body as a pure
    # force at the wheel centre, so the component breakdown has to carry the
    # same term at the same point or the body does not close. Omitting it left
    # a constant 129 N on the front upright -- exactly one unsprung weight --
    # and 1242 N in the bump case, where the inertia term dominates.
    #
    # The wheel centre is where the SOLVER puts it, not where the assembly's
    # centre of mass actually is; using the real c.g. would disagree with the
    # link forces this breakdown is supposed to explain. That makes it a
    # modelling choice worth naming, so it is flagged.
    if abs(loads.unsprung_mass) > 0.0:
        upright["UM"] = (np.array([0.0, 0.0, -loads.unsprung_mass * loads.gravity])
                         - loads.unsprung_mass
                         * np.asarray(loads.unsprung_accel, dtype=float))
        prov.add("unsprung_cg", "at the wheel centre",
                 "Unsprung weight and inertia are applied at the wheel centre "
                 "because that is where the link solver applies them. The real "
                 "assembly c.g. is offset from it, which redistributes load "
                 "between the upper and lower arms without changing the total.")

    # ---- the two arms ----
    #
    # Exactly one of them can be the bending member, and which one it is comes
    # from the declared pickup. The other is a clean two-force pair. With the
    # rod on the upright, NEITHER bends.
    p1 = np.asarray(upper[0].inboard, dtype=float)
    p2 = np.asarray(upper[1].inboard, dtype=float)
    pr_vec = _axial_vector(rod_in, rod_out, f_rod)

    if ball_upper is not None:
        # OBJ is the reaction of the upright on the arm = -(arm on upright).
        wishbone_upper = _bending_arm(p1, p2, ball, -ball_upper, rod_out, pr_vec)
    else:
        wishbone_upper = _two_force_arm(upper, upper_axial, pose, ball)

    if ball_lower is not None:
        # ---- lower arm carries the rod: same body, roles swapped ----
        lq1 = np.asarray(lower[0].inboard, dtype=float)
        lq2 = np.asarray(lower[1].inboard, dtype=float)
        lower_ball = pose.apply(lower[0].outboard)
        wishbone_lower = _bending_arm(lq1, lq2, lower_ball, -ball_lower,
                                      rod_out, pr_vec)
    else:
        # ---- lower arm: two-force members, at BOTH ends ----
        #
        # Reporting only the outboard ends leaves the arm with a large net force
        # (2.5 kN on the front under braking) because the chassis pickups were
        # simply absent from the body. They are also what the frame team needs.
        #
        # Reported per LINK rather than as one OBJ/FBJ/RBJ set, because the rear
        # five-link's lower links do not share an outboard point and are not one
        # arm at all.
        #
        # Labels come from the pickups' LONGITUDINAL POSITION, not from the order
        # the links happen to sit in the topology list. `DOUBLE_WISHBONE_PUSHROD`
        # lists LowLead before LowTrail, and "lead" is the FORWARD link, so zipping
        # against ("RL", "FL") named the front link "rear" and vice versa on both
        # axles. Pinned by test_lower_arm_labels_follow_geometry.
        wishbone_lower = {}
        for label, link, f in lower_pairs:
            ob = pose.apply(link.outboard)
            wishbone_lower[label] = -_axial_vector(link.inboard, ob, f)
            wishbone_lower[label + "_IB"] = _axial_vector(link.inboard, ob, f)

    # ---- rods: two-force members, so both ends follow from one axial force ----
    rods = {"OP": -pr_vec, "IP": pr_vec}
    tie_rod = {"OP": -_axial_vector(tie.inboard, pose.apply(tie.outboard), f_tie),
               "IP": _axial_vector(tie.inboard, pose.apply(tie.outboard), f_tie)}

    # ---- rocker ----
    rocker: Dict[str, np.ndarray] = {}
    rocker_points: Dict[str, np.ndarray] = {}
    rocker_axis = None
    if corner.actuator.has_rocker:
        rocker, rocker_points, rocker_axis = _rocker_forces(
            corner, state, -rods["IP"], rod_in, spring_force, prov)

    # Only a BENDING arm gets the moment exemption. Its two chassis pickups lie
    # on a line and cannot react moment about it, so how they split that
    # component is a modelling choice rather than physics. A two-force arm has
    # no such freedom -- its pickups take their own legs' axial loads and the
    # body closes about every axis -- so exempting it would hide a real error.
    hinge_axes = {}
    if ball_upper is not None:
        hinge_axes["wishbone_upper"] = (p2 - p1, "along")
    if ball_lower is not None:
        hinge_axes["wishbone_lower"] = (
            np.asarray(lower[1].inboard, float) - np.asarray(lower[0].inboard, float),
            "along")
    if rocker_axis is not None:
        hinge_axes["rocker"] = (rocker_axis, "perp")

    components = {
        "upright": upright,
        "wishbone_upper": wishbone_upper,
        "wishbone_lower": wishbone_lower,
        "rod": rods,
        "tie_rod": tie_rod,
        "hub": hub,
    }
    points = {
        # `lower_pairs`, NOT `lower` -- the sorted order. On the front both
        # lower links share one outboard ball joint so the difference is
        # invisible, but the rear five-link has two distinct outboard points
        # and using the raw order attaches each force to the other link's
        # point. That left the rear upright with up to 100 N.m of unbalanced
        # moment while the front closed perfectly.
        "upright": {
            "UBJ": ball, "OB": hub_points["OB"], "IB": hub_points["IB"],
            "BC": mounts["centroid"],
            "TR": pose.apply(tie.outboard),
            "UM": wc,
            **({"LBJ": pose.apply(lower[0].outboard)} if ball_lower is not None else {}),
            **({"PR": rod_out} if pickup == "upright" else {}),
            **{label: pose.apply(link.outboard) for label, link, _ in lower_pairs},
        },
        "wishbone_upper": {
            "OBJ": ball, "FBJ": p1 if p1[0] >= p2[0] else p2,
            "RBJ": p2 if p1[0] >= p2[0] else p1,
            **({"PR": rod_out} if ball_upper is not None else {}),
        },
        "wishbone_lower": _lower_arm_points(lower, lower_pairs, pose, ball_lower,
                                            rod_out),
        "rod": {"OP": rod_out, "IP": rod_in},
        "tie_rod": {"OP": pose.apply(tie.outboard), "IP": np.asarray(tie.inboard, float)},
        "hub": hub_points,
    }
    if rocker:
        components["rocker"] = rocker
        points["rocker"] = rocker_points

    return ComponentForces(
        components=components, points=points, couples=couples,
        hinge_axes=hinge_axes, case_name=case_name, corner=corner.name,
        provenance=prov)


def _rocker_forces(corner: CornerGeometry, state: CornerState,
                   rod_on_rocker: np.ndarray, rod_in: np.ndarray,
                   spring_force: Optional[float], prov: Provenance):
    """Rod, spring/damper and pivot reactions on the bellcrank.

    The rocker has ONE rotational degree of freedom, so moments about its pivot
    axis close on a single scalar -- the spring force -- and the pivot then
    takes everything else, including the moment components the axis cannot
    react. That is the whole solve; there is no iteration and no assumption.

    The pivot reaction is the number worth having: it is typically the largest
    single load into the chassis at that corner, and nothing upstream in this
    repo computed it.
    """
    act = corner.actuator
    rs = rocker_state(corner, state)
    pivot = np.asarray(act.rocker_pivot, dtype=float)
    axis = np.asarray(act.rocker_axis, dtype=float) - pivot
    axis = axis / np.linalg.norm(axis)

    # The rod force was solved against the rod's DESIGN inboard point, so the
    # rocker must use the same one or the two bodies disagree about where the
    # rod pulls. At design ride height they coincide; off design they do not,
    # and silently using the rotated point would break the two-force member.
    attach = np.asarray(rs.rod_attach, dtype=float)
    drift = float(np.linalg.norm(attach - rod_in))
    if drift > 1e-3:
        prov.add("rocker.rod_attach", f"{drift * 1000:.1f} mm from design",
                 "Rocker has rotated away from design position, but the rod "
                 "force was solved against the DESIGN inboard point. Rocker "
                 "lever arms are taken at the design point to keep the rod a "
                 "true two-force member; the resulting spring/pivot loads are "
                 "off by roughly this fraction of the rod arm.")

    spring_attach = np.asarray(rs.spring_attach, dtype=float)
    spring_chassis = np.asarray(act.spring_chassis, dtype=float)
    w = spring_attach - spring_chassis
    w = w / np.linalg.norm(w)

    r_rod = rod_in - pivot
    r_spring = spring_attach - pivot

    if spring_force is None:
        # Moment about the pivot axis: rod moment + spring moment = 0.
        lever = float(np.cross(r_spring, -w) @ axis)
        if abs(lever) < 1e-9:
            raise ValueError(
                f"{corner.name}: the spring line passes through the rocker "
                f"pivot axis, so it exerts no moment and its force is "
                f"indeterminate. Check Rocker_Spring / Spring_Chassis.")
        spring_force = -float(np.cross(r_rod, rod_on_rocker) @ axis) / lever

    F_spring = -spring_force * w
    F_pivot = -(rod_on_rocker + F_spring)

    return ({"RD": rod_on_rocker, "SP": F_spring, "PV": F_pivot},
            {"RD": rod_in, "SP": spring_attach, "PV": pivot},
            axis)


def check_action_reaction(cf: ComponentForces, atol: float = 1e-6) -> Dict[str, float]:
    """Residuals of every action/reaction pair. Empty dict = all consistent.

    A force set whose pairs do not cancel is worse than none at all: it looks
    authoritative and quietly double-counts or loses load. Cheap to check, so
    it is checked.
    """
    c = cf.components
    bad = {}
    pairs = [
        ("UBJ/OBJ", c["upright"]["UBJ"] + c["wishbone_upper"]["OBJ"]),
        ("TR", c["upright"]["TR"] + c["tie_rod"]["OP"]),
    ]
    # Which body the rod loads, and how the lower arm meets the upright, both
    # follow from the declared pickup -- so the pairs are looked up rather than
    # assumed. A hardcoded `wishbone_upper["PR"]` would simply KeyError on a
    # lower- or upright-mounted rod, i.e. the check would stop checking.
    for body in ("wishbone_upper", "wishbone_lower", "upright"):
        if "PR" in c.get(body, {}):
            pairs.append(("rod", c[body]["PR"] + c["rod"]["OP"]))
            break
    if "LBJ" in c["upright"]:
        pairs.append(("LBJ/OBJ", c["upright"]["LBJ"] + c["wishbone_lower"]["OBJ"]))
    else:
        pairs.append(("RL", c["upright"]["RL"] + c["wishbone_lower"]["RL"]))
        pairs.append(("FL", c["upright"]["FL"] + c["wishbone_lower"]["FL"]))
    if "hub" in c:
        pairs.append(("bearing_OB", c["upright"]["OB"] + c["hub"]["OB"]))
        pairs.append(("bearing_IB", c["upright"]["IB"] + c["hub"]["IB"]))
    if "rocker" in c:
        pairs.append(("rod_inboard", c["rocker"]["RD"] + c["rod"]["IP"]))
    for name, resid in pairs:
        n = float(np.linalg.norm(resid))
        if n > atol:
            bad[name] = n
    return bad


def check_equilibrium(cf: ComponentForces, atol: float = 1e-6
                      ) -> Dict[str, Dict[str, float]]:
    """Force AND moment closure on every body. Empty dict = all in equilibrium.

    WHY THIS EXISTS SEPARATELY FROM `check_action_reaction`
    -------------------------------------------------------
    Action/reaction only says neighbouring bodies agree on the force at their
    shared joint. It says nothing about whether a body's forces SUM to zero, and
    nothing at all about where they act -- a set with every pair matched can
    still have a body spinning up under a net moment, which is exactly the
    failure mode when a reaction is put at the wrong point or a pure couple is
    forgotten.

    The upper wishbone is exempt from the MOMENT check about the component it
    cannot react (the line through its two chassis pickups is a hinge, and the
    split there is a modelling choice, not physics), so only the force sum and
    the two reactable moment components are checked.
    """
    bad: Dict[str, Dict[str, float]] = {}
    for body, forces in cf.components.items():
        pts = cf.points.get(body, {})
        if len(pts) < len(forces):
            continue                      # points not recorded; nothing to check
        F = np.zeros(3)
        M = np.zeros(3)
        origin = next(iter(pts.values()))
        for label, vec in forces.items():
            F = F + np.asarray(vec, dtype=float)
            M = M + np.cross(np.asarray(pts[label], float) - origin,
                             np.asarray(vec, float))
        for label, cpl in cf.couples.get(body, {}).items():
            M = M + np.asarray(cpl, dtype=float)

        f_res = float(np.linalg.norm(F))
        hinge = cf.hinge_axes.get(body)
        if hinge is not None:
            axis, mode = hinge
            h = np.asarray(axis, dtype=float)
            h = h / np.linalg.norm(h)
            M = (M - float(M @ h) * h) if mode == "along" else float(M @ h) * h
        m_res = float(np.linalg.norm(M))
        if f_res > atol or m_res > atol:
            bad[body] = {"force": f_res, "moment": m_res}
    return bad

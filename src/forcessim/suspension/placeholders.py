"""Synthesized geometry for points the hardpoint sheets do not carry -- yet.

WHY THIS MODULE IS SEPARATE, AND LOUD
--------------------------------------
The NFR26 `Suspension Points` sheets define the linkage completely, but the
component force breakdown needs a few points they do not contain:

    caliper mount bolts     -- to put the brake reaction somewhere on the upright
    bearing seat positions  -- `IB_Bearing`/`OB_Bearing` if the sheet has them,
                               else the ESTIMATE in config `bearings`
    thrust row              -- which bearing row takes axial load

Rather than bury guesses at the call sites where they would silently become
"the model", every one of them lives here, is tagged with its provenance, and
is reported by `provenance_report()` so a number derived from a guess can never
be mistaken for a number derived from the car.

>>> EVERY VALUE IN THIS MODULE IS A PLACEHOLDER. <<<

Replace them by adding the real points to the hardpoint CSV (the loader picks
up any point name; see `CALIPER_POINT_NAMES`) or by setting the corresponding
config keys. `resolve_caliper_mounts` prefers real geometry whenever it finds
it and only falls back to synthesis, so the swap needs no code change.

WHAT THE PLACEHOLDER CALIPER GEOMETRY ASSUMES
----------------------------------------------
`resolve_caliper_mounts` works down three rungs, and only the bottom one is a
guess in all three coordinates:

  (a) `BPad_Center` in the sheet -- the pad friction centroid MEASURED, all
      three of its coordinates used: radius, clock and axial offset, read in
      the design frame and re-applied at the case's pose. Mount bolts come from
      `BC_Upper`/`BC_Lower` if the sheet has them and are synthesized around the
      centroid if not; synthesized bolts move nothing equilibrium depends on.
  (b) `BC_Upper`/`BC_Lower` only -- the caliper's CLOCK position is real, but
      its axial position is not in the pair's midpoint in any useful sense
      (the bolts sit on the upright, not on the rotor centreplane), so the
      centroid is placed on the rotor's effective radius in the bolts'
      direction, IN THE WHEEL-CENTRE PLANE. That assumption is recorded.
  (c) neither -- a radially-mounted caliper centred at the TRAILING edge of the
      rotor (directly aft of the wheel centre), with two mount bolts straddling
      it tangentially.

The (c) position is the common FSAE package -- it clears the lower wishbone and
the steering arm -- but it is a guess, and it matters in a specific, bounded
way: the caliper reaction on the upright is TANGENTIAL, so moving the caliper
around the clock ROTATES that force without changing its magnitude. Magnitude
is set by brake torque and effective radius, both of which are MEASURED. So
component NET loads are trustworthy; their direction on the upright is not,
until the real mount lands.

The AXIAL coordinate is a separate matter from the clock position, and it is
what (a) buys that (b) and (c) cannot. See `PAD_CENTRE_POINT_NAMES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CALIPER_POINT_NAMES", "PAD_CENTRE_POINT_NAMES", "BEARING_POINT_NAMES",
    "PlaceholderNote", "Provenance", "wheel_plane_basis",
    "synth_caliper_mounts", "resolve_caliper_mounts", "resolve_bearing_offsets",
    "resolve_thrust_row",
]

CALIPER_POINT_NAMES: Sequence[Tuple[str, str]] = (
    ("BC_Upper", "BC_Lower"),
    ("Caliper_Upper", "Caliper_Lower"),
    ("BC_Fwd", "BC_Aft"),
)
"""Point-name pairs the loader will accept for real caliper mounts, in order of
preference. Add either pair to the hardpoint CSV and synthesis switches off."""

PAD_CENTRE_POINT_NAMES: Sequence[str] = ("BPad_Center", "BPad_Centre",
                                         "Pad_Centroid")
"""Point names accepted for the PAD FRICTION CENTROID, in order of preference.

This is the single point at which the two pads' friction resultant acts on the
rotor: on the rotor CENTREPLANE, at the rotor's effective radius, at the
caliper's clock position. It is not the caliper body, not the mount bolts, and
not the rotor centre -- it is where the braking force is applied.

WHY ITS AXIAL COORDINATE IS THE POINT OF CARRYING IT AT ALL
------------------------------------------------------------
The clock position can be recovered from the mount bolts; the axial position
cannot, and it is not zero. The rotor centreplane is offset from the wheel
centre by whatever the hat and the wheel offset make it -- tens of millimetres
on a typical FSAE corner -- and that offset carries a real moment.

Write the pad force as `F = (T / r_eff) * t_hat`, applied at
`p - wc = r_eff * r_hat + dy * n`, with `n` the spin axis and `dy` the rotor
centreplane's offset from the wheel-centre plane. The moment about the wheel
centre is then

    (r_eff*r_hat + dy*n) x F  =  T * n  -  (T / r_eff) * dy * r_hat

The first term is the brake couple, which any centroid on the right radius
reproduces. The second is a RADIAL moment that exists only when `dy != 0`, and
placing the centroid in the wheel-centre plane silently throws it away. It is
not small: at 913 N of pad force and a 25 mm offset it is ~23 N.m, which the
bearing rows split into a few hundred newtons apiece on `OB`/`IB` -- a real
load on real hardware that used to appear nowhere.

So a sheet carrying this point gets ALL THREE of its coordinates used -- as the
body-fixed triple (radius, clock, axial offset), not as raw numbers, since the
sheet is design-position geometry and the case is not; see
`resolve_caliper_mounts`, "THE TWO FRAMES". The mount bolts then become what
they always should have been: reporting detail that tells the upright where the
load is bolted on, not an input to equilibrium."""

BEARING_POINT_NAMES: Sequence[Tuple[str, str]] = (
    ("IB_Bearing", "OB_Bearing"),
    ("Bearing_Inner", "Bearing_Outer"),
)
"""Point-name pairs for the two bearing seats, inboard first, in order of
preference. Put them in the hardpoint CSV -- the same sheet as every other
hardpoint -- and the `bearings.*_offset_m` config estimates stop being used.

The names follow the sheets' own IB_/OB_ prefixes for inboard/outboard, with
`Bearing_Inner`/`Bearing_Outer` accepted for a sheet that spells it out."""


@dataclass(frozen=True)
class PlaceholderNote:
    """One synthesized quantity and why it is not real."""

    key: str
    value: str
    reason: str


@dataclass
class Provenance:
    """Accumulates which values in a result came from guesses."""

    notes: List[PlaceholderNote] = field(default_factory=list)

    def add(self, key: str, value, reason: str) -> None:
        self.notes.append(PlaceholderNote(key, str(value), reason))

    @property
    def any_placeholder(self) -> bool:
        return bool(self.notes)

    def report(self) -> str:
        if not self.notes:
            return "All geometry from measured hardpoints."
        lines = [f"{len(self.notes)} PLACEHOLDER value(s) in use:"]
        for n in self.notes:
            lines.append(f"  {n.key} = {n.value}")
            lines.append(f"      {n.reason}")
        return "\n".join(lines)


def wheel_plane_basis(spin_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis of the wheel plane: (forward-ish, upward-ish).

    `spin_axis` must be the VEHICLE-FRAME spin axis (+y-oriented), not the raw
    outboard spindle axis -- otherwise the basis flips handedness between the
    two sides of the car and every synthesized point mirrors incorrectly.

    Returns (e_fwd, e_up) with e_fwd x e_up = spin_axis, so a point at angle
    `theta` measured from e_fwd toward e_up has tangential direction
    `spin_axis x r_hat` for a positive rotation about the spin axis.
    """
    n = np.asarray(spin_axis, dtype=float)
    n = n / np.linalg.norm(n)

    e_f = np.array([1.0, 0.0, 0.0]) - float(np.array([1.0, 0.0, 0.0]) @ n) * n
    if np.linalg.norm(e_f) < 1e-6:
        # Spin axis is essentially longitudinal, which no wheel on a car is,
        # but fall back rather than divide by ~0.
        e_f = np.array([0.0, 0.0, 1.0]) - float(np.array([0.0, 0.0, 1.0]) @ n) * n
    e_f = e_f / np.linalg.norm(e_f)
    e_u = np.cross(n, e_f)
    e_u = e_u / np.linalg.norm(e_u)
    # Enforce e_f x e_u = +n (rather than trusting the cross order), so the
    # sign of every tangential force downstream is fixed by construction.
    if float(np.cross(e_f, e_u) @ n) < 0.0:
        e_u = -e_u
    return e_f, e_u


#: Caliper centre angle measured from straight-ahead, toward "up", in the wheel
#: plane. 180 deg puts it at the trailing edge of the rotor.
CALIPER_CLOCK_DEG: float = 180.0

#: Tangential separation of the two mount bolts. Typical FSAE radial mount.
CALIPER_BOLT_SPACING_M: float = 0.060


def synth_caliper_mounts(wheel_centre: np.ndarray, spin_axis: np.ndarray,
                         rotor_effective_radius: float,
                         clock_deg: float = CALIPER_CLOCK_DEG,
                         bolt_spacing: float = CALIPER_BOLT_SPACING_M,
                         prov: Optional[Provenance] = None,
                         ) -> Dict[str, np.ndarray]:
    """Two placeholder caliper mount points, and the pad-force centroid.

    Returns {"centroid": p, "bolt_1": .., "bolt_2": .., "r_eff": r} -- points in
    vehicle coordinates, `r_eff` in metres.

    The centroid is the point at which the pad friction resultant acts: on the
    rotor's effective radius, at the caliper's clock position. Representing the
    caliper reaction as a single force there reproduces BOTH the net force and
    the brake couple exactly, which is why the bolt points are reported but not
    required for equilibrium.

    `r_eff` is the centroid's ACTUAL radial distance from the wheel centre, and
    the caller must divide brake torque by it rather than by its own config
    value -- see `resolve_caliper_mounts`. Here they are the same number by
    construction, because the centroid is built at `rotor_effective_radius`.

    The synthesized centroid sits in the WHEEL-CENTRE PLANE -- zero axial offset
    -- which is a guess like the clock position, and one that drops the radial
    moment described in `PAD_CENTRE_POINT_NAMES`.
    """
    wc = np.asarray(wheel_centre, dtype=float)
    n = np.asarray(spin_axis, dtype=float)
    n = n / np.linalg.norm(n)
    e_f, e_u = wheel_plane_basis(n)

    theta = np.radians(clock_deg)
    r_hat = np.cos(theta) * e_f + np.sin(theta) * e_u
    t_hat = np.cross(n, r_hat)

    centroid = wc + rotor_effective_radius * r_hat
    half = 0.5 * bolt_spacing
    out = {
        "centroid": centroid,
        "bolt_1": centroid + half * t_hat,
        "bolt_2": centroid - half * t_hat,
        "r_eff": float(rotor_effective_radius),
    }
    if prov is not None:
        prov.add("caliper_mounts",
                 f"synthesized at {clock_deg:.0f} deg clock, "
                 f"{bolt_spacing * 1000:.0f} mm bolt spacing, "
                 f"in the wheel-centre plane",
                 "No caliper mount points in the hardpoint sheet. Reaction "
                 "MAGNITUDE is set by measured brake torque and rotor radius "
                 "and is trustworthy; its DIRECTION on the upright follows "
                 "this assumed clock position and is not, and the pad centroid "
                 "is assumed to sit in the wheel-centre plane, which drops the "
                 "radial moment the rotor's axial offset carries. Add "
                 f"{CALIPER_POINT_NAMES[0][0]}/{CALIPER_POINT_NAMES[0][1]} and "
                 f"{PAD_CENTRE_POINT_NAMES[0]} to the CSV to replace it.")
    return out


#: How far a sheet-supplied pad centroid's own radius may differ from config
#: `brakes.rotor_effective_radius_m` before it is flagged.
#:
#: THE TWO ARE ALLOWED TO DISAGREE A LITTLE, AND THE REASON IS NOT THE ROTOR.
#: The radius is measured from the WHEEL CENTRE, and the wheel centre is not in
#: the hardpoint sheet at all -- it comes from `wheels.centre_m.<axle>`, i.e.
#: from the track figure. So a radius derived from `BPad_Center` inherits the
#: track's error, and a millimetre or two of honest disagreement with a
#: `[MEASURED]` rotor radius says nothing is wrong.
#:
#: 2 mm is the line because the failure this must catch is not a small one: the
#: classic error is a coordinate typed into the wrong column of the sheet -- the
#: SolidWorks X column is the AXIAL one, not an in-plane one -- which moves the
#: radius by tens of millimetres and cannot hide under 2 mm. Same spirit as the
#: "sheet carries a transcribed estimate" note in `resolve_bearing_offsets`:
#: tolerate the noise, name the disagreement.
PAD_RADIUS_TOL_M: float = 0.002


def resolve_caliper_mounts(points: Optional[Dict[str, np.ndarray]],
                           wheel_centre: np.ndarray, spin_axis: np.ndarray,
                           rotor_effective_radius: float,
                           prov: Optional[Provenance] = None,
                           static_wheel_centre: Optional[np.ndarray] = None,
                           static_spin_axis: Optional[np.ndarray] = None,
                           ) -> Dict[str, np.ndarray]:
    """The pad friction centroid, from the sheet where the sheet has it.

    Returns {"centroid": p, "bolt_1": .., "bolt_2": .., "r_eff": r}. `points` is
    the raw hardpoint dict; pass None to force synthesis.

    `wheel_centre` / `spin_axis` are the POSED frame -- where the corner is in
    this load case -- and the returned centroid is in it. `static_wheel_centre` /
    `static_spin_axis` are the DESIGN-position frame the sheet's coordinates are
    written in; both default to the posed pair, which is right only when the
    corner is at design position. See "THE TWO FRAMES" below.

    THREE RUNGS, BEST FIRST
    ------------------------
    (a) `BPad_Center` (see `PAD_CENTRE_POINT_NAMES`) -- a measured point, so all
        three of its coordinates are used, including the AXIAL one that is the
        whole reason it exists: it carries the radial moment `(T/r_eff) * dy`
        that a centroid pinned to the wheel-centre plane throws away. Bolts come
        from `CALIPER_POINT_NAMES` when present and are synthesized tangentially
        around the centroid when not -- synthesizing BOLTS is harmless, since
        they enter no equilibrium equation, so that fallback is noted as being
        about the bolts alone and never taints the centroid.

    (b) mount bolts only -- the centroid goes on the rotor's effective radius in
        the direction of the bolts' midpoint, in the wheel-centre plane. The
        CLOCK position is then real, but the AXIAL position is an assumption,
        and it is recorded as one.

    (c) neither -- `synth_caliper_mounts`, where the clock position is a guess
        too.

    THE TWO FRAMES, AND WHY (a) IS REBUILT RATHER THAN COPIED
    ----------------------------------------------------------
    The sheet's points are DESIGN-POSITION geometry. The `wheel_centre` and
    `spin_axis` this function is called with are the POSED ones -- the corner
    has moved in bump and steer. Subtracting a posed wheel centre from a
    design-position point would fold the suspension's own travel straight into
    the pad centroid's offset, which is not an offset of the rotor at all.

    Under the old behaviour that mismatch was harmless, because only a unit
    DIRECTION survived the subtraction and a few millimetres of travel barely
    rotates it. Keeping the axial coordinate makes it first-order, so the point
    is decomposed in the STATIC frame into the three body-fixed numbers that
    describe where the rotor is on the upright --

        dr      radial distance from the wheel centre  (this is `r_eff`)
        theta   clock angle in the static wheel-plane basis
        dy      offset along the spin axis

    -- and rebuilt at the posed frame. Exactly the trade `resolve_bearing_offsets`
    makes with the seat positions: reduce the sheet point to body-fixed scalars
    where the sheet's frame is valid, then re-apply them at whatever pose the
    case is in.

    `centroid` is therefore always in the POSED frame. Bolt points that came
    from the sheet are passed straight back in the sheet's own design-position
    coordinates, un-posed: nothing here computes with them, and a caller that
    wants them at the pose has the upright's own rigid transform, which is a
    better answer than anything this function could reconstruct.

    WHICH WAY `dy` POINTS
    ----------------------
    `dy` is measured along `spin_axis`, which `_spin_axis_vehicle_frame` has
    already oriented toward **+y, ISO LEFT, on both sides of the car**. So `dy`
    is positive for a rotor centreplane LEFT of the wheel centre, whichever
    corner this is: positive is inboard on a left corner and OUTBOARD on a right
    one. That is deliberately NOT the convention its neighbour
    `resolve_bearing_offsets` uses -- that one is handed the raw outboard
    spindle axis, which flips sign side to side, so its offsets are
    outboard-positive everywhere. The two functions sit nine lines apart and
    disagree about the sign of "outboard"; read the axis, not the word.

    The physics does not care which way the axis was oriented -- the radial
    moment `(T/r_eff) * dy` flips sign with `dy` and with `t_hat` together --
    but anyone reading a `dy` out of a provenance note does.

    WHY `r_eff` COMES BACK OUT
    ---------------------------
    `caliper_force` builds the pad force as `torque / r_eff`, and the couple it
    reproduces is `torque` only when that `r_eff` is the centroid's OWN radial
    distance from the wheel centre. On rungs (b) and (c) the centroid is
    CONSTRUCTED at `rotor_effective_radius`, so the two agree by definition. On
    rung (a) they need not: the sheet's point is at whatever radius the sheet
    puts it, and dividing by a config number that disagrees would reproduce a
    couple of `T * r_sheet / r_config` -- the wrong braking torque, arrived at
    silently, with the corner's free body no longer closing.

    So geometry wins, exactly as it does for the bearing seats (see
    `resolve_bearing_offsets`, "WHY THE SHEET WINS OVER THE CONFIG"): the radius
    ACTUALLY used is returned and the caller must use it. A disagreement beyond
    `PAD_RADIUS_TOL_M` is recorded with both numbers rather than raised, because
    the run is still self-consistent -- it is the config that is stale, and
    stopping the sweep over a stale config key helps nobody.
    """
    wc = np.asarray(wheel_centre, dtype=float)
    n = np.asarray(spin_axis, dtype=float)
    n = n / np.linalg.norm(n)

    # The frame the SHEET's coordinates live in. Defaulting it to the posed
    # frame keeps every existing caller and test working unchanged -- and is
    # exactly right whenever the corner is at design position -- while the one
    # production call site passes the static geometry explicitly.
    wc_s = wc if static_wheel_centre is None else np.asarray(
        static_wheel_centre, dtype=float)
    n_s = n if static_spin_axis is None else np.asarray(
        static_spin_axis, dtype=float)
    n_s = n_s / np.linalg.norm(n_s)

    if points:
        bolts = None
        bolt_names = None
        for a, b in CALIPER_POINT_NAMES:
            if a in points and b in points:
                bolts = (np.asarray(points[a], dtype=float),
                         np.asarray(points[b], dtype=float))
                bolt_names = (a, b)
                break

        # ---- (a) a real pad centroid ----
        for name in PAD_CENTRE_POINT_NAMES:
            if name not in points:
                continue
            # Decomposed in the STATIC frame, where the sheet's coordinates are
            # meaningful; rebuilt at the posed frame below. See "THE TWO FRAMES".
            sheet = np.asarray(points[name], dtype=float)
            d = sheet - wc_s
            dy = float(d @ n_s)
            radial_s = d - dy * n_s
            dr = float(np.linalg.norm(radial_s))
            if dr < 1e-9:
                raise ValueError(
                    f"pad centroid {name} sits on the spindle axis, so the pad "
                    f"friction resultant has no moment arm and cannot carry "
                    f"brake torque at all")

            if abs(dr - rotor_effective_radius) > PAD_RADIUS_TOL_M \
                    and prov is not None:
                prov.add("brakes.rotor_effective_radius_m",
                         f"{dr * 1000:.1f} mm from {name} (config says "
                         f"{rotor_effective_radius * 1000:.1f} mm)",
                         f"The sheet's {name} sits at a different radius than "
                         f"`brakes.rotor_effective_radius_m`. GEOMETRY WINS: "
                         f"brake torque is divided by the point's own radius, "
                         f"because that is the only divisor for which the pad "
                         f"force reproduces the commanded couple and the corner "
                         f"closes. Pad force scales as 1/radius, so the two "
                         f"numbers differ by "
                         f"{abs(dr / rotor_effective_radius - 1.0):.1%} in "
                         f"every brake-derived load. The radius is measured from "
                         f"the wheel centre, which comes from `wheels.centre_m` "
                         f"and not from the sheet, so check the track figure and "
                         f"the sheet's axial column before assuming the rotor "
                         f"changed.")

            # Clock angle in the static basis, re-applied in the posed one. Both
            # bases are built by `wheel_plane_basis` from their own axis, so the
            # angle is the body-fixed quantity that survives the pose.
            e_f_s, e_u_s = wheel_plane_basis(n_s)
            cos_t = float(radial_s @ e_f_s) / dr
            sin_t = float(radial_s @ e_u_s) / dr
            e_f, e_u = wheel_plane_basis(n)
            r_hat = cos_t * e_f + sin_t * e_u
            centroid = wc + dr * r_hat + dy * n

            if bolts is None:
                # Bolts, and ONLY bolts, are synthesized here. They are
                # reporting detail -- where the caliper is fastened to the
                # upright -- and appear in no force or moment equation, so this
                # fallback cannot move a number. The centroid above is measured.
                t_hat = np.cross(n, r_hat)
                half = 0.5 * CALIPER_BOLT_SPACING_M
                bolts = (centroid + half * t_hat, centroid - half * t_hat)
                if prov is not None:
                    prov.add("caliper_bolts",
                             f"synthesized, {CALIPER_BOLT_SPACING_M * 1000:.0f} "
                             f"mm apart, straddling {name}",
                             f"The BOLT POSITIONS only -- the pad centroid is "
                             f"the measured {name} and is not affected. Bolt "
                             f"locations enter no equilibrium equation; they "
                             f"describe where the reaction is fastened to the "
                             f"upright, which matters to a stress model and to "
                             f"nothing computed here. Add "
                             f"{CALIPER_POINT_NAMES[0][0]}/"
                             f"{CALIPER_POINT_NAMES[0][1]} to the CSV to "
                             f"replace them.")

            return {"centroid": centroid, "bolt_1": bolts[0],
                    "bolt_2": bolts[1], "r_eff": dr}

        # ---- (b) mount bolts, but no pad centroid ----
        if bolts is not None:
            a, b = bolt_names
            # The clock angle is read in the STATIC frame and re-applied in the
            # posed one, for the same reason rung (a) is: the bolts are
            # design-position coordinates. Only a direction is taken here, so
            # this correction is second-order -- it is made anyway because a
            # function that mixes frames in one branch and not the other is a
            # trap for whoever edits it next.
            mid = 0.5 * (bolts[0] + bolts[1])
            radial_s = (mid - wc_s) - float((mid - wc_s) @ n_s) * n_s
            if np.linalg.norm(radial_s) < 1e-9:
                raise ValueError(
                    f"caliper mounts {a}/{b} sit on the spindle axis, so "
                    f"the caliper clock position is undefined")
            radial_s = radial_s / np.linalg.norm(radial_s)
            e_f_s, e_u_s = wheel_plane_basis(n_s)
            e_f, e_u = wheel_plane_basis(n)
            r_hat = (float(radial_s @ e_f_s) * e_f
                     + float(radial_s @ e_u_s) * e_u)
            if prov is not None:
                prov.add("caliper_centroid_axial", "wheel-centre plane (assumed)",
                         f"{a}/{b} fix the caliper's CLOCK position, but the "
                         f"bolts are on the upright, not on the rotor "
                         f"centreplane, so they say nothing about where the pad "
                         f"force acts along the spindle axis. The centroid is "
                         f"placed at zero axial offset, which reproduces the "
                         f"brake couple exactly but DROPS the radial moment "
                         f"`(T/r_eff) * dy` that a real centreplane offset `dy` "
                         f"carries -- of order 20 N.m at a typical 25 mm offset, "
                         f"which splits across the bearing rows as a few hundred "
                         f"newtons on OB/IB. Add {PAD_CENTRE_POINT_NAMES[0]} to "
                         f"the CSV to recover it.")
            return {"centroid": wc + rotor_effective_radius * r_hat,
                    "bolt_1": bolts[0], "bolt_2": bolts[1],
                    "r_eff": float(rotor_effective_radius)}

    # ---- (c) nothing at all ----
    return synth_caliper_mounts(wheel_centre, spin_axis,
                                rotor_effective_radius, prov=prov)


#: How far off the spindle axis a bearing seat point may sit before it is
#: flagged. The seats ARE on the axis by construction, so a point further out
#: than this is a mis-typed coordinate or a point measured somewhere else --
#: only its axial component is used either way, so the run continues.
BEARING_RADIAL_TOL_M: float = 0.005


def resolve_bearing_offsets(points: Optional[Dict[str, np.ndarray]],
                            wheel_centre: np.ndarray, spindle_axis: np.ndarray,
                            inner_default: float, outer_default: float,
                            prov: Optional[Provenance] = None,
                            ) -> Tuple[float, float]:
    """Bearing seat offsets along the spindle axis, from the sheet if it has them.

    Returns `(inner_offset, outer_offset)` in metres, measured from the wheel
    centre along the OUTBOARD-pointing spindle axis -- so the outer seat is
    positive and the inner one negative, the same convention the config keys
    `bearings.inner_offset_m` / `bearings.outer_offset_m` use.

    WHY THE SHEET WINS OVER THE CONFIG
    ----------------------------------
    Both describe the same thing, and only one of them is geometry. The config
    numbers are the ESTIMATE this module exists to flag; a sheet carrying
    `IB_Bearing`/`OB_Bearing` is measured hardware, so it takes precedence and
    the estimate is not consulted. That also means editing the seat positions
    is a spreadsheet edit like every other hardpoint, with no second copy in
    the config to disagree with it.

    WHY POINTS, NOT OFFSETS, AND WHY ONLY THE AXIAL PART SURVIVES
    -------------------------------------------------------------
    A seat is a place on the car, so it is measurable in the same frame as
    every other point and needs no separate convention to get wrong. But two
    point forces on a common axis is the free body `hub_bearing_reactions`
    solves, and it is parameterized by axial position alone -- so the point is
    projected onto the axis, and a radial component means the point is wrong,
    not that the model is richer. Anything beyond `BEARING_RADIAL_TOL_M` is
    recorded rather than silently dropped.
    """
    wc = np.asarray(wheel_centre, dtype=float)
    n = np.asarray(spindle_axis, dtype=float)
    n = n / np.linalg.norm(n)

    if points:
        for inner_name, outer_name in BEARING_POINT_NAMES:
            if inner_name not in points or outer_name not in points:
                continue

            offsets = []
            for name in (inner_name, outer_name):
                d = np.asarray(points[name], dtype=float) - wc
                axial = float(d @ n)
                radial = float(np.linalg.norm(d - axial * n))
                if radial > BEARING_RADIAL_TOL_M and prov is not None:
                    prov.add(f"bearings.{name}",
                             f"{radial * 1000:.1f} mm off the spindle axis",
                             "Only the component ALONG the spindle axis is "
                             "used -- the bearing free body is two point "
                             "forces on one axis. A seat this far off the "
                             "axis is more likely a coordinate error than a "
                             "real offset; check the sheet.")
                offsets.append(axial)

            inner, outer = offsets
            if abs(outer - inner) < 1e-6:
                raise ValueError(
                    f"{inner_name}/{outer_name} project to the same point on "
                    f"the spindle axis ({inner * 1000:.1f} mm from the wheel "
                    f"centre). Bearing spacing splits the contact-patch "
                    f"moment between the rows, so it cannot be zero.")
            if outer < inner:
                # Not a swap to fix silently: which row is which decides where
                # the thrust load lands via `bearings.axially_located`, so a
                # sheet that has them backwards must be corrected in the sheet.
                raise ValueError(
                    f"{outer_name} sits INBOARD of {inner_name} along the "
                    f"spindle axis ({outer * 1000:.1f} mm vs "
                    f"{inner * 1000:.1f} mm, measured outboard-positive from "
                    f"the wheel centre). The two are swapped in the sheet.")

            # A PAIR THAT DOES NOT STRADDLE THE WHEEL CENTRE CHANGES THE ANSWER
            # BY A FACTOR, NOT A PERCENT, SO IT IS REPORTED EVERY TIME.
            #
            # `hub_bearing_reactions` solves it without complaint -- two point
            # forces on an axis with the load outside them is a perfectly good
            # cantilever -- so nothing downstream fails and nothing looks odd.
            # What comes out is a much larger, lopsided pair, because the rows
            # react on a lever arm instead of sharing the load.
            #
            # NFR27 is such a hub, confirmed with the team: both seats 21.4 and
            # 56.9 mm OUTBOARD of the wheel centre, 35.5 mm apart, at both ends.
            # Its predicted row loads run ~2.3x the team's own spreadsheet,
            # which is still built on last year's straddling ~70 mm hub.
            #
            # So this is a NOTE, not a warning: the geometry is legitimate and
            # refusing to model it would be worse than reporting it. What it
            # must not do is stay invisible, because a reader who assumes a
            # straddling pair will read these numbers as wrong when they are
            # right -- or, worse, size hardware off the smaller ones.
            if inner * outer > 0.0 and prov is not None:
                side = "outboard" if inner > 0.0 else "inboard"
                prov.add("bearing_straddle",
                         f"cantilevered hub -- both rows {side} of the wheel "
                         f"centre ({inner * 1000:+.1f} / {outer * 1000:+.1f} mm, "
                         f"span {(outer - inner) * 1000:.1f} mm)",
                         f"{inner_name} and {outer_name} are on the SAME side "
                         f"of the wheel centre, so the rows carry a lever-arm "
                         f"reaction rather than splitting the load, and both "
                         f"come out several times larger than a straddling pair "
                         f"of the same span would give. Intended on this car; "
                         f"verify against CAD on any other. Row loads also scale "
                         f"roughly as 1/span, so they are only as good as these "
                         f"two seats and the wheel centre (`wheels.centre_m`, "
                         f"i.e. the track) they are measured from.")

            # A SHEET CAN CARRY A TRANSCRIBED ESTIMATE, AND IT STILL IS ONE.
            #
            # The NFR26 sheets ship with the seats written in at the config's
            # symmetric +/-35 mm precisely so they are easy to edit -- which
            # would otherwise LAUNDER the estimate into "measured hardpoint"
            # the moment it entered the CSV, and silence the note that says
            # row loads scale as 1/spacing. Reproducing the fallback to within
            # 0.1 mm is taken as the estimate still being in place. A real
            # measurement that lands there anyway gets a note asking for a
            # second look, which costs nothing.
            if (abs(inner - inner_default) < 1e-4
                    and abs(outer - outer_default) < 1e-4 and prov is not None):
                prov.add("bearing_offsets",
                         f"inner {inner * 1000:.1f} mm, "
                         f"outer {outer * 1000:.1f} mm (sheet, = the estimate)",
                         f"{inner_name}/{outer_name} are in the sheet but "
                         f"match the `bearings.*_offset_m` estimate exactly, "
                         f"so they are almost certainly it transcribed rather "
                         f"than measured seats. Individual row loads scale "
                         f"roughly as 1/spacing; edit the two rows in the "
                         f"hardpoint CSV once the real seats are known.")
            return inner, outer

    if prov is not None:
        prov.add("bearing_offsets",
                 f"inner {inner_default * 1000:.1f} mm, "
                 f"outer {outer_default * 1000:.1f} mm (config estimate)",
                 "No bearing seat points in the hardpoint sheet, so the "
                 "`bearings.*_offset_m` config estimate is in use. Individual "
                 "row loads scale roughly as 1/spacing, so they are only as "
                 "good as that guess. Add "
                 f"{BEARING_POINT_NAMES[0][0]}/{BEARING_POINT_NAMES[0][1]} to "
                 "the CSV to replace it.")
    return float(inner_default), float(outer_default)


def resolve_thrust_row(axially_located, prov: Optional[Provenance] = None) -> str:
    """Which bearing row carries axial thrust: "inner", "outer", or "split".

    Config `bearings.axially_located` is currently UNCONFIRMED, which is not a
    number and must not be silently coerced into one. Anything unrecognised
    falls back to "outer" -- the more common arrangement, and the conservative
    one for the outer row -- and is recorded as a placeholder.
    """
    val = str(axially_located).strip().lower()
    if val in {"inner", "outer", "split"}:
        return val
    if prov is not None:
        prov.add("bearings.axially_located", "outer (assumed)",
                 f"config says {axially_located!r}. Axial load is assigned "
                 f"entirely to the OUTER row. This changes each row's axial "
                 f"share but not the radial split, and not any link force.")
    return "outer"

"""Suspension hardpoint geometry -- topology-generic.

THE CENTRAL ABSTRACTION
-----------------------
A corner is a LIST OF LINKS, never a named topology. This is not a stylistic
choice; it falls out of the kinematics:

    double wishbone : UCA (2 inboard -> 1 shared outboard)
                    + LCA (2 inboard -> 1 shared outboard)
                    + tie rod                                 = 5 constraints
    five-link       : upper wishbone (2 -> 1 shared)
                    + 2 independent lower links
                    + toe link                                = 5 constraints

Both are five two-force members plus one actuation rod. A double wishbone is
simply a five-link in which two pairs of links share an outboard node. The
solver, the force extractor and the K&C maps therefore need no knowledge of
which is which -- and NFR26 runs a double wishbone at the front and a five-link
at the rear, so this is exercised, not hypothetical.

Modelling an A-arm as two independent length constraints from its two chassis
pickups to the outboard ball joint is exact for a rigid A-arm with a spherical
outboard joint. See docs/04 SSA.1.

FRAME
-----
Internal storage is ISO 8855 in metres: x forward, y left, z up, origin at the
ground plane under the wheelbase midpoint.

The NFR26 sheets carry three coordinate blocks (OptimumK / SolidWorks / MATLAB
inches). SolidWorks is the common frame across the front and rear files, and
maps to ISO as a pure axis permutation with NO sign flips:

    ISO_x = SW_z    (forward)
    ISO_y = SW_x    (left)
    ISO_z = SW_y    (up)

Verified against quantities that depend only on hardpoints, not on any
construction convention: caster 4.5712 vs 4.571 reported, kingpin 4.8186 vs
4.819, scrub 34.160 vs 34.16 mm, and the steering axis meeting the ground
765.176 mm forward of origin against a wheelbase/2 of 765.175.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.units import assert_plausible_geometry_dataset

__all__ = [
    "Link", "Actuator", "CornerGeometry",
    "DOUBLE_WISHBONE_PUSHROD", "FIVE_LINK_PULLROD",
    "ROD_PICKUPS", "DEFAULT_ROD_PICKUP",
    "load_hardpoints_csv", "build_corner",
    "write_hardpoints_csv", "blank_template_csv",
]

#: Where the push/pullrod's OUTBOARD end picks up. A STRUCTURAL-TOPOLOGY
#: choice, not a parameter: it decides which free-body idealization is valid,
#: not just a coefficient inside one (docs/11 SS5).
#:
#:   "upper_wishbone"  the upper arm carries the rod -> upper arm is a BENDING
#:                     member, its ball-joint reaction on the upright is a
#:                     general 3D force, and the lower links stay two-force.
#:                     NFR26 at both ends.
#:   "lower_wishbone"  the same picture with the roles SWAPPED -- common on
#:                     pullrod cars. The lower arm bends; the upper becomes two
#:                     clean two-force links.
#:   "upright"         the rod picks up on the upright itself. Every member is
#:                     then a two-force member and the plain 6x6
#:                     (`forces.solve_link_forces`) is the right free body.
ROD_PICKUPS = ("upper_wishbone", "lower_wishbone", "upright")

#: What an undeclared pickup means. This is the historical behaviour of this
#: module (the rod was ALWAYS carried by the upper arm when one could be
#: identified), kept as the default so a corner built without an explicit
#: declaration behaves exactly as it did before the field existed. Config-driven
#: builds never rely on it -- `core.config.build_corner_geometry` requires
#: `suspension.<axle>.rod_pickup` to be a member of `ROD_PICKUPS` and raises
#: otherwise.
DEFAULT_ROD_PICKUP = "upper_wishbone"


@dataclass(frozen=True)
class Link:
    """A two-force member. `inboard` is chassis-fixed, `outboard` upright-fixed."""

    name: str
    inboard: np.ndarray
    outboard: np.ndarray
    steered: bool = False
    """True for the tie rod / toe link: its inboard point moves with the rack."""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.outboard - self.inboard))

    @property
    def unit(self) -> np.ndarray:
        v = self.outboard - self.inboard
        return v / np.linalg.norm(v)


@dataclass(frozen=True)
class Actuator:
    """Push/pull rod or direct-acting spring-damper.

    `kind` is descriptive only. Pushrod, pullrod and direct-acting differ in
    the SIGN and MAGNITUDE of the motion ratio, which the kinematics solver
    computes -- not in the equations. Treating them as one object with a
    derived motion ratio is what makes them interchangeable per the
    year-to-year generality requirement.
    """

    kind: str                     # "pushrod" | "pullrod" | "direct"
    outboard: np.ndarray          # on the upright, or on a control arm
    inboard: np.ndarray           # on the rocker, or on the chassis if direct
    rocker_pivot: Optional[np.ndarray] = None
    rocker_axis: Optional[np.ndarray] = None
    rocker_spring: Optional[np.ndarray] = None
    spring_chassis: Optional[np.ndarray] = None

    rod_pickup: str = DEFAULT_ROD_PICKUP
    """WHICH BODY THE OUTBOARD END BOLTS TO -- one of `ROD_PICKUPS`.

    Replaces the never-set, never-read `mounted_on_link`. It is a declaration,
    not a hint: `kinematics.rod_outboard_point` and the force extractors read
    it directly rather than inferring the layout from the topology, because
    inference cannot tell an upright-mounted rod on a car WITH an upper
    wishbone (an ordinary layout) from a wishbone-mounted one. Measured on the
    NFR26 front corner at 25 mm of bump, the two constructions put the pickup
    4.05 mm apart in z and the rod length 2.7 mm apart (0.36697 m carried by the
    arm against 0.36429 m carried by the upright) -- which propagates into
    motion ratio, wheel rate, roll stiffness and every link force."""

    def __post_init__(self) -> None:
        if self.rod_pickup not in ROD_PICKUPS:
            raise ValueError(
                f"rod_pickup={self.rod_pickup!r} is not a known pickup "
                f"(have {sorted(ROD_PICKUPS)}).")

    @property
    def has_rocker(self) -> bool:
        return self.rocker_pivot is not None

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.inboard - self.outboard))


# Topology = an ordered list of (inboard point name, outboard point name).
# Adding a new suspension type means adding a list here, not writing code.
LinkSpec = Tuple[str, str, bool]      # (inboard, outboard, is_steered)

DOUBLE_WISHBONE_PUSHROD: Sequence[LinkSpec] = (
    ("IB_UppLead", "OB_UppPnt", False),
    ("IB_UppTrail", "OB_UppPnt", False),
    ("IB_LowLead", "OB_LowPnt", False),
    ("IB_LowTrail", "OB_LowPnt", False),
    ("IB_TiePnt", "OB_TiePnt", True),
)

FIVE_LINK_PULLROD: Sequence[LinkSpec] = (
    ("IB_UppLead", "OB_UppPnt", False),
    ("IB_UppTrail", "OB_UppPnt", False),
    ("IB_LowLead", "OB_Low_Lead", False),
    ("IB_LowTrail", "OB_Low_Trail", False),
    ("IB_TiePnt", "OB_TiePnt", True),
)


@dataclass
class CornerGeometry:
    """One corner's hardpoints, in ISO metres."""

    name: str
    links: List[Link]
    actuator: Optional[Actuator] = None
    wheel_centre: Optional[np.ndarray] = None
    spindle_axis: Optional[np.ndarray] = None
    contact_patch: Optional[np.ndarray] = None
    loaded_radius: Optional[float] = None
    points: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.links) != 5:
            raise ValueError(
                f"{self.name}: got {len(self.links)} links, expected exactly 5. "
                f"A corner with 5 two-force members plus one actuation rod has "
                f"exactly one degree of freedom; any other count is either "
                f"over-constrained or under-constrained and the pose solver "
                f"will not converge."
            )

    @property
    def steered_links(self) -> List[Link]:
        return [l for l in self.links if l.steered]

    @property
    def link_lengths(self) -> np.ndarray:
        return np.array([l.length for l in self.links])


# ---------------------------------------------------------------------------
# NFR26 spreadsheet loader
# ---------------------------------------------------------------------------

# SolidWorks (x lateral-left, y up, z forward) -> ISO (x fwd, y left, z up).
_SW_COLUMNS = {"iso_x": 2, "iso_y": 0, "iso_z": 1}

#: Column of the "SolidWorks" label in the original multi-block sheets, used
#: only when the header cell cannot be found at all.
_LEGACY_LABEL_COLUMN = 5

#: Header cell that marks the start of the SolidWorks block.
_SW_HEADER = "solidworks"


def _find_label_column(path: Path) -> int:
    """Column index of the SolidWorks block's label, found by its header.

    WHY THIS IS SEARCHED FOR RATHER THAN HARDCODED
    ----------------------------------------------
    The original sheets are a wide export with three coordinate systems side by
    side -- OptimumK, SolidWorks, MATLAB -- and the SolidWorks label happens to
    land in column 6. Hardcoding that made the loader unable to read a sheet
    containing ONLY the SolidWorks points, which is the format
    `write_hardpoints_csv` produces and the one the template hands out.

    Locating the block by its header reads both, and it is what makes the
    simplified sheets possible without a second parser to disagree with this
    one. Matched exactly (case-insensitively) so the "SolidWorks Coordinates:"
    caption further down the sheet is not mistaken for the header.
    """
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            for i, cell in enumerate(row):
                if cell.strip().lower() == _SW_HEADER:
                    return i
    return _LEGACY_LABEL_COLUMN


def load_hardpoints_csv(path, overrides: Optional[Dict[str, float]] = None,
                        scale: float = 1e-3) -> Dict[str, np.ndarray]:
    """Read a 'Suspension Points' sheet into ISO metres.

    Args:
        path: the CSV. Either the full multi-block export or the simplified
            SolidWorks-only sheet -- the block is located by its header.
        overrides: {"POINT.axis": value_mm} patches applied after parsing, for
            cells the source sheet leaves blank. Axis is one of x/y/z in ISO.
        scale: source-to-metre factor; sheets are in mm.

    The SolidWorks block is used rather than OptimumK because the front and
    rear sheets share the SolidWorks origin, while their OptimumK X origins
    differ by the wheelbase.
    """
    path = Path(path)
    pts: Dict[str, np.ndarray] = {}
    blank: List[str] = []
    label_col = _find_label_column(path)

    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            # Label, then three coordinate columns.
            if len(row) < label_col + 4:
                continue
            name = row[label_col].strip()
            if not name or name.lower() == _SW_HEADER or name in {"x", "Origin"}:
                continue

            raw = [row[label_col + 1].strip(), row[label_col + 2].strip(),
                   row[label_col + 3].strip()]

            # Discriminate an ANNOTATION row from a HARDPOINT WITH A MISSING
            # CELL by how many coordinates parse as numbers. These sheets
            # interleave notes ("Right Lateral | neg x", "Forward | pos z")
            # into the same columns, so a blanket "skip if anything is blank"
            # would silently drop IB_LowLead -- the exact point whose absence
            # started this. Zero parseable values means prose; one or two means
            # a real point with a hole in it, which must be loud.
            vals: List[float] = []
            n_numeric = 0
            for v in raw:
                try:
                    vals.append(float(v))
                    n_numeric += 1
                except ValueError:
                    vals.append(0.0)

            if n_numeric == 0:
                continue                      # annotation / prose row
            if n_numeric < 3:
                blank.append(name)

            sw = np.array(vals, dtype=float)
            pts[name] = np.array([sw[_SW_COLUMNS["iso_x"]],
                                  sw[_SW_COLUMNS["iso_y"]],
                                  sw[_SW_COLUMNS["iso_z"]]]) * scale

    if overrides:
        for key, mm in overrides.items():
            pname, axis = key.rsplit(".", 1)
            if pname not in pts:
                raise KeyError(f"override for unknown point {pname!r}")
            pts[pname]["xyz".index(axis)] = mm * scale
            if pname in blank:
                blank.remove(pname)

    if blank:
        raise ValueError(
            f"{path.name}: blank/unparseable coordinates for {sorted(set(blank))}. "
            f"The NFR26 front sheet leaves IB_LowLead's longitudinal cell empty "
            f"and all three coordinate blocks derive from it, so it reads as "
            f"zero rather than as missing. Supply an override, e.g. "
            f"{{'IB_LowLead.x': 847.6}}. Refusing to silently use 0."
        )

    # A sheet with no points is a legitimate parse result -- a blank template is
    # exactly that -- and it must come back as an empty dict rather than dying
    # inside numpy on a zero-size reduction. `validate_hardpoints` is what turns
    # "nothing parsed" into a message a user can act on, and it cannot do that
    # if this raises `ValueError: zero-size array to reduction` first.
    if pts:
        assert_plausible_geometry_dataset(np.array(list(pts.values())),
                                          context=path.name)
    return pts


# ---------------------------------------------------------------------------
# Writing the simplified sheet
# ---------------------------------------------------------------------------

#: The sign-convention block. These rows carry no numbers, so the loader treats
#: them as prose and skips them -- they are for the human holding the file.
#: They are NOT decoration: a sheet whose axes are transposed parses cleanly and
#: builds a mirror-image car, and this is the only place the convention is
#: written down next to the numbers it applies to.
_CONVENTION_ROWS = [
    [],
    ["SolidWorks Coordinates:", "", "", "", ""],
    # Origin verified from the sheets themselves, not assumed: the front and
    # rear SW_z values straddle zero at +/-765 mm (half the 1530 mm wheelbase),
    # and every SW_y is positive with the wheel centre at the loaded radius.
    ["Origin", "wheelbase midpoint, vehicle centreline, ground plane", "", "", ""],
    ["Right Lateral", "neg x", "", "", ""],
    ["Upwards", "pos y", "", "", ""],
    ["Forward", "pos z", "", "", ""],
    ["*for right geo", "", "", "", ""],
    ["*units are millimetres", "", "", "", ""],
]

_HEADER_ROW = ["SolidWorks", "X", "Y", "Z", "Note"]

#: Point order for a written sheet: links first, then actuation, then rocker,
#: then optional extras. Groups are separated by a blank row for readability.
_WRITE_GROUPS: Sequence[Tuple[str, Sequence[str]]] = (
    ("links", ("IB_LowLead", "IB_LowTrail", "IB_UppLead", "IB_UppTrail",
               "OB_LowPnt", "OB_Low_Lead", "OB_Low_Trail", "OB_UppPnt",
               "IB_TiePnt", "OB_TiePnt")),
    ("actuation", ("OB_Push", "IB_Push", "OB_Pull", "IB_Pull")),
    ("rocker", ("Rocker_Frame", "Rocker_Spring", "Rocker_Axis",
                "Spring_Chassis")),
    ("brakes", ("BC_Upper", "BC_Lower", "BPad_Center")),
    ("bearings", ("IB_Bearing", "OB_Bearing")),
)

_POINT_NOTES = {
    "Rocker_Axis": "*perp to rocker plane",
    "BC_Upper": "optional -- caliper mount, else synthesized",
    "BC_Lower": "optional -- caliper mount, else synthesized",
    "BPad_Center": "optional -- brake pad friction centroid, on the rotor "
                   "centreplane",
    "IB_Bearing": "optional -- inboard bearing seat, on the spindle axis",
    "OB_Bearing": "optional -- outboard bearing seat, on the spindle axis",
}


def _iso_to_solidworks(p: np.ndarray, scale: float = 1e-3) -> List[float]:
    """ISO metres -> the sheet's own SolidWorks millimetres.

    Exactly the inverse of the read mapping, derived from `_SW_COLUMNS` rather
    than written out again, so the two cannot drift apart.
    """
    sw = [0.0, 0.0, 0.0]
    for iso_index, key in enumerate(("iso_x", "iso_y", "iso_z")):
        sw[_SW_COLUMNS[key]] = float(p[iso_index]) / scale
    return sw


def write_hardpoints_csv(points: Dict[str, np.ndarray], path,
                         scale: float = 1e-3, title: Optional[str] = None) -> None:
    """Write a simplified, SolidWorks-only sheet that this loader can read back.

    The output contains the points and the sign conventions, and nothing else:
    no OptimumK or MATLAB duplicate blocks, and no derived parameters (caster,
    scrub, roll centre, anti-dive...). Those parameters are OUTPUTS of the
    hardpoints -- this repo derives them in `suspension/derived.py` -- so
    carrying a second, hand-entered copy alongside the geometry only creates
    something to disagree with.

    Points not in `_WRITE_GROUPS` are still written, in a trailing group, so a
    sheet that carries something this module has not heard of survives a
    round trip rather than losing it silently.
    """
    path = Path(path)
    known = {n for _, names in _WRITE_GROUPS for n in names}
    rows: List[List] = [_HEADER_ROW]

    def _emit(name: str) -> None:
        sw = _iso_to_solidworks(points[name], scale)
        # 12 significant figures: the sheets carry up to 9 (667.170199), and a
        # written sheet that quietly rounds its own input is a slow way to lose
        # geometry across successive edit/export cycles.
        rows.append([name, f"{sw[0]:.12g}", f"{sw[1]:.12g}", f"{sw[2]:.12g}",
                     _POINT_NOTES.get(name, "")])

    for _, names in _WRITE_GROUPS:
        present = [n for n in names if n in points]
        if not present:
            continue
        for n in present:
            _emit(n)
        rows.append([])

    # Anything the topology tables do not name is still written, but LABELLED.
    # These are construction scratch points on the NFR26 sheets ("rockers pt",
    # "bend pt") and alternate constructions ("OB_LowPnt (virtual)"). Dropping
    # them would lose team work; leaving them unmarked invites someone to treat
    # a scratch point as a hardpoint.
    extra = [n for n in points if n not in known]
    if extra:
        rows.append(["*carried through, not used by the model", "", "", "", ""])
        for n in extra:
            _emit(n)
        rows.append([])

    if title:
        rows.append([f"*{title}", "", "", "", ""])
    rows.extend(_CONVENTION_ROWS)

    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def blank_template_csv(topology: str = "double_wishbone",
                       actuation: str = "pushrod") -> str:
    """An empty sheet with every point name this topology needs, as CSV text.

    Handed out by the GUI so a new corner can be filled in without guessing
    point names -- which is the single most common way an uploaded sheet fails
    validation. Required points come first and are marked; optional ones follow
    with a note saying what they unlock.
    """
    from .validate import OPTIONAL_POINTS, required_points

    need = required_points(topology, actuation)
    rows: List[List] = [_HEADER_ROW]
    rows.append([f"*{topology} / {actuation} -- fill X, Y, Z in millimetres",
                 "", "", "", ""])
    rows.append([])

    for name in need:
        rows.append([name, "", "", "", "REQUIRED"])
    rows.append([])

    for name, why in OPTIONAL_POINTS.items():
        if name in need:
            continue
        rows.append([name, "", "", "", f"optional -- {why}"])
    rows.extend(_CONVENTION_ROWS)

    import io
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue()


def build_corner(name: str, points: Dict[str, np.ndarray],
                 topology: Sequence[LinkSpec],
                 actuator_kind: str = "pushrod",
                 rod: Tuple[str, str] = ("IB_Push", "OB_Push"),
                 rod_pickup: str = DEFAULT_ROD_PICKUP) -> CornerGeometry:
    """Assemble a CornerGeometry from a point dict and a topology spec.

    `rod_pickup` is stored on the actuator and is what every downstream
    construction reads -- see `ROD_PICKUPS`.
    """
    missing = [n for spec in topology for n in spec[:2] if n not in points]
    if missing:
        raise KeyError(f"{name}: topology needs points not in the sheet: {sorted(set(missing))}")

    links = [Link(f"{ib}->{ob}", points[ib], points[ob], steered)
             for ib, ob, steered in topology]

    act = None
    if rod[0] in points and rod[1] in points:
        act = Actuator(
            kind=actuator_kind,
            inboard=points[rod[0]], outboard=points[rod[1]],
            rocker_pivot=points.get("Rocker_Frame"),
            rocker_axis=points.get("Rocker_Axis"),
            rocker_spring=points.get("Rocker_Spring"),
            spring_chassis=points.get("Spring_Chassis"),
            rod_pickup=rod_pickup,
        )

    return CornerGeometry(name=name, links=links, actuator=act, points=dict(points))

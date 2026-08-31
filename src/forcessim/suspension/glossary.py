"""What every abbreviation in a force table means.

WHY THIS IS A MODULE AND NOT A COMMENT IN THE EXPORT
-----------------------------------------------------
The attachment-point codes (`UBJ`, `PR`, `OB`...) come from the team's own
spreadsheet, and they are opaque to anyone who did not write it. A force table
whose row labels cannot be decoded is not a deliverable -- and the person who
needs to decode it is usually reading an xlsx six months later, with no access
to this repo.

So the definitions live in one place, and both the GUI and the workbook render
from it. Writing them out separately in each would guarantee they eventually
disagree, and a glossary that disagrees with the table is worse than none.

The codes are also load-bearing in a second way: `OB`/`IB` mean OUTER/INNER
BEARING on the upright and hub, but `_IB` as a SUFFIX (`RL_IB`) means INBOARD.
That collision is inherited from the source spreadsheet and is exactly the kind
of thing a reader has to be told rather than left to infer.

PORT NOTE (forcesSim, phase 4) -- `glossary_rows` trimmed to match the team's
reference workbook (`NFR27ComponentForces.xlsx`), not the NFRFullVehicleSim
original it was ported from: that reference's Key sheet has no "Component"
section (so `COMPONENT_TERMS` is kept, for anyone who wants it, but
`glossary_rows` does not emit it) and no "Left vs right" convention row (so
that entry is not in `CONVENTIONS` below). Both are straightforward to add
back -- the original module has them -- if a future export wants the fuller
Key sheet; this port stays byte-for-byte against the reference instead.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..core.frames import get_report_frame

__all__ = ["POINT_TERMS", "COMPONENT_TERMS", "CONVENTIONS", "case_terms",
           "glossary_rows"]

#: code -> (full term, what the force at that point IS)
POINT_TERMS: Dict[str, Tuple[str, str]] = {
    "UBJ": ("Upper Ball Joint",
            "Force the upper wishbone exerts on the upright at their shared "
            "outboard joint. A general 3D force either way: when the rod picks "
            "up on this arm it carries bending, and when it does not the two "
            "legs' axial forces still meet at one point."),
    "LBJ": ("Lower Ball Joint",
            "The same, for a lower arm that carries the rod (`rod_pickup: "
            "lower_wishbone`). Appears INSTEAD of RL/FL, because a bending arm "
            "hands the upright one 3D force, not two axial ones."),
    "OBJ": ("Outboard Ball Joint",
            "The same joint seen from the wishbone. Equal and opposite to the "
            "upright's UBJ."),
    "RL": ("Rear Link",
           "Rearward lower link, at its outboard end. Two-force member, so the "
           "force is along the link."),
    "FL": ("Front Link",
           "Forward lower link, at its outboard end. Two-force member."),
    "RL_IB": ("Rear Link, InBoard",
              "Chassis pickup of the rearward lower link. Equal and opposite "
              "to RL. Note the collision: a leading `IB` means bearing, a "
              "trailing `_IB` means inboard."),
    "FL_IB": ("Front Link, InBoard",
              "Chassis pickup of the forward lower link."),
    "TR": ("Tie Rod",
           "Steering (front) or toe (rear) link where it meets the upright. "
           "Two-force member."),
    "OB": ("Outer Bearing",
           "Outboard wheel-bearing row. On the upright this is the force the "
           "hub applies; on the hub it is the reaction. Seat position comes "
           "from the hardpoint sheet's `OB_Bearing` row when it has one."),
    "IB": ("Inner Bearing",
           "Inboard wheel-bearing row. Can act OPPOSITE in direction to the "
           "outer row -- the contact-patch moment is what splits them. Seat "
           "position comes from the sheet's `IB_Bearing` row when it has one."),
    "BC": ("Brake Caliper",
           "Pad friction reaction on the upright, tangential at the rotor's "
           "effective radius. It acts at the PAD FRICTION CENTROID, not at the "
           "bolts: the centroid comes from the sheet's `BPad_Center` row when "
           "it has one, which keeps the real axial offset out to the rotor "
           "centreplane, and is otherwise reconstructed radially from the "
           "`BC_Upper`/`BC_Lower` midpoint at the config effective radius. "
           "Zero when the caliper is mounted inboard, because then the couple "
           "never crosses into this free body."),
    "UM": ("Unsprung Mass",
           "Weight plus d'Alembert inertia of the corner assembly, applied at "
           "the wheel centre. Dominated by inertia in the bump case."),
    "PR": ("Push/Pull Rod",
           "Rod force where it picks up. It appears on whichever body carries "
           "it -- upper wishbone (NFR26), lower wishbone, or the upright "
           "itself -- as declared by `suspension.<axle>.rod_pickup`."),
    "FBJ": ("Front Ball Joint",
            "Forward chassis pickup of a wishbone."),
    "RBJ": ("Rear Ball Joint",
            "Rearward chassis pickup of a wishbone. On the arm that CARRIES "
            "the rod, FBJ and RBJ cannot react moment about the line joining "
            "them -- that is what sets the rod force. On an arm that does not, "
            "each simply takes its own leg's axial load."),
    "OP": ("Outboard Point", "The outboard end of a rod or tie rod."),
    "IP": ("Inboard Point", "The inboard end of a rod or tie rod."),
    "RD": ("Rod", "Push/pullrod force on the rocker."),
    "SP": ("Spring",
           "Spring/damper force on the rocker, from moment balance about the "
           "pivot axis."),
    "PV": ("Pivot",
           "Rocker pivot reaction into the chassis. Typically the largest "
           "single load into the frame at that corner."),
    "CP": ("Contact Patch", "Tire force on the hub, at the ground."),
    "BR": ("Brake Rotor",
           "Pad friction on the rotor. A force when the caliper is outboard, "
           "a pure couple when it is inboard."),
}

#: component key -> (sheet name, what the free body is)
COMPONENT_TERMS: Dict[str, Tuple[str, str]] = {
    "upright": ("Uprights",
                "The knuckle. Everything the links, bearings and caliper "
                "apply to it, plus its own unsprung weight and inertia."),
    "wishbone_upper": ("Wishbones Upper",
                       "Upper A-arm. NOT a two-force member on this car -- the "
                       "rod picks up on it, so it carries BENDING. With the rod "
                       "declared elsewhere it reverts to two two-force legs."),
    "wishbone_lower": ("Wishbones Lower",
                       "Lower arm, modelled as two independent two-force "
                       "links from their chassis pickups to the outboard "
                       "joint. Exact for a rigid arm with a spherical joint -- "
                       "and it stops being exact if the rod picks up here, in "
                       "which case this arm is the bending member instead."),
    "rod": ("Push-Pullrods",
            "The push/pullrod -- NOT the tie rod. Which one it is depends on "
            "the axle and is stated in that sheet's header (NFR26: pushrod "
            "front, pullrod rear). The two differ only in geometry and in the "
            "SIGN of the motion ratio, never in the equations. Two-force "
            "member; size against buckling when the value is negative "
            "(compression)."),
    "tie_rod": ("Tie Rods", "Steering or toe link. Two-force member."),
    "rocker": ("Rockers",
               "Bellcrank between rod and spring/damper. One rotational "
               "degree of freedom about its pivot axis."),
    "hub": ("Hubs",
            "Wheel hub and rotor, cut at the bearings and the pads. Takes the "
            "contact patch in and the bearing rows out."),
}

#: Frame-independent conventions. The FRAME row is not here -- it depends on
#: which triad the workbook was written in, so `glossary_rows` builds it from
#: the frame it is given. A Key sheet that says "ISO 8855" on a SolidWorks
#: export would be worse than no Key sheet, because it is the one place a
#: reader goes to resolve exactly that doubt.
CONVENTIONS: List[Tuple[str, str]] = [
    ("Sign", "Every vector is the force applied TO the named component by "
             "whatever attaches at that point. The upright's UBJ and the upper "
             "wishbone's OBJ are the same joint and are equal and opposite."),
    ("Units", "lbf or N as stated in each sheet header. Moments, where shown, "
              "are N.m. The model is SI internally; lbf is a reporting "
              "conversion only."),
    ("Net", "Magnitude of the (Fx, Fy, Fz) vector, not a sum of components."),
    ("Tension/compression",
     "For two-force members (rods, tie rods, lower links) a POSITIVE axial "
     "value is tension. Compression sizes against buckling."),
    ("Envelope", "Worst magnitude per point across all cases, and the case "
                 "that produced it. Steady cases are ALTERNATIVES and are "
                 "never summed; 'bump + <case>' rows are the one legitimate "
                 "superposition, because a bump happens while the car is "
                 "already braking or cornering."),
]


def case_terms(cases: Sequence = ()) -> List[Tuple[str, str]]:
    """(case name, what it means) for the standard cases plus the bump.

    Reads the g values off the `LoadCase` objects rather than repeating them,
    so editing a case cannot leave the glossary describing the old one.
    """
    from ..vehicle.load_cases import STANDARD_CASES

    cases = cases or STANDARD_CASES
    blurb = {
        "braking": "Straight-line braking. Longitudinal force split by brake "
                   "bias, so the front carries most of it.",
        "outside_cornering": "The LOADED corner of a steady turn. Positive a_y "
                             "is a left turn in ISO, which loads the right-hand "
                             "tires.",
        "inside_cornering": "The UNLOADED corner of that same turn -- the same "
                            "geometry at mirrored a_y, not a separate case.",
        "acceleration": "Straight-line acceleration. Split by drive fraction, "
                        "which is NOT the same number as brake bias.",
    }
    out = []
    for c in cases:
        detail = blurb.get(c.name, "")
        out.append((c.name,
                    f"a_x = {c.a_x_g:+.2f} g, a_y = {c.a_y_g:+.2f} g. {detail}"))
    out.append((
        "bump",
        "Single-wheel bump from the 2-DOF quarter car, swept over speed and "
        "reported at the speed producing the worst LINK load -- which is not "
        "the speed producing the worst contact-patch load. Much of the peak "
        "contact force accelerates the unsprung mass instead of reaching the "
        "links."))
    out.append((
        "bump + <case>",
        "The bump's vertical load INCREMENT (peak minus static, so weight is "
        "not counted twice) superposed on that steady case -- a bump happens "
        "WHILE the car is braking or cornering, which is the one combination "
        "that physically co-occurs. Each has its own tab, and they also feed "
        "the Envelope. Extraction is linear in the applied load, so these are "
        "solved cases, not estimates."))
    return out


def _frame_term(frame) -> Tuple[str, str]:
    """The Convention/Coordinate Frame row, describing the triad actually
    exported."""
    frame = get_report_frame(frame)
    origin = ("Origin at the wheelbase midpoint, on the centreline, at the "
              "ground plane.")
    if frame.is_internal:
        return ("Coordinate Frame", f"ISO 8855: x forward, y LEFT, z up. {origin}")
    return ("Coordinate Frame",
            f"{frame.name}: {frame.legend}. {origin} The model works "
            f"internally in ISO 8855 (x forward, y left, z up); these "
            f"components are the SAME vectors written in the "
            f"{frame.name} triad, so magnitudes are unchanged and only the "
            f"column each component lands in differs.")


def glossary_rows(frame=None) -> List[Tuple[str, str, str]]:
    """(section, term, meaning) for every entry, ready to tabulate.

    `frame` is the reporting triad the surrounding table is written in; it only
    affects the Convention/Coordinate Frame row. No "Component" section --
    see the port note at the top of this module.
    """
    rows: List[Tuple[str, str, str]] = []
    for code, (term, meaning) in POINT_TERMS.items():
        rows.append(("Attachment point", f"{code} — {term}", meaning))
    for name, meaning in case_terms():
        rows.append(("Load case", name, meaning))
    for name, meaning in (_frame_term(frame), *CONVENTIONS):
        rows.append(("Convention", name, meaning))
    return rows

"""Coordinate frames, sign conventions, and frame conversions.

THE INTERNAL CONVENTION IS ISO 8855. This is settled -- see docs/03 §1.

    x  forward
    y  left
    z  up

Rotations follow the right-hand rule about those axes:

    roll  (phi)    about +x
    pitch (theta)  about +y
    yaw   (psi)    about +z

------------------------------------------------------------------------------
DESIGN NOTE -- why this module defines conventions rather than describing them
------------------------------------------------------------------------------
Textbook statements about "what SAE says" versus "what ISO says" are a common
source of sign errors, because the standards define *quantities* (slip angle,
inclination angle) with their own sign rules that do not always follow
mechanically from the axis triad.

So this module does not try to encode a complete reading of either standard.
It does two things instead:

  1. Defines OUR convention precisely, for every quantity we use.
  2. Provides the axis transform between ISO 8855 and SAE J670 triads, so that
     any vector or moment lifted from SAE-convention literature (RCVD, most TTC
     material) can be converted at the boundary.
  3. Provides REPORTING frames (`ReportFrame`) -- the triad a force table is
     written down in, e.g. SolidWorks. These are an export-boundary concern
     with no physics attached, and are a separate type from `Convention` for
     that reason; see the section below.

Every convention below is pinned by a test in tests/test_frames.py that states
a concrete physical scenario ("car turning left, outside front tire, ..."), not
by an appeal to a standard. If a convention here is ever wrong, the test is what
tells us -- and the test is checkable against the real car.

References:
    ISO 8855:2011  Road vehicles -- Vehicle dynamics and road-holding ability
    SAE J670       Vehicle Dynamics Terminology
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "Convention",
    "Corner",
    "R_ISO_SAE",
    "convert_vector",
    "convert_moment",
    "convert_euler_angles",
    "slip_angle",
    "slip_ratio",
    "INTERNAL_CONVENTION",
    "ReportFrame",
    "REPORT_FRAMES",
    "DEFAULT_REPORT_FRAME",
    "get_report_frame",
    "to_report_frame",
]


class Convention(enum.Enum):
    """Axis triad conventions."""

    ISO_8855 = "ISO8855"  # x fwd, y left,  z up    <-- ours
    SAE_J670 = "SAEJ670"  # x fwd, y right, z down


INTERNAL_CONVENTION = Convention.ISO_8855


class Corner(enum.Enum):
    """Wheel positions. Order is canonical for all per-corner arrays."""

    FL = 0
    FR = 1
    RL = 2
    RR = 3

    @property
    def is_front(self) -> bool:
        return self in (Corner.FL, Corner.FR)

    @property
    def is_left(self) -> bool:
        return self in (Corner.FL, Corner.RL)

    @property
    def sign_y(self) -> float:
        """+1 for left-side corners, -1 for right-side, in the ISO frame."""
        return 1.0 if self.is_left else -1.0


# ---------------------------------------------------------------------------
# Axis transform: ISO 8855 <-> SAE J670
# ---------------------------------------------------------------------------
#
# ISO -> SAE flips both y and z:   (x, y, z) -> (x, -y, -z)
#
# This is a 180-degree rotation about the x-axis. Two consequences that matter:
#
#   * det(R) = +1, so it is a PROPER rotation. Moments, angular velocities and
#     other pseudovectors therefore transform exactly like vectors -- there is
#     no extra sign flip for axial quantities. (This would NOT be true for a
#     reflection, which is why it is worth stating explicitly.)
#
#   * R is its own inverse (R @ R = I), so ISO->SAE and SAE->ISO are the same
#     operation. `convert_vector` exploits this; there is deliberately only one
#     implementation so the two directions cannot drift apart.
#
R_ISO_SAE = np.diag([1.0, -1.0, -1.0])


def convert_vector(v, frm: Convention, to: Convention):
    """Convert a free vector (force, velocity, position offset) between triads.

    Args:
        v: array_like, shape (..., 3). Last axis is (x, y, z).
        frm, to: source and target conventions.

    Returns:
        ndarray, same shape as `v`.
    """
    v = np.asarray(v, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"expected last axis of length 3, got shape {v.shape}")
    if frm is to:
        return v.copy()
    return v @ R_ISO_SAE.T


def convert_moment(m, frm: Convention, to: Convention):
    """Convert a moment / angular velocity between triads.

    Identical to `convert_vector` because the ISO<->SAE transform is a proper
    rotation (see note above). Provided as a separate name so that call sites
    read unambiguously and so this stays correct if the transform ever changes.
    """
    return convert_vector(m, frm, to)


def convert_euler_angles(roll: float, pitch: float, yaw: float,
                         frm: Convention, to: Convention) -> Tuple[float, float, float]:
    """Convert (roll, pitch, yaw) between triads.

    Under the 180-degree x-rotation, the x-component is preserved and the y and
    z components flip:

        roll  ->  roll        (rotation about x, unchanged)
        pitch -> -pitch
        yaw   -> -yaw
    """
    if frm is to:
        return (roll, pitch, yaw)
    return (roll, -pitch, -yaw)


# ---------------------------------------------------------------------------
# Reporting frames: the axis triad a force TABLE is written in
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT `Convention`
# ----------------------------
# `Convention` is a DYNAMICS convention. Picking ISO over SAE changes the sign
# of slip angle, of inclination angle, and of half the tire model's
# coefficients, and every one of those rules is pinned by a physical test above.
#
# What follows is a REPORTING convention and nothing more: the same force,
# written down with its components in a different order so it can be pasted
# into CAD without a reader doing the permutation by hand. It changes no
# physics, is applied at the export boundary only -- exactly where `to_lbf` is
# applied -- and is deliberately a separate type so that a CAD frame cannot be
# handed to `convert_vector` and silently reinterpreted as a dynamics triad.
#
# HANDEDNESS IS CHECKED, NOT ASSUMED
# ----------------------------------
# A frame is declared as a per-output-axis (source index, sign) pair, and
# `__post_init__` rejects anything whose matrix is not a proper rotation. That
# matters because forces and MOMENTS both get reported: under a proper rotation
# a moment transforms exactly like a vector, so one transform serves both,
# whereas a mirrored frame would need moments flipped separately and every
# rocker torque in the workbook would come out backwards. Adding a left-handed
# frame is therefore a loud failure at import rather than a quiet sign error in
# a sheet someone builds a part from.


@dataclass(frozen=True)
class ReportFrame:
    """An axis triad a force table can be written in.

    `axes[i]` is `(source, sign)`: output axis `i` is `sign` times the internal
    ISO 8855 component at index `source` (0 = forward, 1 = left, 2 = up).
    """

    key: str
    name: str
    axes: Tuple[Tuple[int, float], ...]
    legend: str
    """Human-readable axis meanings, e.g. "x left, y up, z forward". Written
    into every sheet header, because a force table whose frame is not on the
    face of it is the same trap as one with no unit on it."""

    def __post_init__(self) -> None:
        if len(self.axes) != 3:
            raise ValueError(f"{self.key}: need exactly 3 axes, got {len(self.axes)}")
        sources = [s for s, _ in self.axes]
        if sorted(sources) != [0, 1, 2]:
            raise ValueError(
                f"{self.key}: axes must use each ISO component exactly once, "
                f"got sources {sources}")
        if any(sign not in (1.0, -1.0) for _, sign in self.axes):
            raise ValueError(f"{self.key}: axis signs must be +1 or -1")
        det = float(np.linalg.det(self.matrix))
        if abs(det - 1.0) > 1e-12:
            raise ValueError(
                f"{self.key}: not a proper rotation (det = {det:+.1f}). A "
                f"left-handed reporting frame would need moments transformed "
                f"differently from forces; fix the signs instead.")

    @property
    def matrix(self) -> np.ndarray:
        """3x3 `M` with `v_frame = M @ v_iso`."""
        m = np.zeros((3, 3))
        for i, (source, sign) in enumerate(self.axes):
            m[i, source] = sign
        return m

    @property
    def header(self) -> str:
        """"SolidWorks (x left, y up, z forward)" -- for a sheet title."""
        return f"{self.name} ({self.legend})"

    @property
    def is_internal(self) -> bool:
        """True when reporting is a no-op, i.e. this is the ISO 8855 frame."""
        return self.axes == _IDENTITY_AXES


_IDENTITY_AXES: Tuple[Tuple[int, float], ...] = ((0, 1.0), (1, 1.0), (2, 1.0))

#: Every frame a force export can be written in, keyed by the string a caller
#: passes. Additions belong here rather than at a call site, so that the GUI
#: dropdown, the CLI and the workbook can never offer different sets.
REPORT_FRAMES: Dict[str, ReportFrame] = {
    "iso8855": ReportFrame(
        key="iso8855",
        name="ISO 8855",
        axes=_IDENTITY_AXES,
        legend="x forward, y left, z up",
    ),
    # The team's SolidWorks assembly is modelled with the car pointing along
    # +z, standing on -y. Right-handed, and a pure cyclic permutation of ISO
    # with no sign flips: frame x = ISO y, frame y = ISO z, frame z = ISO x.
    "solidworks": ReportFrame(
        key="solidworks",
        name="SolidWorks",
        axes=((1, 1.0), (2, 1.0), (0, 1.0)),
        legend="x left, y up, z forward",
    ),
}

DEFAULT_REPORT_FRAME = REPORT_FRAMES["iso8855"]


def get_report_frame(spec) -> ReportFrame:
    """Resolve `None` / a key / a `ReportFrame` to a `ReportFrame`.

    Accepts the object itself so a caller that already resolved one can pass it
    down without a round trip through the registry, and raises with the valid
    keys listed rather than falling back to the default -- a typo'd frame name
    that silently exported ISO would be indistinguishable from a correct run.
    """
    if spec is None:
        return DEFAULT_REPORT_FRAME
    if isinstance(spec, ReportFrame):
        return spec
    try:
        return REPORT_FRAMES[str(spec).strip().lower().replace(" ", "")]
    except KeyError:
        raise ValueError(
            f"unknown reporting frame {spec!r}; "
            f"expected one of {sorted(REPORT_FRAMES)}") from None


def to_report_frame(v, frame) -> np.ndarray:
    """Rewrite a vector's components in `frame`. Magnitudes are unchanged.

    Args:
        v: array_like, shape (..., 3), in the internal ISO 8855 frame.
        frame: a `ReportFrame`, a key from `REPORT_FRAMES`, or `None`.
    """
    frame = get_report_frame(frame)
    v = np.asarray(v, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"expected last axis of length 3, got shape {v.shape}")
    if frame.is_internal:
        return v.copy()
    return v @ frame.matrix.T


# ---------------------------------------------------------------------------
# Tire slip quantities -- OUR definitions
# ---------------------------------------------------------------------------

def slip_angle(v_x: float, v_y: float) -> float:
    """Slip angle in the internal (ISO) convention, radians.

    Defined so that POSITIVE slip angle produces POSITIVE lateral force, i.e.
    in the linear range::

        Fy = C_alpha * alpha        with cornering stiffness C_alpha > 0

    Args:
        v_x: longitudinal velocity of the contact patch, in the wheel frame.
        v_y: lateral velocity of the contact patch, in the wheel frame
             (+y = left).

    Physical check (pinned by test_slip_angle_sign_left_turn):
        A wheel whose velocity vector points to the RIGHT of the wheel plane
        (v_y < 0) has POSITIVE slip angle, and generates a lateral force to the
        LEFT (+y). That is the outside front tire of a car turning left, which
        is the case worth being able to reason about without thinking.

    Note the abs() on v_x: slip angle is measured relative to the wheel plane,
    so it must not flip sign when the car reverses.
    """
    return float(np.arctan2(-v_y, abs(v_x)))


def slip_ratio(v_x: float, omega: float, r_effective: float) -> float:
    """Longitudinal slip ratio, dimensionless.

        kappa = (omega * r_e - v_x) / |v_x|

    Positive under drive (wheel spinning faster than the ground speed),
    negative under braking, so that Fx and kappa share a sign.

    Args:
        v_x: longitudinal velocity of the wheel centre.
        omega: wheel angular velocity (positive = forward rolling).
        r_effective: effective rolling radius.

    Guarded at low speed: below `_V_EPS` the ratio is not physically meaningful
    and would blow up. The transient model needs a proper low-speed formulation
    (relaxation-length based) before it can handle standing starts -- see
    docs/07. Until then this returns 0.0 and the caller must not rely on it
    near zero speed.
    """
    if abs(v_x) < _V_EPS:
        return 0.0
    return float((omega * r_effective - v_x) / abs(v_x))


_V_EPS = 0.1  # m/s -- below this, slip ratio is not meaningful

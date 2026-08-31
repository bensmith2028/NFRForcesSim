"""Pacejka Magic Formula -- pure-slip lateral force.

MF 5.2 (a.k.a. Pacejka 2002) pure-slip lateral. Implemented in SI and the
internal ISO 8855 convention, so that:

    positive slip angle  ->  positive Fy      (leftward, +y)
    positive Fz          ->  loaded tire

CITATION NOTE
-------------
These equations are NOT from RCVD. RCVD (Milliken & Milliken, 1995) predates
MF 5.2 and covers the earlier Pacejka '89/'94 form in Ch. 14; the coefficient
naming and the load/camber dependence used here come from:

    Pacejka, "Tire and Vehicle Dynamics", 3rd ed., Ch. 4 (the "Magic Formula
    Tyre Model"), pure-slip lateral equations (4.E19)-(4.E30).

Where a formula in this project DOES come from RCVD it is cited inline as
`RCVD Eq. x.y, p. N` with the frame conversion noted. This module is flagged
explicitly because the difference matters: MF5.2 has camber and load-dependent
peak, curvature and shift terms that the RCVD-era form does not.

WHY MF5.2 AND NOT THE PREVIOUS TEAM MODEL
-----------------------------------------
The NFR24 script fitted `Fy = D*sin(C*atan(B*alpha))` with `D = (p1 + p2/1000*Fz)*Fz`.
That has load sensitivity (good) but no E term, no camber dependence, and no
shifts. Camber is a required input for goal G5, so that model cannot answer the
question it was being asked. See docs/02 SS3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

import numpy as np

__all__ = ["MF52Lateral", "LATERAL_PARAM_NAMES"]


# Order matters: it defines the packing used by the fitter.
LATERAL_PARAM_NAMES = (
    "pCy1",                                  # shape
    "pDy1", "pDy2", "pDy3",                  # peak: nominal, load, camber
    "pEy1", "pEy2", "pEy3", "pEy4",          # curvature
    "pKy1", "pKy2", "pKy3",                  # cornering stiffness
    "pHy1", "pHy2", "pHy3",                  # horizontal shift
    "pVy1", "pVy2", "pVy3", "pVy4",          # vertical shift
)


@dataclass
class MF52Lateral:
    """MF5.2 pure-slip lateral force model.

    Args:
        Fz0: nominal load used to non-dimensionalise, N.
        lambda_muy: friction scaling. This is the belt-to-asphalt correlation
            factor -- a CALIBRATION parameter, not a fitted one. Fitting is
            done at lambda_muy = 1.0 against raw belt data; the scaling is
            applied afterwards from vehicle-level g-g data. See docs/02 SS5.1.
    """

    Fz0: float = 1100.0
    lambda_muy: float = 1.0

    # Defaults are a PHYSICALLY SENSIBLE FSAE-shaped starting point, not zeros.
    # They matter: they seed the fitter, and a seed that puts the force peak at
    # an impossible slip angle can settle the optimiser in a bad basin.
    #
    # Calibration of pKy1: peak slip angle is set by B = K/(C*D). For a peak
    # near 6 deg at Fz=700 N with D ~ 1750 N and C = 1.35 we need B ~ 17.5,
    # i.e. K ~ 41 kN/rad (~730 N/deg), which gives pKy1 ~ 55. An earlier value
    # of 20 put the peak at 15.6 deg -- nothing like a real tire.
    pCy1: float = 1.35
    pDy1: float = 2.4
    pDy2: float = -0.3
    pDy3: float = 0.0
    pEy1: float = -0.5
    pEy2: float = 0.0
    pEy3: float = 0.0
    pEy4: float = 0.0
    pKy1: float = 55.0
    pKy2: float = 1.6
    pKy3: float = 0.5
    pHy1: float = 0.0
    pHy2: float = 0.0
    pHy3: float = 0.0
    pVy1: float = 0.0
    pVy2: float = 0.0
    pVy3: float = 0.0
    pVy4: float = 0.0

    # ------------------------------------------------------------------
    def fy(self, alpha, Fz, gamma=0.0):
        """Lateral force, N.

        Args:
            alpha: slip angle, rad (ISO sign -- positive alpha gives positive Fy).
            Fz: vertical load, N, positive for a loaded tire.
            gamma: inclination angle, rad.

        Returns:
            Fy in newtons, same shape as the broadcast inputs.
        """
        alpha = np.asarray(alpha, dtype=float)
        Fz = np.asarray(Fz, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        # A tire off the ground carries no lateral force. Without this clamp the
        # model happily returns force at Fz <= 0, which shows up as phantom grip
        # during wheel lift in the transient model.
        Fz_safe = np.maximum(Fz, 0.0)
        dfz = (Fz_safe - self.Fz0) / self.Fz0

        Cy = self.pCy1
        mu_y = (self.pDy1 + self.pDy2 * dfz) * (1.0 - self.pDy3 * gamma**2) * self.lambda_muy
        Dy = mu_y * Fz_safe

        SHy = (self.pHy1 + self.pHy2 * dfz) + self.pHy3 * gamma
        SVy = Fz_safe * ((self.pVy1 + self.pVy2 * dfz)
                         + (self.pVy3 + self.pVy4 * dfz) * gamma) * self.lambda_muy

        alpha_y = alpha + SHy

        Ey = ((self.pEy1 + self.pEy2 * dfz)
              * (1.0 - (self.pEy3 + self.pEy4 * gamma) * np.sign(alpha_y)))
        # E > 1 makes the curve non-physical (the arctan argument turns over).
        Ey = np.minimum(Ey, 1.0)

        Ky = (self.pKy1 * self.Fz0
              * np.sin(2.0 * np.arctan2(Fz_safe, self.pKy2 * self.Fz0))
              * (1.0 - self.pKy3 * np.abs(gamma)))

        By = Ky / np.maximum(Cy * Dy, 1e-9)

        arg = By * alpha_y
        fy = Dy * np.sin(Cy * np.arctan(arg - Ey * (arg - np.arctan(arg)))) + SVy
        return np.where(Fz_safe > 0.0, fy, 0.0)

    # ------------------------------------------------------------------
    def cornering_stiffness(self, Fz, gamma=0.0):
        """dFy/dalpha at alpha = 0, N/rad. Always positive in ISO."""
        Fz_safe = np.maximum(np.asarray(Fz, dtype=float), 0.0)
        return (self.pKy1 * self.Fz0
                * np.sin(2.0 * np.arctan2(Fz_safe, self.pKy2 * self.Fz0))
                * (1.0 - self.pKy3 * np.abs(gamma)))

    def peak_mu(self, Fz, gamma=0.0):
        """Peak lateral friction coefficient at this load and camber."""
        Fz_safe = np.maximum(np.asarray(Fz, dtype=float), 0.0)
        dfz = (Fz_safe - self.Fz0) / self.Fz0
        return ((self.pDy1 + self.pDy2 * dfz)
                * (1.0 - self.pDy3 * np.asarray(gamma, dtype=float) ** 2)
                * self.lambda_muy)

    # ------------------------------------------------------------------
    def get_params(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in LATERAL_PARAM_NAMES], dtype=float)

    def set_params(self, values) -> "MF52Lateral":
        for name, v in zip(LATERAL_PARAM_NAMES, np.asarray(values, dtype=float)):
            setattr(self, name, float(v))
        return self

    def without_conicity_and_plysteer(self) -> "MF52Lateral":
        """Copy with the non-camber horizontal and vertical shifts zeroed.

        SHy and SVy absorb conicity and plysteer, which are real but are
        measured on ONE tire in ONE orientation. Applying them identically to
        both sides of a car makes the whole vehicle pull, and makes mirrored
        operating points disagree -- so symmetric vehicle work should normally
        zero them. Keep them only when deliberately studying pull.

        The camber-dependent shift terms (pHy3, pVy3, pVy4) are retained: those
        are physical camber thrust and mirror correctly once per-side camber
        signs are right (see solvers.steady_state.symmetric_camber).
        """
        import copy as _copy
        m = _copy.copy(self)
        m.pHy1 = m.pHy2 = 0.0
        m.pVy1 = m.pVy2 = 0.0
        return m

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MF52Lateral":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

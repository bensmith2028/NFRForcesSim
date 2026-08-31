"""Pure-slip aligning moment `Mz`, fitted to TTC data.

WHY THIS EXISTS
---------------
`CornerLoads` has carried an `Mz` field since the force extractor was written,
and `extract_components` plumbs it correctly into the upright's moment balance.
Nothing ever set it. Every structural case in this repo was therefore solved
with zero aligning moment, and `docs/04` item 2 names Mz as precisely the load
that **loads the tie rod**.

The cost of that was not small. On the front FR corner in `outside_cornering`
(Fy = 1899 N), the reported tie-rod load moves with Mz like this:

    Mz = 0 N.m   ->   tie rod   455 N     <- what the repo used to report
    Mz = -40     ->             1092 N
    Mz = -60     ->             1410 N
    Mz = -80     ->             1728 N

Pushrod and ball-joint loads barely move (<2%), so this is specifically a tie
rod, steering arm and rack finding. The old NFR24 MATLAB model fitted an MZ
curve and fed it into its corner free body; this repo regressed on that.

WHAT THE DATA SAYS, AND WHY THE PEAK IS NOT AT THE LIMIT
--------------------------------------------------------
Pneumatic trail COLLAPSES as the tire saturates. Measured on this tire
(Round 9, near-zero camber, Fz 900-1300 N), t = -Mz/Fy:

    |alpha| deg    0.8   1.5   2.5   3.5   4.5   5.5   6.5   7.5   9.0  11.0
    trail mm      46.6  40.4  34.5  29.1  26.2  20.7  16.6  12.8   8.8   3.7

So |Mz| peaks at roughly 3-5 deg of slip -- WELL BEFORE the lateral limit --
and is nearly gone by the time the tire is saturated. A structural case
evaluated only at the lateral limit would still under-predict the tie rod. That
is why `peak_mz` exists and why `corner_loads_for_case` uses it: the sizing
number is the peak over slip angle, not the value at maximum a_y.

FIT QUALITY -- LOWER THAN THE LATERAL FIT, AND SAID PLAINLY
------------------------------------------------------------
Fitted to the same runs and the same conditioning as the lateral model
(Round 9 runs 4 and 6 at the 82.7 kPa set point; run 5 is rejected because only
32 samples survive the pressure band). Overall R^2 = 0.839, RMSE = 10.5 N.m
against a peak of 50-80 N.m -- nothing like the lateral fit's 0.998. MZ is a far
noisier channel. Broken down, over 202679 samples:

    Fz 100- 400 N    R^2 0.29   RMSE 10.6     <- signal is small here; R^2 misleads
    Fz 400- 700 N    R^2 0.79   RMSE  7.8
    Fz 700-1000 N    R^2 0.89   RMSE  8.9     <- the loaded outside corner
    Fz 1000-1700 N   R^2 0.86   RMSE 13.1     <- and here
    gamma ~ 0        R^2 0.84   RMSE  9.7     <- so it is not mainly a camber failure

Take absolute Mz as +/-20% in the load band that matters. The tie-rod
conclusion (455 N -> ~1.4 kN) is robust to that: it is a 3x change against a
20% uncertainty.

KNOWN WEAKNESS -- THE HIGH-SLIP TAIL
-------------------------------------
Above each load's own peak this fit is not trustworthy, in two ways: it
under-predicts the tail at low load, and the `Ez <= 1` clamp is one-sided so
the two slip signs diverge. At Fz = 900 N, where the peak sits at 3.15 deg, the
branches are 0.1% apart at the peak, 3% at 5 deg, 7% at 6 deg and 39% at
10 deg; at Fz = 400 N the peak is at 1.07 deg and the divergence sets in
correspondingly earlier.

Neither reaches a structural number, because `peak_mz` sweeps BOTH branches and
reads the peak, where the curve is odd to 0.1% at every load. But `mz()` on its
own past the peak should not be used to study anything without refitting.
Pinned by test_high_slip_tail_is_asymmetric_and_that_is_documented_not_fixed,
so it stays a known limitation rather than turning into a surprise.

ONE COEFFICIENT IS BOUNDED FOR A PHYSICAL REASON, NOT A NUMERICAL ONE
----------------------------------------------------------------------
`qEz4` scales the Magic Formula's slip-SIGN asymmetry in the curvature term.
Left free it pins at -20 and makes the curve 20% lopsided between +10 and
-10 deg -- which on a symmetric car is a left/right asymmetry, the same class
of error `MF52Lateral.without_conicity_and_plysteer` and
`solvers.steady_state.symmetric_camber` both exist to prevent. Constrained to
|qEz4| <= 1 the optimiser settles at 0.234, comfortably inside the bound, and
R^2 is very slightly BETTER (0.8392 against 0.8388). Nothing was traded for the
symmetry; the -20 was the optimiser abusing a shape parameter as a fudge
factor. Any refit must keep that bound.

SIGN CONVENTION
---------------
ISO 8855, consistent with `MF52Lateral`: positive slip angle gives positive Fy
and NEGATIVE Mz. The moment is a restoring one -- Fy acts a pneumatic trail
BEHIND the contact centre, so `M = (-t*x_hat) x (Fy*y_hat) = -t*Fy*z_hat`.

NOT MODELLED
------------
Pressure (single set point, like the lateral fit), combined slip (Mz falls
sharply with longitudinal force -- so using this alongside a braking Fx is
conservative), and Mzr's residual-spin term. Turn-slip is out of scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["MF52Aligning", "ALIGNING_PARAM_NAMES", "ALIGNING_PARAM_BOUNDS",
           "AligningFitResult", "fit_aligning"]


# Order matters: it defines the packing used by any refit.
ALIGNING_PARAM_NAMES = (
    "qBz1", "qBz2", "qBz3", "qBz4",          # stiffness: nominal, load, load^2, camber
    "qCz1",                                   # shape
    "qDz1", "qDz2", "qDz3", "qDz4",          # peak: nominal, load, camber, camber^2
    "qEz1", "qEz2", "qEz4",                  # curvature, incl. slip-sign asymmetry
    "qHz1", "qHz2",                          # horizontal shift
    "qVz1", "qVz2",                          # vertical shift
)


@dataclass
class MF52Aligning:
    """MF5.2-shaped pure-slip aligning moment.

    Defaults are the FITTED coefficients for the tire this car runs (Hoosier
    16.0x7.5-10 R20 C2000 on a 7.0 in rim, 82.7 kPa), not placeholders -- see
    the module docstring for the fit and its limits. They are carried as
    dataclass defaults rather than loaded from a config path so that populating
    Mz needs no config plumbing; `from_dict` reads the same JSON layout as
    `MF52Lateral` if a fit file is preferred later.

    Args:
        Fz0: nominal load used to non-dimensionalise, N. Matches the lateral
            fit's value so `dfz` means the same thing in both models.
        R0: unloaded radius, m. Sets the length scale of the trail.
        lambda_mz: friction scaling, the Mz analogue of `MF52Lateral.lambda_muy`.

            >>> DEFAULT 1.0 IS THE RAW BELT VALUE, AND THAT IS DELIBERATE. <<<

            Mz scales with the lateral force that generates it, so on track
            this should track `lambda_muy` (0.62 on this car). Leaving it at
            1.0 therefore OVER-states Mz by roughly 1/0.62, which is
            conservative for sizing a tie rod and is the safe default for a
            structural path that may not know the vehicle's calibration. Pass
            `lambda_mz=cfg tires.scaling.mu_y` when you want the on-track value.
    """

    Fz0: float = 672.4821437950824
    R0: float = 0.2032
    lambda_mz: float = 1.0

    qBz1: float = 20.870615040226703
    qBz2: float = -34.04339101339857
    qBz3: float = 25.76199281460853
    qBz4: float = -0.4775264600285215
    qCz1: float = 2.7706004228264502
    qDz1: float = -0.20089368954603004
    qDz2: float = -0.06073250399026236
    qDz3: float = 0.016121549525831908
    qDz4: float = -12.597042886320017
    qEz1: float = 0.7418790747154079
    qEz2: float = -1.007870403427878
    qEz4: float = 0.23354843381579413
    qHz1: float = 0.0008116117950383983
    qHz2: float = -0.10382218493906226
    qVz1: float = 0.00922415181654249
    qVz2: float = 2.3228006304460465

    # ------------------------------------------------------------------
    def mz(self, alpha, Fz, gamma=0.0):
        """Aligning moment, N.m.

        Args:
            alpha: slip angle, rad (ISO -- positive alpha gives positive Fy and
                negative Mz).
            Fz: vertical load, N, positive for a loaded tire.
            gamma: inclination angle, rad.

        Returns:
            Mz in newton-metres, broadcast over the inputs.
        """
        alpha = np.asarray(alpha, dtype=float)
        Fz = np.asarray(Fz, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        # A tire off the ground carries no aligning moment either. Same clamp,
        # and the same reason, as MF52Lateral.fy -- phantom moment during wheel
        # lift would feed straight into a tie-rod load.
        Fz_safe = np.maximum(Fz, 0.0)
        dfz = (Fz_safe - self.Fz0) / self.Fz0

        Dz = (Fz_safe * self.R0 * (self.qDz1 + self.qDz2 * dfz)
              * (1.0 + self.qDz3 * np.abs(gamma) + self.qDz4 * gamma ** 2))
        Bz = ((self.qBz1 + self.qBz2 * dfz + self.qBz3 * dfz * dfz)
              * (1.0 + self.qBz4 * np.abs(gamma)))
        Cz = self.qCz1

        alpha_z = alpha + (self.qHz1 + self.qHz2 * gamma)
        arg = Bz * alpha_z

        # E > 1 makes the arctan argument turn over, exactly as in the lateral
        # model. Clamp for the same reason.
        Ez = np.minimum(
            (self.qEz1 + self.qEz2 * dfz)
            * (1.0 + self.qEz4 * (2.0 / np.pi) * np.arctan(Bz * Cz * alpha_z)),
            1.0)

        SVz = Fz_safe * self.R0 * (self.qVz1 + self.qVz2 * gamma)

        mz = Dz * np.sin(Cz * np.arctan(arg - Ez * (arg - np.arctan(arg)))) + SVz
        return np.where(Fz_safe > 0.0, mz * self.lambda_mz, 0.0)

    # ------------------------------------------------------------------
    def pneumatic_trail(self, alpha, Fz, gamma=0.0, fy=None):
        """t = -Mz/Fy, m. Diagnostic -- this is what the docstring table shows.

        `fy` must come from the lateral model (or from data). Returns nan where
        Fy is too small for the ratio to mean anything, rather than a spike.
        """
        if fy is None:
            raise ValueError("pneumatic_trail needs Fy; pass the lateral model's")
        fy = np.asarray(fy, dtype=float)
        mz = self.mz(alpha, Fz, gamma)
        return np.where(np.abs(fy) > 1.0, -mz / np.where(np.abs(fy) > 1.0, fy, 1.0),
                        np.nan)

    # ------------------------------------------------------------------
    def peak_mz(self, Fz, gamma=0.0, max_abs_alpha_rad: float = np.radians(15.0),
                n: int = 601) -> np.ndarray:
        """Largest |Mz| over slip angle, returned signed at the worst point.

        >>> THIS, NOT `mz` AT THE LIMIT, IS THE STRUCTURAL NUMBER. <<<

        Pneumatic trail collapses as the tire saturates (see the module
        docstring's measured table), so |Mz| peaks near 3-5 deg of slip and is
        small again at maximum lateral acceleration. A tie rod sized from the
        limit state alone would be sized from almost the only part of the sweep
        where the aligning moment is NOT significant.

        SWEEPS BOTH SIGNS OF SLIP ANGLE, which is not redundant even after
        `without_conicity_and_plysteer`. The Magic Formula's `qEz4` term makes
        the curvature depend on the SIGN of slip -- that asymmetry is part of
        the model, not a fitting artefact -- so the two branches differ by a few
        tenths of a percent here and the positive branch is the smaller one.
        Sweeping one branch would quietly return the lesser of the two; it is
        negligible on this fit but free to get right, and it is not negligible
        on a fit that still carries conicity (12% on the raw coefficients).

        Callers that know the direction of the corner should apply their own
        sign -- `corner_loads_for_case` signs it against Fy.

        Swept rather than solved: the curve is cheap, this is called four times
        per load case (not in any inner loop), and a grid cannot land on the
        wrong branch the way a Newton solve can. `n` points over +/-15 deg
        resolves the peak to 0.05 deg, far finer than the fit's own accuracy.
        """
        grid = np.linspace(-max_abs_alpha_rad, max_abs_alpha_rad, 2 * n - 1)
        # Broadcast the two inputs against each other FIRST -- a scalar gamma
        # against an array Fz has size 1 and cannot be reshaped to the common
        # shape on its own. Then (..., 1) against (n,) puts the sweep on the
        # last axis.
        Fz_b, gamma_b = np.broadcast_arrays(np.asarray(Fz, dtype=float),
                                            np.asarray(gamma, dtype=float))
        vals = self.mz(grid, Fz_b[..., None], gamma_b[..., None])
        idx = np.argmax(np.abs(vals), axis=-1)
        return np.take_along_axis(vals, np.expand_dims(idx, -1), axis=-1)[..., 0]

    def peak_mz_slip(self, Fz, gamma=0.0, max_abs_alpha_rad: float = np.radians(15.0),
                     n: int = 601) -> np.ndarray:
        """Slip angle at which `peak_mz` occurs, rad. Reporting aid.

        Same two-sided sweep as `peak_mz`, so the two always agree about which
        point they are describing.
        """
        grid = np.linspace(-max_abs_alpha_rad, max_abs_alpha_rad, 2 * n - 1)
        Fz_b, gamma_b = np.broadcast_arrays(np.asarray(Fz, dtype=float),
                                            np.asarray(gamma, dtype=float))
        vals = self.mz(grid, Fz_b[..., None], gamma_b[..., None])
        return grid[np.argmax(np.abs(vals), axis=-1)]

    # ------------------------------------------------------------------
    def without_conicity_and_plysteer(self) -> "MF52Aligning":
        """Copy with the non-camber horizontal and vertical shifts zeroed.

        Exactly the policy `MF52Lateral.without_conicity_and_plysteer` applies,
        and for exactly the same reason -- but it matters MORE here, because
        `peak_mz` sweeps only positive slip angle and would silently pick one
        branch of an asymmetric curve.

        On this fit the raw shifts (qHz1 = 0.0018 rad, qVz1 = 0.0154) make the
        curve lean: at Fz = 1200 N it gives -54.8 N.m at +3 deg against
        +61.5 N.m at -3 deg, a 12% split. Sweeping the positive branch alone
        would take the SMALLER of the two and under-report the tie rod by that
        much, while a mirrored pair of corners would disagree with each other.

        The camber-dependent shifts (qHz2, qVz2) are retained: those are
        physical camber effects and mirror correctly once per-side camber signs
        are right, same as the lateral model's pHy3/pVy3/pVy4.
        """
        import copy as _copy
        m = _copy.copy(self)
        m.qHz1 = 0.0
        m.qVz1 = 0.0
        return m

    # ------------------------------------------------------------------
    def get_params(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in ALIGNING_PARAM_NAMES], dtype=float)

    def set_params(self, values) -> "MF52Aligning":
        for name, v in zip(ALIGNING_PARAM_NAMES, np.asarray(values, dtype=float)):
            setattr(self, name, float(v))
        return self

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MF52Aligning":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Refitting
# ---------------------------------------------------------------------------
#
# `data/` is GITIGNORED -- the raw TTC drop is NDA'd and large -- so the fit
# artifact that produced the defaults above is not in the repository and cannot
# be. The coefficients are therefore carried as dataclass defaults, and this is
# the routine that regenerates them, so the fit stays reproducible by anyone
# holding the data rather than being a set of magic numbers.

ALIGNING_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "qBz1": (0.5, 80.0), "qBz2": (-50.0, 50.0), "qBz3": (-50.0, 50.0),
    "qBz4": (-20.0, 20.0),
    "qCz1": (0.3, 4.0),
    "qDz1": (-5.0, 0.0), "qDz2": (-5.0, 5.0),
    "qDz3": (-20.0, 20.0), "qDz4": (-20.0, 20.0),
    "qEz1": (-20.0, 1.0), "qEz2": (-20.0, 20.0),
    # >>> KEEP THIS BOUND. <<< See the module docstring: unbounded, qEz4 pins at
    # -20 and turns a symmetric-car model into a lopsided one, for no gain in
    # R^2 at all.
    "qEz4": (-1.0, 1.0),
    "qHz1": (-0.2, 0.2), "qHz2": (-2.0, 2.0),
    "qVz1": (-2.0, 2.0), "qVz2": (-5.0, 5.0),
}


@dataclass
class AligningFitResult:
    model: MF52Aligning
    rmse: float
    r_squared: float
    n_samples: int
    converged: bool
    message: str = ""
    r_squared_by_load_band: Dict[str, float] = field(default_factory=dict)

    def report(self) -> str:
        lines = [f"Mz fit: RMSE {self.rmse:.2f} N.m, R^2 {self.r_squared:.4f}, "
                 f"n = {self.n_samples}",
                 f"  converged: {self.converged} ({self.message})"]
        for band, r2 in self.r_squared_by_load_band.items():
            lines.append(f"  Fz {band:>12}   R^2 {r2:.4f}")
        return "\n".join(lines)


def fit_aligning(alpha, Fz, gamma, Mz, *,
                 Fz0: float = 672.4821437950824, R0: float = 0.2032,
                 fz_band_N: Tuple[float, float] = (400.0, 1400.0),
                 out_of_band_weight: float = 0.3,
                 initial: Optional[MF52Aligning] = None,
                 max_nfev: int = 20000) -> AligningFitResult:
    """Fit `MF52Aligning` to conditioned samples.

    Takes plain arrays rather than a `ConditionedData`, because that container
    carries Fy and not Mz. Condition the runs however the lateral fit does, then
    pass the surviving channels straight in.

    Args:
        alpha, gamma: rad. Fz: N. Mz: N.m. All ISO 8855, all same length.
        fz_band_N: the load range the car actually runs. Samples outside it are
            down-weighted rather than dropped, matching `fit.fit_lateral`'s
            policy -- see its docstring for why weighting beats truncation.
        initial: starting point. Defaults to the shipped coefficients, which is
            a good seed for a refit on a similar tire and a poor one for a very
            different one; pass `MF52Aligning()` explicitly to be sure.

    Returns:
        `AligningFitResult`. CHECK `converged` -- a stage that stopped on
        `max_nfev` still returns coefficients and can still report a decent
        R^2, exactly as `fit.fit_lateral` warns.
    """
    from scipy.optimize import least_squares

    alpha, Fz = np.asarray(alpha, float), np.asarray(Fz, float)
    gamma, Mz = np.asarray(gamma, float), np.asarray(Mz, float)
    if not (len(alpha) == len(Fz) == len(gamma) == len(Mz)):
        raise ValueError("alpha, Fz, gamma and Mz must be the same length")
    if len(alpha) == 0:
        raise ValueError("no samples to fit")

    seed = MF52Aligning() if initial is None else initial
    lo = np.array([ALIGNING_PARAM_BOUNDS[n][0] for n in ALIGNING_PARAM_NAMES])
    hi = np.array([ALIGNING_PARAM_BOUNDS[n][1] for n in ALIGNING_PARAM_NAMES])
    x0 = np.clip(seed.get_params(), lo, hi)

    band_lo, band_hi = fz_band_N
    w = np.where((Fz > band_lo) & (Fz < band_hi), 1.0, out_of_band_weight)

    def build(p) -> MF52Aligning:
        m = MF52Aligning(Fz0=Fz0, R0=R0, lambda_mz=1.0)
        return m.set_params(p)

    def resid(p):
        return (build(p).mz(alpha, Fz, gamma) - Mz) * w

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                        xtol=1e-12, ftol=1e-12, max_nfev=max_nfev)
    model = build(sol.x)
    pred = model.mz(alpha, Fz, gamma)
    ss_res = float(np.sum((pred - Mz) ** 2))
    ss_tot = float(np.sum((Mz - Mz.mean()) ** 2))

    by_band = {}
    for zlo, zhi in ((100, 400), (400, 700), (700, 1000), (1000, 1700)):
        m = (Fz > zlo) & (Fz < zhi)
        if m.sum() < 500:
            continue
        a, b = Mz[m], pred[m]
        denom = float(np.sum((a - a.mean()) ** 2))
        if denom > 0:
            by_band[f"{zlo}-{zhi} N"] = float(1.0 - np.sum((b - a) ** 2) / denom)

    return AligningFitResult(
        model=model,
        rmse=float(np.sqrt(np.mean((pred - Mz) ** 2))),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n_samples=int(len(alpha)),
        converged=sol.status > 0,
        message=str(sol.message),
        r_squared_by_load_band=by_band,
    )

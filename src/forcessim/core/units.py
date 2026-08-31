"""Unit handling.

THE INTERNAL UNIT SYSTEM IS SI. Always. m, kg, s, N, N*m, rad, Pa, K.

Conversions live here and are applied at I/O boundaries ONLY -- config parsing,
data import, report generation. Never mid-calculation.

Why this module exists at all, given that "just use SI" is a one-line rule:

  * RCVD and most of the classic vehicle-dynamics literature is in mixed
    imperial units (lbf, in, lb-ft, mph). Every formula lifted from it needs a
    documented conversion, and having the factors in one audited place is
    better than scattering 0.4536 through the codebase.
  * TTC data ships in two unit variants. Misdetecting that variant produces a
    car with ~4.4x the real grip, and the result still looks plausible on a
    plot. `assert_plausible_*` below exists to catch exactly that class of bug
    loudly rather than silently.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "DEG", "RAD",
    "deg2rad", "rad2deg",
    "IN_TO_M", "FT_TO_M", "MM_TO_M",
    "LBF_TO_N", "LBM_TO_KG", "LBFT_TO_NM",
    "MPH_TO_MPS", "KPH_TO_MPS",
    "PSI_TO_PA", "KPA_TO_PA",
    "G",
    "assert_plausible_tire_force",
    "assert_plausible_tire_force_dataset",
    "assert_plausible_length",
    "assert_plausible_geometry_dataset",
]

# --- fundamental -----------------------------------------------------------
G = 9.80665  # m/s^2, standard gravity

# --- angle -----------------------------------------------------------------
DEG = math.pi / 180.0  # multiply degrees by this to get radians
RAD = 180.0 / math.pi  # multiply radians by this to get degrees


def deg2rad(x):
    return x * DEG


def rad2deg(x):
    return x * RAD


# --- length ----------------------------------------------------------------
IN_TO_M = 0.0254
FT_TO_M = 0.3048
MM_TO_M = 1.0e-3

# --- force / mass / moment -------------------------------------------------
LBF_TO_N = 4.4482216152605
LBM_TO_KG = 0.45359237
LBFT_TO_NM = 1.3558179483314004

# --- velocity --------------------------------------------------------------
MPH_TO_MPS = 0.44704
KPH_TO_MPS = 1.0 / 3.6

# --- pressure --------------------------------------------------------------
PSI_TO_PA = 6894.757293168361
KPA_TO_PA = 1.0e3


# ---------------------------------------------------------------------------
# Plausibility guards
# ---------------------------------------------------------------------------
#
# These are runtime assertions for data import, where a unit-system error is
# silent and expensive.
#
# IMPORTANT -- why these operate on datasets, not single values:
#
#   A single number cannot reveal a unit-system error, because any single value
#   is plausible in SOME unit system. A 900 N corner load misread from a lbf
#   file reads as 202 N -- which is a perfectly legitimate TTC light-load test
#   point. There is no per-value threshold that separates them.
#
#   What DOES separate them is the range spanned by a whole dataset. TTC
#   cornering files always sweep up to a high load set-point, so the MAXIMUM
#   load in a real file is order 900-1300 N. Read the same file as if lbf were
#   N and that maximum collapses to ~250. That is unambiguous.
#
# So: `assert_plausible_*` are cheap per-value sanity checks that catch only
# gross nonsense, and `assert_plausible_*_dataset` are the real unit-variant
# detectors. Import paths should call the dataset versions.

# CALIBRATED against our Round 9 cornering data (2026-07-28).
#
# Measured peak |Fz| across the 16.0x7.5 runs: 1189-1279 N, tightly clustered.
# The bounds below sit comfortably either side of that, wide enough not to
# reject a legitimate low-load run but narrow enough that a lbf/N mix-up
# (which would put the peak near 270 N, or near 5500 N if double-converted)
# is caught.
_TTC_PEAK_FZ_MIN_N = 600.0
_TTC_PEAK_FZ_MAX_N = 2500.0


def assert_plausible_tire_force(fz_newtons: float, context: str = "") -> None:
    """Cheap per-value sanity check. Catches gross nonsense only.

    This CANNOT detect a lbf/N mix-up -- use `assert_plausible_tire_force_dataset`
    for that. See the module note above.
    """
    if not (1.0 <= abs(fz_newtons) <= 20_000.0):
        raise ValueError(
            f"implausible vertical tire load {fz_newtons:.1f} N{_ctx(context)}. "
            f"Expected order 300-1600 N for FSAE."
        )


def assert_plausible_tire_force_dataset(fz_newtons, context: str = "") -> None:
    """Detect a lbf-vs-N unit error across a whole tire dataset.

    Args:
        fz_newtons: array_like of vertical loads from one test file, in the
            units we believe them to be (N).

    Raises:
        ValueError: if the peak load is inconsistent with a real TTC file read
            in newtons.
    """
    peak = float(np.max(np.abs(np.asarray(fz_newtons, dtype=float))))

    if peak < _TTC_PEAK_FZ_MIN_N:
        raise ValueError(
            f"peak vertical load in this dataset is only {peak:.0f} N{_ctx(context)}. "
            f"A real TTC file should peak above ~{_TTC_PEAK_FZ_MIN_N:.0f} N. "
            f"This strongly suggests the source data is in lbf and was read as N "
            f"(multiply by {LBF_TO_N:.4f})."
        )
    if peak > _TTC_PEAK_FZ_MAX_N:
        raise ValueError(
            f"peak vertical load in this dataset is {peak:.0f} N{_ctx(context)}, "
            f"above the ~{_TTC_PEAK_FZ_MAX_N:.0f} N expected for FSAE tire testing. "
            f"This suggests a value already in N was converted again as if it "
            f"were lbf."
        )


def assert_plausible_length(metres: float, context: str = "") -> None:
    """Cheap per-value sanity check on a geometry coordinate.

    Catches mm (350) and large inch values (13.8) read as metres. Does NOT
    catch small inch values -- use `assert_plausible_geometry_dataset`.
    """
    if abs(metres) > 5.0:
        raise ValueError(
            f"implausible length {metres:.3f} m{_ctx(context)}. "
            f"Hardpoints should be order 0.01-1.0 m. "
            f"Did the source file use mm or inches?"
        )


def assert_plausible_geometry_dataset(coords_metres, context: str = "") -> None:
    """Detect mm/inch/metre confusion across a whole hardpoint set.

    The largest coordinate magnitude in a full FSAE hardpoint set is order
    0.3-1.6 m (roughly half-track to wheelbase). In mm that becomes ~1600; in
    inches, ~60.
    """
    peak = float(np.max(np.abs(np.asarray(coords_metres, dtype=float))))

    if peak > 5.0:
        guess = "millimetres" if peak > 100.0 else "inches"
        raise ValueError(
            f"largest hardpoint coordinate is {peak:.2f} m{_ctx(context)}. "
            f"Expected order 0.3-1.6 m -- the source file is probably in {guess}."
        )
    if peak < 0.2:
        raise ValueError(
            f"largest hardpoint coordinate is only {peak:.4f} m{_ctx(context)}. "
            f"Expected order 0.3-1.6 m for a full corner. Is this a partial "
            f"hardpoint set, or already-scaled data?"
        )


def _ctx(context: str) -> str:
    return f" ({context})" if context else ""

"""YAML vehicle config -> the dataclasses the forces pipeline actually takes.

PORT NOTE (forcesSim, phase 1)
----------------------------------------
This is a TRIMMED port of `nfrsim.core.config` from NFRFullVehicleSim. That
original module is the config loader for the WHOLE vehicle sim -- T1 steady
state, T2 GGV/lap, tire fitting, dampers, powertrain torque curves, the lot --
and imports `solvers.steady_state`, `tire.longitudinal`, `tire.magic_formula`,
`vehicle.powertrain` and `vehicle.quarter_car` at module scope to build all of
that. None of those modules exist in this repo on purpose: this repo ports
only the validated forces/component-forces pipeline (see the package
docstring in `forcessim.suspension.forces`), and importing this module
must not drag the rest of that machinery back in.

So this file keeps exactly the functions that pipeline needs to go from a
YAML config to a `CornerGeometry` and a `VehicleParams`, verbatim from the
original where kept (same reasoning comments, same numbers, same edge cases),
and drops the rest:

    KEPT     load_config, get_path, set_path, resolve_config_path, ConfigError
             build_corner_geometry     (topology, actuation, hardpoints -> CornerGeometry)
             derive_kinematics / check_derived_against_config / _hardpoint_signature
             build_vehicle_params      (mass, dimensions, chassis, aero -> VehicleParams)
             build_aero_params, _roll_stiffness, _unsprung_cg_height, _opt
             build_brake_bias, build_drive_split, build_caliper_on_upright
                                       (which free body the brake/drive couple loads)
             build_tire                (ADDED phase 3 -- see note below)

    DROPPED  build_corner_setup, build_powertrain, build_full_vehicle,
             build_anti_geometry, FullVehicle -- all of these need a module
             this repo does not carry (`solvers.steady_state`,
             `vehicle.powertrain`). If a later phase needs anti-dive/anti-
             squat, port that deliberately then, not as a side effect of
             this file.

PHASE 3 ADDITION -- `build_tire`
---------------------------------
The original's `build_tire` returns a bare `MF52Lateral` and needs only
`tire.magic_formula` (now ported to `tire/magic_formula.py`, see that module's
and `tire/longitudinal.py`'s docstrings for what's fitted vs. invented in the
tire model). It does NOT need `tire.longitudinal.CombinedSlipTire` --
that wrapper exists in the original repo to add a *placeholder* slip-ratio
shape on top of the fitted lateral model for the GGV solver's combined-slip
need, which `gearbox/accel.py`'s straight-line, load-per-corner use of
`peak_mu_x` does not have (see `accel.py`'s docstring for exactly which part
of `CombinedSlipTire` it borrows and why). So `build_tire` is ported here
verbatim (same reasoning comments, same numbers, same edge cases -- only
the docstring's file-path example is unchanged since it already pointed at
the right file), but the `CombinedSlipTire`/`stub_combined_tire` wrapper and
`build_full_vehicle`'s `FullVehicle.tire: CombinedSlipTire` field are still
out of scope; `gearbox/accel.py` builds its own thin `CombinedSlipTire`
instance directly where it needs `peak_mu_x`, not through this loader.

PHASE 4 ADDITION -- `build_quarter_car`, `build_bump_envelope`
-----------------------------------------------------------------
The bump structural case needs `vehicle.quarter_car` (now ported to
`vehicle/quarter_car.py` -- a self-contained 2-DOF model that only needs
`core.units`, so importing it here drags in nothing else). Both builders are
ported verbatim from the original: `build_quarter_car` converts the config's
AT-THE-DAMPER rates to AT-THE-WHEEL using the motion ratio `derive_kinematics`
gets from the hardpoints, and `build_bump_envelope` sweeps the configured
speed list through it. `suspension/force_matrix.py` is what actually calls
these to build the bump case and the swept-bump envelope for the export
workbook.

Everything below this note is the original module's own docstring/code,
unmodified except for the deletions above.

----

Why this exists: nothing in the codebase read `configs/nfr26_baseline.yaml`
programmatically before this module. Every T1/T2 test built `VehicleParams`
by hand from numbers copied out of the file. That's fine for a fixed test
fixture; it is not fine for a sweep, which needs to load a base config,
perturb ONE field by a dotted path, and reconstruct -- doing that by editing
Python literals defeats the entire point of a config file.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It does not implement the `pydantic` schema docs/08 SS1 recommends. Every
other dataclass in this codebase (`VehicleParams`, `Track`, `GGV`, ...) is a
plain `@dataclass` with manual validation, and introducing a second
validation framework for just this one file would be inconsistent with
everything it feeds into for no real benefit -- the dotted-path override a
sweep needs is a few lines against a nested dict, not a reason to adopt a new
dependency.

TWO VALUES STILL COMPUTED OR OVERRIDABLE OUTSIDE THE RAW FIELDS
--------------------------------------------------------------------
1. **Kinematic outputs are DERIVED, not read.** Motion ratio, roll centre
   height and camber gain in roll all come from the hardpoint sheets via
   `derive_kinematics` (see `suspension/derived.py`), so moving a hardpoint
   moves the vehicle with no YAML field to remember. The corresponding YAML
   entries survive as ACCEPTANCE TARGETS and are cross-checked by
   `check_derived_against_config`; they are used as inputs only when the
   sheets are absent. An explicit `camber_gain_per_rad_roll_front/rear`
   argument still overrides everything (see `build_corner_setup` upstream;
   not ported here).
2. **Roll stiffness** (`roll_stiffness_front/rear`, N*m/rad). NOT a stored
   field -- it is DERIVED from spring rate, motion ratio, tire vertical rate
   and track width, exactly per the hand-calc already written into the
   config's `chassis:` comment block (springs in series with the tire, wheel
   rate scaled by MR^2, K_roll = 0.5*k_wheel*track^2). `_roll_stiffness`
   below reproduces that calc in code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml

from ..suspension.derived import DerivedKinematics, derive_both_axles
from ..suspension.geometry import (
    DEFAULT_ROD_PICKUP, DOUBLE_WISHBONE_PUSHROD, FIVE_LINK_PULLROD,
    ROD_PICKUPS, CornerGeometry, build_corner, load_hardpoints_csv,
)
from ..suspension.kinematics import _nominal_spindle_axis
from ..tire.magic_formula import MF52Lateral
from ..vehicle.loadtransfer import AeroParams, VehicleParams
from ..vehicle.quarter_car import QuarterCar, bump_envelope

__all__ = [
    "ConfigError", "load_config", "get_path", "set_path", "resolve_config_path",
    "build_aero_params", "build_vehicle_params",
    "build_corner_geometry", "derive_kinematics", "check_derived_against_config",
    "build_brake_bias", "build_drive_split", "build_caliper_on_upright",
    "build_tire", "build_combined_tire",
    "build_quarter_car", "build_bump_envelope",
]

_SOURCE_DIR_KEY = "__source_dir__"
"""Reserved config key holding the directory the YAML was loaded from, so a
relative path inside the file resolves against the config file's own
location rather than the process's current working directory -- a GUI or a
script invoked from anywhere must get the same car regardless of cwd."""


class ConfigError(ValueError):
    """A required field is missing, `null`, or otherwise unusable.

    Distinct from a plain ValueError so a caller can catch "this config point
    is incomplete" separately from "this config point is physically
    nonsensical" (VehicleParams' own validation).
    """


# ---------------------------------------------------------------------------
# Loading and dotted-path access
# ---------------------------------------------------------------------------

def load_config(path) -> Dict[str, Any]:
    """Parse a vehicle YAML into a plain nested dict. No validation here --
    validation happens at each `build_*` call, against what that call
    actually needs, so an incomplete config can still be loaded and inspected.

    Stashes the source directory under `_SOURCE_DIR_KEY` so relative paths
    inside the file (currently just `suspension.<axle>.hardpoints_file`)
    resolve correctly regardless of the caller's working directory.
    """
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    cfg[_SOURCE_DIR_KEY] = str(path.resolve().parent)
    return cfg


def resolve_config_path(cfg: Dict[str, Any], raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    base = cfg.get(_SOURCE_DIR_KEY)
    return (Path(base) / p) if base else p


def get_path(cfg: Dict[str, Any], dotted: str) -> Any:
    """Read a dotted path, e.g. `"mass.cg_height_m"`.

    Raises `ConfigError` (not `KeyError`) so a caller referencing a field
    that got renamed fails with a message pointing at the field, not an
    opaque KeyError three frames down inside dict traversal.
    """
    node = cfg
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(
                f"config has no field {dotted!r} "
                f"(failed at {'.'.join(parts[:i + 1])!r})")
        node = node[part]
    return node


def set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> Dict[str, Any]:
    """Return a NEW config with one dotted path overridden. Does not mutate
    `cfg` -- a sweep runs many points off one base config, and a mutating
    setter would make point N's override leak into point N+1 the moment two
    axes shared a parent dict, which is exactly the kind of bug that would
    silently corrupt a whole sweep rather than crash it."""
    out = copy.deepcopy(cfg)
    node = out
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"config has no field {dotted!r}")
        node = node[part]
    if parts[-1] not in node:
        raise ConfigError(f"config has no field {dotted!r}")
    node[parts[-1]] = value
    return out


# ---------------------------------------------------------------------------
# Derived quantities with a hand-calc to validate against
# ---------------------------------------------------------------------------

def _roll_stiffness(spring_rate_Npm: float, motion_ratio: float,
                    arb_wheelrate_Npm: float, tire_rate_Npm: float,
                    track_m: float) -> float:
    """N*m/rad at one axle. Reproduces the `chassis:` comment block in the
    YAML exactly -- see the module docstring, item 2."""
    k_wheel_spring = spring_rate_Npm * motion_ratio ** 2 + arb_wheelrate_Npm
    k_wheel_series = 1.0 / (1.0 / k_wheel_spring + 1.0 / tire_rate_Npm)
    return 0.5 * k_wheel_series * track_m ** 2


def _unsprung_cg_height(cfg: Dict[str, Any]) -> float:
    """Mass-weighted mean of front/rear loaded radius.

    APPROXIMATION, not a measurement: unsprung mass is dominated by the
    wheel/tire/upright assembly, and loaded radius is the only vertical
    reference this config carries for that assembly.
    """
    mf = get_path(cfg, "mass.unsprung_kg_per_corner.front")
    mr = get_path(cfg, "mass.unsprung_kg_per_corner.rear")
    rf = get_path(cfg, "wheels.loaded_radius_m.front")
    rr = get_path(cfg, "wheels.loaded_radius_m.rear")
    return (mf * rf + mr * rr) / (mf + mr)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_aero_params(cfg: Dict[str, Any]) -> AeroParams:
    a = cfg["aero"]
    return AeroParams(
        ClA=a["ClA"], CdA=a["CdA"], balance_front=a["balance_front"],
        cp_height_m=a["cp_height_m"], air_density=a["air_density_kg_m3"],
    )


def _opt(cfg: Dict[str, Any], dotted: str, default=None):
    """`get_path` that returns a default instead of raising for a missing key.

    Used only for genuinely optional sections. Everything a build actually
    REQUIRES still goes through `get_path` so it fails loudly.
    """
    try:
        return get_path(cfg, dotted)
    except ConfigError:
        return default


_TOPOLOGIES = {"double_wishbone": DOUBLE_WISHBONE_PUSHROD,
               "five_link": FIVE_LINK_PULLROD}
_ROD_POINTS = {"pushrod": ("IB_Push", "OB_Push"),
               "pullrod": ("IB_Pull", "OB_Pull")}


def build_corner_geometry(cfg: Dict[str, Any], axle: str) -> CornerGeometry:
    """One axle's CornerGeometry, straight from the hardpoint sheet.

    Everything comes from the config: topology, actuation, the sheet path, the
    `hardpoint_overrides` patch, the wheel centre and the static camber/toe
    that set the design spindle axis. Nothing is hand-typed here, so a hardpoint
    edit reaches the kinematics with no second place to remember.
    """
    sus = get_path(cfg, f"suspension.{axle}")
    topo = _TOPOLOGIES.get(sus["topology"])
    if topo is None:
        raise ConfigError(
            f"suspension.{axle}.topology = {sus['topology']!r} is not a known "
            f"topology (have {sorted(_TOPOLOGIES)}). Add it to "
            f"geometry.py's topology table, not here.")
    rod = _ROD_POINTS.get(sus["actuation"])
    if rod is None:
        raise ConfigError(
            f"suspension.{axle}.actuation = {sus['actuation']!r} is not known "
            f"(have {sorted(_ROD_POINTS)}).")

    # WHERE THE ROD PICKS UP IS A REQUIRED DECLARATION, AND IT IS VALIDATED.
    #
    # It selects the free-body idealization, not a coefficient inside one, so
    # an unrecognised value must not fall through to a default: a typo like
    # `upper-wishbone` silently choosing the upright construction moves the
    # pickup several mm at bump travel and propagates into every link force.
    pickup = sus.get("rod_pickup", DEFAULT_ROD_PICKUP)
    if pickup not in ROD_PICKUPS:
        raise ConfigError(
            f"suspension.{axle}.rod_pickup = {pickup!r} is not a known pickup "
            f"(have {sorted(ROD_PICKUPS)}). It decides which body carries the "
            f"rod, and therefore which free body the force extraction uses.")

    path = resolve_config_path(cfg, sus["hardpoints_file"])
    if not path.is_file():
        raise FileNotFoundError(path)

    # `hardpoint_overrides` is stored in METRES (`*_m` suffix); the loader
    # patches in the sheet's own millimetres. Convert here rather than storing
    # a bare mm number in a file whose header says "SI ONLY".
    overrides = {}
    for key, value in (_opt(cfg, f"hardpoint_overrides.{axle}") or {}).items():
        if not key.endswith("_m"):
            raise ConfigError(
                f"hardpoint_overrides.{axle}.{key} must end in `_m` -- this "
                f"config is SI throughout and the loader needs to know the "
                f"unit to convert from.")
        point, _, axis = key[:-2].rpartition("_")
        overrides[f"{point}.{axis}"] = float(value) * 1000.0

    pts = load_hardpoints_csv(path, overrides=overrides or None)
    corner = build_corner(axle.upper()[0] + "R", pts, topo, sus["actuation"], rod,
                          rod_pickup=pickup)
    corner.wheel_centre = np.array(get_path(cfg, f"wheels.centre_m.{axle}"), dtype=float)
    corner.spindle_axis = _nominal_spindle_axis(
        np.radians(sus["static_camber_deg"]), np.radians(sus["static_toe_deg"]),
        left=False)
    corner.loaded_radius = get_path(cfg, f"wheels.loaded_radius_m.{axle}")
    return corner


def _hardpoint_signature(cfg: Dict[str, Any]) -> Optional[tuple]:
    """Hashable identity of everything `derive_kinematics` reads.

    Includes the CONTENTS of both hardpoint sheets, so editing a sheet
    invalidates the cache -- the whole point of deriving rather than storing.
    """
    try:
        parts = []
        for axle in ("front", "rear"):
            sus = get_path(cfg, f"suspension.{axle}")
            path = resolve_config_path(cfg, sus["hardpoints_file"])
            if not path.is_file():
                return None
            parts.append((
                hashlib.sha256(path.read_bytes()).hexdigest(),
                sus["topology"], sus["actuation"],
                sus.get("rod_pickup", DEFAULT_ROD_PICKUP),
                sus["static_camber_deg"], sus["static_toe_deg"],
                tuple(get_path(cfg, f"wheels.centre_m.{axle}")),
                tuple(sorted((_opt(cfg, f"hardpoint_overrides.{axle}") or {}).items())),
            ))
        parts.append((get_path(cfg, "dimensions.track_front_m"),
                      get_path(cfg, "dimensions.track_rear_m")))
        return tuple(parts)
    except (KeyError, TypeError, ConfigError):
        return None


_DERIVED_CACHE: Dict[tuple, DerivedKinematics] = {}


def derive_kinematics(cfg: Dict[str, Any]) -> Optional[DerivedKinematics]:
    """Motion ratio, roll centre and camber gain, computed from the hardpoints.

    Returns None when the geometry sheets are not available -- they are
    checked in, but a stripped checkout or a synthetic config should degrade to
    the YAML values rather than fail to build a vehicle at all.

    Cached on the sheet CONTENTS (see `_hardpoint_signature`).
    """
    sig = _hardpoint_signature(cfg)
    if sig is None:
        return None
    if sig not in _DERIVED_CACHE:
        _DERIVED_CACHE[sig] = derive_both_axles(
            build_corner_geometry(cfg, "front"),
            build_corner_geometry(cfg, "rear"),
            get_path(cfg, "dimensions.track_front_m"),
            get_path(cfg, "dimensions.track_rear_m"))
    return _DERIVED_CACHE[sig]


# Fractional disagreement between a stored config value and the geometry-derived
# one that is treated as "the config is stale" rather than rounding.
_STALE_TOL = 0.02


def check_derived_against_config(cfg: Dict[str, Any]) -> Dict[str, tuple]:
    """Stored kinematic values that DISAGREE with the hardpoints.

    Returns {dotted path: (stored, derived, fractional error)}, empty when
    everything agrees. The stored numbers are acceptance targets now, not
    inputs.
    """
    derived = derive_kinematics(cfg)
    if derived is None:
        return {}
    bad = {}
    for path, value in derived.as_dict().items():
        stored = get_path(cfg, path)
        if stored is None:
            continue
        stored = float(stored)
        err = abs(stored - value) / max(abs(value), 1e-12)
        if err > _STALE_TOL:
            bad[path] = (stored, value, err)
    return bad


def build_vehicle_params(cfg: Dict[str, Any]) -> VehicleParams:
    """Everything the load-transfer / load-case pipeline needs, built from the YAML.

    Roll centre heights come from `kinematics_acceptance.*.roll_centre_height_m`
    -- these are the OptimumK/acceptance-test TARGETS, not (yet) an
    independently derived output -- only when the hardpoint sheets are absent;
    normally they come straight from `derive_kinematics`.
    """
    m = cfg["mass"]
    d = cfg["dimensions"]
    c = cfg["chassis"]

    # GEOMETRY FIRST. Roll centre height and motion ratio are OUTPUTS of the
    # hardpoints; the YAML entries are acceptance targets kept for cross-check.
    # Deriving them here is what stops a hardpoint edit from leaving a stale
    # ratio steering wheel rate -> roll stiffness -> LLTD. Falls back to the
    # stored values only when the sheets are absent.
    derived = derive_kinematics(cfg)
    if derived is None:
        roll_centre_f = get_path(cfg, "kinematics_acceptance.front.roll_centre_height_m")
        roll_centre_r = get_path(cfg, "kinematics_acceptance.rear.roll_centre_height_m")
        mr_f = get_path(cfg, "dampers.motion_ratio_at_ride.front")
        mr_r = get_path(cfg, "dampers.motion_ratio_at_ride.rear")
    else:
        roll_centre_f = derived.front.roll_centre_height_m
        roll_centre_r = derived.rear.roll_centre_height_m
        mr_f = derived.front.motion_ratio
        mr_r = derived.rear.motion_ratio

    tire_rate = get_path(cfg, "tires.vertical_rate_Npm")
    roll_k_f = _roll_stiffness(
        get_path(cfg, "suspension.front.spring_rate_Npm"), mr_f,
        get_path(cfg, "suspension.front.arb_wheelrate_Npm"),
        tire_rate, d["track_front_m"])
    roll_k_r = _roll_stiffness(
        get_path(cfg, "suspension.rear.spring_rate_Npm"), mr_r,
        get_path(cfg, "suspension.rear.arb_wheelrate_Npm"),
        tire_rate, d["track_rear_m"])

    return VehicleParams(
        mass=m["vehicle_kg"] + m["driver_kg"],
        cg_height=m["cg_height_m"],
        weight_dist_front=m["weight_dist_front"],
        wheelbase=d["wheelbase_m"],
        track_front=d["track_front_m"],
        track_rear=d["track_rear_m"],
        unsprung_front=m["unsprung_kg_per_corner"]["front"],
        unsprung_rear=m["unsprung_kg_per_corner"]["rear"],
        unsprung_cg_height=_unsprung_cg_height(cfg),
        roll_centre_front=roll_centre_f,
        roll_centre_rear=roll_centre_r,
        roll_stiffness_front=roll_k_f,
        roll_stiffness_rear=roll_k_r,
        chassis_torsional_stiffness=(
            c["torsional_stiffness_Nm_per_deg"] * 180.0 / 3.141592653589793),
        aero=build_aero_params(cfg),
    )


def build_brake_bias(cfg: Dict[str, Any]) -> Optional[float]:
    """Front brake bias from the config, or None if the car has no fixed bias.

    Exists so there is ONE place that answers "what brake bias does this
    config specify" -- a GUI reading the field directly and a solver reading
    it directly could otherwise disagree about which car they are modelling.
    """
    bias = get_path(cfg, "brakes.bias_front")
    if bias is None:
        return None
    bias = float(bias)
    if not 0.0 < bias < 1.0:
        raise ConfigError(
            f"brakes.bias_front = {bias} is out of range; it is the FRACTION "
            f"of brake force at the front axle and must lie strictly between "
            f"0 and 1.")
    return bias


#: Powertrain layout -> (front share of drive torque, {axle: couple reacts on
#: the upright}). Read straight from `powertrain.architecture`. A structural
#: load case needs neither the torque curve nor the diff -- only where the
#: torque enters the car and where its reaction lands.
_DRIVE_LAYOUTS: Dict[str, Tuple[float, Dict[str, bool]]] = {
    "awd_quad_hub": (0.5, {"front": True, "rear": True}),
    "rwd_hub":      (0.0, {"front": True, "rear": True}),
    "rwd_diff":     (0.0, {"front": False, "rear": False}),
    "fwd_diff":     (1.0, {"front": False, "rear": False}),
    # Front hub motors, rear motor through a diff: the flag genuinely differs
    # per axle, which is exactly why this returns a dict and not a bool.
    "awd_hybrid":   (0.5, {"front": True, "rear": False}),
}


def build_drive_split(cfg: Dict[str, Any]) -> Tuple[float, Dict[str, bool]]:
    """How drive torque is shared, and where its reaction lands, per axle.

    WHY THIS IS NOT A DETAIL
    ------------------------
    Both halves of the answer change LINK loads, not just wheel forces:

      * the front share decides whether the front corner carries any tractive
        force at all in the acceleration case;
      * where the couple reacts decides whether the links carry the full
        contact-patch moment (hub motor: the motor stator is on the upright,
        so the couple is internal to that free body -- exactly NFR27's case,
        which is the whole reason this repo exists) or are relieved of it
        (inboard motor: the halfshaft carries it to the chassis).
    """
    arch = str(get_path(cfg, "powertrain.architecture")).strip()
    layout = _DRIVE_LAYOUTS.get(arch)
    if layout is None:
        raise ConfigError(
            f"powertrain.architecture = {arch!r} has no drive split defined "
            f"(have {sorted(_DRIVE_LAYOUTS)}). It sets the front torque share "
            f"and where the drive couple reacts, both of which change link "
            f"loads, so it must not fall through to a default. Add it to "
            f"`_DRIVE_LAYOUTS` in core/config.py.")
    share, reacts = layout
    return share, dict(reacts)


def build_caliper_on_upright(cfg: Dict[str, Any]) -> bool:
    """True when `brakes.caliper_mount` puts the caliper on the upright.

    Outboard calipers make the brake couple internal to the upright's free
    body, so the links carry it; inboard brakes hand it to the chassis instead.
    Same shape of decision as `build_drive_split`, and the same consequence.
    """
    mount = str(get_path(cfg, "brakes.caliper_mount")).strip().lower()
    if mount not in ("upright", "inboard"):
        raise ConfigError(
            f"brakes.caliper_mount = {mount!r} must be 'upright' or 'inboard'; "
            f"it decides whether the brake couple is reacted through the "
            f"suspension links.")
    return mount == "upright"


def build_tire(cfg: Dict[str, Any], *,
              coefficients_path: Optional[str] = None,
              lambda_muy: Optional[float] = None) -> MF52Lateral:
    """Build the fitted lateral tire model.

    Reads `tires.coefficients` (resolved relative to the config file's own
    directory if relative) and `tires.scaling.mu_y` by default. Either can be
    overridden explicitly -- an explicit argument always wins over the
    config, so a sweep axis or a GUI control over tire coefficient sets does
    not need a YAML edit. Still raises `ConfigError` if a field is `null` and
    no override was given, per the config's own [UNKNOWN] contract -- that
    can happen with an older or partial config file even though the checked-in
    baseline now has real values.

    Conicity/plysteer are zeroed (`without_conicity_and_plysteer`), matching
    every existing T1/T2 fixture -- see that method's docstring for why a
    symmetric-vehicle study should not carry them.
    """
    path = coefficients_path
    if path is None:
        raw = get_path(cfg, "tires.coefficients")
        if raw is None:
            raise ConfigError(
                "tires.coefficients is null (per the config's own [UNKNOWN] "
                "contract) and no coefficients_path override was given. "
                "Point at a fit file, e.g. "
                "data/Tire/fits/hoosier_16x75_R20_7in_p83.mf.json.")
        path = resolve_config_path(cfg, raw)

    mu = lambda_muy
    if mu is None:
        mu = get_path(cfg, "tires.scaling.mu_y")
        if mu is None:
            raise ConfigError(
                "tires.scaling.mu_y is null (per the config's own [UNKNOWN] "
                "contract) and no lambda_muy override was given. See docs/02 "
                "SS5.1 for how to calibrate it.")

    t = MF52Lateral.from_dict(json.loads(Path(path).read_text()))
    t.lambda_muy = float(mu)
    return t.without_conicity_and_plysteer()


def build_combined_tire(cfg: Dict[str, Any], *,
                        coefficients_path: Optional[str] = None,
                        lambda_muy: Optional[float] = None):
    """`build_tire`, wrapped in `CombinedSlipTire` -- what `gearbox/accel.py`'s
    real-tire mode actually consumes (`.peak_mu_x(Fz, gamma)`).

    NOT ported from the original `build_full_vehicle` verbatim (that function
    needs `CornerSetup`/`Powertrain`, both out of scope here) -- this is the
    one line of it (`stub_combined_tire(lateral)`) this repo's Tab 2 actually
    needs, pulled out on its own rather than duplicated inline at the call
    site, so there is exactly one place that knows how to turn a config into a
    `CombinedSlipTire`. Default `mu_x_over_mu_y=1.0`/`kappa_peak=0.10`/
    `Cx=1.65`/`Ex=0.0` (the placeholder longitudinal shape -- see
    `tire/longitudinal.py`'s module docstring) are left at `stub_combined_tire`'s
    defaults, same as the original; only `peak_mu_x` (which routes through the
    FITTED lateral load-sensitivity, not these placeholders) is used downstream.
    """
    from ..tire.longitudinal import stub_combined_tire
    lateral = build_tire(cfg, coefficients_path=coefficients_path, lambda_muy=lambda_muy)
    return stub_combined_tire(lateral)


def build_quarter_car(cfg: Dict[str, Any], axle: str) -> QuarterCar:
    """Quarter car for one axle, with every rate referred TO THE WHEEL.

    The config stores damper rates AT THE DAMPER (`dampers.rate_reference:
    at_damper`), so this converts with the motion ratio that
    `derive_kinematics` gets from the hardpoints:

        c_wheel = c_damper * MR^2        v_knee_wheel = v_knee_damper / MR

    Same MR^2 law as the spring. Doing it here, once, is deliberate -- the
    conversion is a factor of ~MR^4 on damping if you invert it, and the
    config's own comment block flags "are these rates at the damper or at the
    wheel?" as an OPEN QUESTION that everything downstream depends on.

    Sprung corner mass is the axle's share of sprung mass split over two
    wheels, using the static weight distribution.
    """
    if axle not in ("front", "rear"):
        raise ConfigError(f"axle must be 'front' or 'rear', got {axle!r}")

    ref = str(get_path(cfg, "dampers.rate_reference")).strip().lower()
    if ref != "at_damper":
        raise ConfigError(
            f"dampers.rate_reference = {ref!r}; this builder only knows how to "
            f"convert 'at_damper' rates. If the team confirms they are at the "
            f"wheel, skip the MR^2 conversion below -- do not silently apply "
            f"it twice.")

    derived = derive_kinematics(cfg)
    mr = (derived.front.motion_ratio if axle == "front"
          else derived.rear.motion_ratio) if derived is not None else \
        get_path(cfg, f"dampers.motion_ratio_at_ride.{axle}")

    p = build_vehicle_params(cfg)
    front_share = p.weight_dist_front if axle == "front" else 1.0 - p.weight_dist_front
    sprung_corner = p.sprung_mass * front_share / 2.0
    unsprung = get_path(cfg, f"mass.unsprung_kg_per_corner.{axle}")

    spring = get_path(cfg, f"suspension.{axle}.spring_rate_Npm")
    arb = get_path(cfg, f"suspension.{axle}.arb_wheelrate_Npm") or 0.0
    hs = get_path(cfg, f"dampers.high_speed.{axle}")
    ls = get_path(cfg, f"dampers.low_speed.{axle}")

    return QuarterCar(
        sprung_mass=sprung_corner,
        unsprung_mass=unsprung,
        # An ARB adds wheel rate in ROLL only, not in the single-wheel bump a
        # quarter car represents -- so it is deliberately NOT added here.
        wheel_rate=spring * mr ** 2,
        tire_rate=get_path(cfg, "tires.vertical_rate_Npm"),
        damper_bump_low=ls["bump_Ns_per_m"] * mr ** 2,
        damper_bump_high=hs["bump_Ns_per_m"] * mr ** 2,
        damper_rebound_low=ls["rebound_Ns_per_m"] * mr ** 2,
        damper_rebound_high=hs["rebound_Ns_per_m"] * mr ** 2,
        knee_velocity=get_path(cfg, "dampers.knee_velocity_m_per_s") / mr,
    )


def build_bump_envelope(cfg: Dict[str, Any], axle: str):
    """The config's bump load case, swept over its speed list.

    Returns a list of `BumpResult`. Take the max PER MEMBER across it -- see
    `quarter_car.bump_envelope` for why a single speed cannot be conservative
    for both the contact patch and the links.
    """
    lc = get_path(cfg, "load_cases.bump")
    speeds = lc["speed_sweep_m_s"]
    if not speeds:
        raise ConfigError("load_cases.bump.speed_sweep_m_s is empty")
    return bump_envelope(build_quarter_car(cfg, axle),
                         height_m=lc["height_m"], length_m=lc["length_m"],
                         speeds=speeds)

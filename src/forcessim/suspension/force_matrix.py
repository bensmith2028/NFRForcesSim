"""Every component, every load case, in one pass.

WHY A BATCH LAYER RATHER THAN A LOOP AT THE CALL SITE
------------------------------------------------------
`extract_components` answers one corner in one case. A structural review needs
all of them at once, and the moment that loop lives in a notebook or a GUI
callback, three things go wrong: the config-derived inputs (brake bias, rotor
radius, bearing offsets, real caliper mounts) get re-derived slightly
differently each time; the bump case gets forgotten because it does not come
from `STANDARD_CASES`; and nobody computes the ENVELOPE, which is the number
that actually sizes a part.

This module does all three in one place, so the GUI and the export render the
same object rather than each assembling their own.

WHICH CORNERS, AND WHY NOT ALL FOUR
------------------------------------
Front and rear geometry, each evaluated in every case. Left and right are NOT
computed separately, because on a symmetric car they are the same component:

  * Braking and acceleration are symmetric, so left and right are identical.
  * Cornering is NOT symmetric -- but "inside" and "outside" are the same
    corner at +/- a_y, which is exactly how `STANDARD_CASES` defines them. The
    outside case IS the left-hand corner of a right-hand turn.

So two geometries times five cases covers the whole car, and it matches the
layout of the team's spreadsheet. Adding mirrored corners would duplicate every
number under a different name and create a second place for them to disagree.

THE ENVELOPE IS THE POINT
-------------------------
`ForceMatrix.envelope()` reports, per component and attachment point, the
largest magnitude across all cases and which case produced it. That is what a
part has to survive.

BUMP-ON-TOP CASES, AND THE ONE PLACE SUPERPOSITION IS LEGITIMATE
-----------------------------------------------------------------
The four steady cases are ALTERNATIVES -- the car is not braking and
accelerating at once, so their forces are never summed. A bump is different: it
happens WHILE the car is doing one of those things, and hitting a kerb
mid-corner is a real and severe structural case.

So the matrix also builds "bump + <steady>" combinations, in which the bump's
vertical load increment is superposed on a steady case's Fx/Fy/Fz. Two details
make this exact rather than approximate:

  * The increment is `Fz_peak - Fz_static`, not `Fz_peak`. The quarter car's
    peak contact load already contains the static weight, so adding it whole
    would count the car's own weight twice.
  * Force extraction is LINEAR in the applied load (pinned by
    `test_forces_scale_linearly_with_load`), so superposing the LOADS and
    re-extracting gives the same answer as superposing the extracted forces.
    The combined case is therefore solved, not estimated.

These combinations appear in the ENVELOPE only, not in the per-component
tables, which stay at the five headline cases. The envelope then resolves
"bump added to the worst case" PER ATTACHMENT POINT, because the worst steady
case is not the same for all of them -- on the front upright, RL/FL/TR/BC are
worst under braking while OB/IB are worst under cornering. One combined case
per axle would have had to pick one of those and be wrong for the other.

ONE INSTANT CANNOT SIZE THE WHOLE CORNER -- THE SWEPT BUMP
-----------------------------------------------------------
The same argument that forces the envelope to resolve the steady case per
attachment point forces it to resolve the INSTANT per attachment point too, and
for a long time it did not.

The headline bump case is one instant of one speed. Whichever instant is
chosen, it is right for at most one thing. Contact-patch load peaks at first
impact; the force that reaches the LINKS, `Fz - m_u*(a_u + g)`, peaks later,
once the unsprung mass has stopped accelerating upward and stopped absorbing
the hit; the unsprung inertia term itself peaks at yet another instant. Picking
the link peak by sampling `link_borne_force` at the CONTACT-PATCH peak -- which
is what `bump_case` reports, one instant, one scalar -- understated every link
in the car (envelope, NFR27 baseline):

    front rod / rocker.RD / wishbone_upper.PR    2690 N -> 3395 N   1.26x
    front rocker.PV                              2976   -> 3756     1.26x
    front upright.IB / hub.IB                   13846   -> 17516    1.27x
    rear  rod / rocker.RD / wishbone_upper.PR    7185   -> 8927     1.24x
    rear  rocker.PV                              6300   -> 7828     1.24x
    rear  wishbone_upper.RBJ                     3964   -> 4848     1.22x

19 of 27 front points and 20 of 27 rear points moved by more than 5%, median
1.11-1.12x, EVERY ONE OF THEM UNCONSERVATIVE -- these are envelope numbers, so
they are the numbers that size parts. Some governing cases moved with them:
front `hub.CP` went from `bump + outside_cornering` to `bump + braking`.

So `envelope()` additionally sweeps EVERY INSTANT of EVERY SPEED, on its own
and superposed on each steady case, and takes the max per attachment point.
`Fz_contact[j]` is always paired with `a_unsprung[j]` at the SAME index j:
pairing each quantity's own maximum would describe an instant that never
happened, which is the error `quarter_car.bump_case` already warns about.

The sweep is 11 speeds x 25001 instants x 5 variants x 2 axles, which is far
too many to re-extract. It does not have to: extraction is LINEAR in the
applied load (see `test_forces_scale_linearly_with_load`), so two basis
extractions per axle turn the whole trace into vector arithmetic -- see
`_swept_bump_envelope`. 0.5 s of arithmetic per axle plus the extractions that
verify it, against minutes for the naive loop, on a run already spending 13 s
per axle inside the quarter car.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.frames import get_report_frame, to_report_frame
from ..vehicle.load_cases import STANDARD_CASES, LoadCase, corner_loads_for_case
from .component_forces import (ComponentForces, check_action_reaction,
                               check_equilibrium, extract_components, to_lbf)
from .forces import CornerLoads
from .kinematics import corner_state
from .placeholders import Provenance

__all__ = ["CaseResult", "ForceMatrix", "compute_force_matrix", "BUMP_CASE_NAME"]

BUMP_CASE_NAME = "bump"

AXLE_CORNER = {"front": "FR", "rear": "RR"}


@dataclass
class CaseResult:
    """One corner, one case."""

    axle: str
    corner: str
    case_name: str
    forces: ComponentForces
    loads: CornerLoads
    note: str = ""
    """Free text shown alongside the numbers -- e.g. which bump speed won."""

    rod_kind: str = ""
    """"pushrod" | "pullrod" | "direct", from the geometry. Reported rather
    than assumed because the two ends of this car differ -- pushrod front,
    pullrod rear -- and a sheet labelled just "Rods" leaves the reader guessing
    which one they are looking at."""

    @property
    def contact_patch(self) -> Tuple[float, float, float]:
        return (self.loads.Fx, self.loads.Fy, self.loads.Fz)


@dataclass
class ForceMatrix:
    """All results, plus the checks and the envelope."""

    results: List[CaseResult] = field(default_factory=list)

    combined: List[CaseResult] = field(default_factory=list)
    """Bump superposed on each steady case. Feeds the ENVELOPE only -- see the
    module docstring for why these are legitimate superpositions and why they
    are kept out of the per-component tables."""

    swept: Dict[str, Dict[str, Dict[str, dict]]] = field(default_factory=dict)
    """{axle: {component: {point: row}}} in NEWTONS -- the worst the bump does
    at each attachment point over every instant of every speed, alone and on
    top of each steady case.

    NOT a `CaseResult`, and it cannot be one: different points peak at
    different instants, so there is no single force set to report. Each row is
    its own point's own worst instant, which is exactly why it catches what the
    headline `bump` case cannot. Feeds the ENVELOPE only."""

    provenance: Provenance = field(default_factory=Provenance)
    failures: Dict[str, str] = field(default_factory=dict)
    """{axle/case: why it could not be computed}. Present so one bad case does
    not silently vanish from a table that looks complete."""

    # -- navigation -------------------------------------------------------
    @property
    def case_names(self) -> List[str]:
        seen: List[str] = []
        for r in self.results:
            if r.case_name not in seen:
                seen.append(r.case_name)
        return seen

    @property
    def components(self) -> List[str]:
        seen: List[str] = []
        for r in self.results:
            for c in r.forces.components:
                if c not in seen:
                    seen.append(c)
        return seen

    def get(self, axle: str, case_name: str) -> Optional[CaseResult]:
        for r in self.results:
            if r.axle == axle and r.case_name == case_name:
                return r
        return None

    def for_axle(self, axle: str) -> List[CaseResult]:
        return [r for r in self.results if r.axle == axle]

    @property
    def combined_case_names(self) -> List[str]:
        """Names of the bump-on-top cases, in the order they were built."""
        seen: List[str] = []
        for r in self.combined:
            if r.case_name not in seen:
                seen.append(r.case_name)
        return seen

    @property
    def swept_case_names(self) -> List[str]:
        """Labels the swept bump can put in an envelope's `Case` column.

        Reported because they name a speed and an instant that no `CaseResult`
        carries, so a reader who finds one in the envelope cannot look it up in
        `case_names`.
        """
        seen: List[str] = []
        for comps in self.swept.values():
            for points in comps.values():
                for row in points.values():
                    if row["Case"] not in seen:
                        seen.append(row["Case"])
        return seen

    def get_combined(self, axle: str, case_name: str) -> Optional[CaseResult]:
        for r in self.combined:
            if r.axle == axle and r.case_name == case_name:
                return r
        return None

    def all_results(self) -> List[CaseResult]:
        """Everything solved -- headline cases and bump-on-top combinations.

        Anything that walks EVERY force should use this rather than `results`,
        which holds only the headline cases. The CSV export used `results` and
        silently omitted all eight combined cases.
        """
        return self.results + self.combined

    # -- the spreadsheet view --------------------------------------------
    def rows(self, axle: str, component: str, lbf: bool = True,
             frame=None) -> List[dict]:
        """One row per (case, attachment point), ready for a table.

        Long format rather than a pivot, because the set of attachment points
        differs between components and a wide table would be mostly holes.

        `frame` is the reporting triad -- see `ComponentForces.table`.
        """
        out: List[dict] = []
        for r in self.for_axle(axle):
            if component not in r.forces.components:
                continue
            for point, v in r.forces.table(component, lbf=lbf, frame=frame).items():
                out.append({"Case": r.case_name, "Point": point,
                            "Fx": v["Fx"], "Fy": v["Fy"], "Fz": v["Fz"],
                            "Net": v["Net"]})
        return out

    def envelope(self, axle: str, component: str, lbf: bool = True,
                 include_combined: bool = True, frame=None) -> List[dict]:
        """Worst magnitude per attachment point, and the case that caused it.

        Considers the five headline cases, the bump-on-top combinations, AND
        the bump swept over every instant of every speed -- so each point is
        resolved against ITS OWN worst steady case and ITS OWN worst instant,
        rather than one of each chosen for the whole axle.

        `include_combined=False` drops BOTH envelope-only pools and leaves
        exactly the five headline cases, i.e. the view that matches the
        per-component table row for row. It is the debugging view, not the
        sizing view -- on NFR26 it reports 2308 N on the front rod against the
        4181 N the rod actually has to survive.

        `frame` is the reporting triad -- see `ComponentForces.table`. The
        winner is selected on `Net`, which no reporting frame changes, so the
        SAME rows win in every frame and only their components are relabelled.
        """
        frame = get_report_frame(frame)
        pool = list(self.for_axle(axle))
        if include_combined:
            pool += [r for r in self.combined if r.axle == axle]

        best: Dict[str, dict] = {}
        for r in pool:
            if component not in r.forces.components:
                continue
            for point, v in r.forces.table(component, lbf=lbf, frame=frame).items():
                if point not in best or v["Net"] > best[point]["Net"]:
                    best[point] = {"Point": point, "Case": r.case_name,
                                   "Fx": v["Fx"], "Fy": v["Fy"],
                                   "Fz": v["Fz"], "Net": v["Net"]}

        # The swept rows are stored in newtons -- they never go through
        # `ComponentForces.table`, which is what converts everywhere else -- so
        # they are converted here, at the same boundary, and re-normed rather
        # than scaled so a row can never disagree with its own components.
        if include_combined:
            for point, row in self.swept.get(axle, {}).get(component, {}).items():
                v = np.array([row["Fx"], row["Fy"], row["Fz"]], dtype=float)
                if lbf:
                    v = to_lbf(v)
                net = float(np.linalg.norm(v))
                v = to_report_frame(v, frame)
                if point not in best or net > best[point]["Net"]:
                    best[point] = {"Point": point, "Case": row["Case"],
                                   "Fx": float(v[0]), "Fy": float(v[1]),
                                   "Fz": float(v[2]), "Net": net}
        return [best[p] for p in best]

    # -- verification ------------------------------------------------------
    def check(self, atol: float = 1e-6) -> Dict[str, dict]:
        """Equilibrium and action/reaction on every result.

        Empty dict means every free body in every case closes. Anything else
        must be surfaced before the numbers are used, which is why the GUI and
        the export both call this rather than trusting the extraction.
        """
        bad: Dict[str, dict] = {}
        for r in self.results + self.combined:
            key = f"{r.axle}/{r.case_name}"
            pairs = check_action_reaction(r.forces, atol)
            bodies = check_equilibrium(r.forces, atol)
            if pairs or bodies:
                bad[key] = {"action_reaction": pairs, "equilibrium": bodies}
        return bad

    def report(self, lbf: bool = True, frame=None) -> str:
        unit = "lbf" if lbf else "N"
        frame = get_report_frame(frame)
        lines = [f"Component forces, {len(self.results)} case(s), {unit}, "
                 f"{frame.header}"]
        for r in self.results:
            lines.append("")
            lines.append(f"=== {r.axle} / {r.case_name} ===")
            if r.note:
                lines.append(f"    {r.note}")
            lines.append(r.forces.report(lbf=lbf, frame=frame))
        if self.provenance.any_placeholder:
            lines.append("")
            lines.append(self.provenance.report())
        return "\n".join(lines)


def _axle_inputs(cfg, axle: str) -> dict:
    """Config-derived arguments for `extract_components`, resolved once.

    Imported lazily: `core.config` imports the suspension package, so a
    module-level import here would be circular.
    """
    from ..core.config import _opt, build_corner_geometry, get_path
    from .geometry import load_hardpoints_csv

    corner = build_corner_geometry(cfg, axle)
    state = corner_state(corner, float(corner.wheel_centre[2]), 0.0)

    hardpoints = None
    try:
        from ..core.config import resolve_config_path
        path = resolve_config_path(cfg, get_path(cfg, f"suspension.{axle}.hardpoints_file"))
        overrides = {}
        for key, value in (_opt(cfg, f"hardpoint_overrides.{axle}") or {}).items():
            point, _, ax = key[:-2].rpartition("_")
            overrides[f"{point}.{ax}"] = float(value) * 1000.0
        hardpoints = load_hardpoints_csv(path, overrides=overrides or None)
    except Exception:
        # Real caliper mounts are optional; synthesis covers their absence and
        # says so. A failure to re-read the sheet must not take the whole run
        # down when the geometry itself already loaded.
        hardpoints = None

    return {
        "corner": corner,
        "state": state,
        "hardpoints": hardpoints,
        "inner_bearing_offset": _opt(cfg, "bearings.inner_offset_m", -0.035),
        "outer_bearing_offset": _opt(cfg, "bearings.outer_offset_m", 0.035),
        "rotor_effective_radius": _opt(
            cfg, f"brakes.rotor_effective_radius_m.{axle}", 0.077216),
        "axially_located": _opt(cfg, "bearings.axially_located", "UNCONFIRMED"),
    }


def compute_force_matrix(cfg, *, cases: Optional[Sequence[LoadCase]] = None,
                         axles: Sequence[str] = ("front", "rear"),
                         include_bump: bool = True,
                         include_combined: bool = True) -> ForceMatrix:
    """All components, all cases, both axles.

    Args:
        cfg: a loaded config dict.
        cases: steady-state cases; defaults to `STANDARD_CASES`.
        include_bump: append the bump case at the speed that produces the worst
            CONTACT PATCH, and sweep the whole bump into `swept` for the
            envelope. The headline case is one instant, which no link peaks at;
            the sweep is what covers them. See `_bump_result`.
        include_combined: also build "bump + <steady>" cases, both as single
            instants and inside the sweep. Envelope only. Requires
            `include_bump`.
    """
    from ..core.config import (build_bump_envelope, build_brake_bias,
                               build_caliper_on_upright, build_drive_split,
                               get_path, build_vehicle_params)

    cases = list(cases if cases is not None else STANDARD_CASES)
    params = build_vehicle_params(cfg)
    bias = build_brake_bias(cfg) or 0.67

    # HOW THE CAR IS DRIVEN AND BRAKED IS READ, NOT ASSUMED.
    #
    # These three used to sit at `corner_loads_for_case`'s defaults -- 50/50
    # drive, couple on the upright, one 0.198 m radius for every corner -- so a
    # rear-drive car with inboard motors got NFR26's quad-hub load path no
    # matter what its config said. The acceleration case is where that shows:
    # the front links carried half the tractive force they should never have
    # seen, and the rear links carried a drive couple the halfshafts actually
    # take to the chassis -- worth more than 2x on the rear lower links at a
    # fixed contact-patch load.
    drive_front, drive_on_upright = build_drive_split(cfg)
    caliper_on_upright = build_caliper_on_upright(cfg)
    loaded_radius = {k: get_path(cfg, "wheels.loaded_radius_m."
                                      + ("front" if k[0] == "F" else "rear"))
                     for k in ("FL", "FR", "RL", "RR")}

    unsprung = {}
    for k in ("FL", "FR", "RL", "RR"):
        side = "front" if k[0] == "F" else "rear"
        try:
            unsprung[k] = get_path(cfg, f"mass.unsprung_kg_per_corner.{side}")
        except Exception:
            unsprung[k] = 0.0

    matrix = ForceMatrix()
    seen_notes = set()

    for axle in axles:
        try:
            inp = _axle_inputs(cfg, axle)
        except Exception as e:
            matrix.failures[f"{axle}/*"] = f"could not build geometry: {e}"
            continue

        corner_key = AXLE_CORNER[axle]
        act = inp["corner"].actuator
        rod_kind = act.kind if act is not None else ""
        extract_kwargs = {k: v for k, v in inp.items()
                          if k not in ("corner", "state")}

        for case in cases:
            try:
                loads = corner_loads_for_case(
                    params, case, brake_bias_front=bias,
                    drive_fraction_front=drive_front,
                    loaded_radius=loaded_radius,
                    caliper_on_upright=caliper_on_upright,
                    drive_reacts_on_upright=drive_on_upright[axle],
                    unsprung_mass=unsprung)[corner_key]
                cf = extract_components(inp["corner"], inp["state"], loads,
                                        case_name=case.name, **extract_kwargs)
                matrix.results.append(CaseResult(
                    axle=axle, corner=corner_key, case_name=case.name,
                    forces=cf, loads=loads, rod_kind=rod_kind))
            except Exception as e:
                matrix.failures[f"{axle}/{case.name}"] = str(e)

        if include_bump:
            sweep = None
            try:
                # Built ONCE and handed to both consumers. `_bump_result` used
                # to build its own, so asking for the swept envelope as well
                # ran 11 quarter-car simulations twice per axle -- 25 s of the
                # run spent computing a trace it already had.
                sweep = build_bump_envelope(cfg, axle)
                res, note = _bump_result(cfg, axle, results=sweep)
                loads = res.to_corner_loads(
                    caliper_on_upright=caliper_on_upright,
                    drive_reacts_on_upright=drive_on_upright[axle])
                cf = extract_components(inp["corner"], inp["state"], loads,
                                        case_name=BUMP_CASE_NAME, **extract_kwargs)
                matrix.results.append(CaseResult(
                    axle=axle, corner=corner_key, case_name=BUMP_CASE_NAME,
                    forces=cf, loads=loads, note=note, rod_kind=rod_kind))
            except Exception as e:
                matrix.failures[f"{axle}/{BUMP_CASE_NAME}"] = str(e)
                res = None

            # ---- bump superposed on each steady case ----
            if res is not None and include_combined:
                # The quarter car's peak load already contains the static
                # weight, so the INCREMENT is what gets added -- adding
                # `Fz_peak` whole would carry the car's weight twice.
                static = res.Fz_peak / res.load_factor if res.load_factor else 0.0
                d_Fz = res.Fz_peak - static
                for case in cases:
                    name = f"{BUMP_CASE_NAME} + {case.name}"
                    try:
                        steady = matrix.get(axle, case.name)
                        if steady is None:
                            continue
                        loads = replace(
                            steady.loads,
                            Fz=steady.loads.Fz + d_Fz,
                            unsprung_accel=np.array(
                                [0.0, 0.0, res.a_unsprung_at_peak]))
                        cf = extract_components(inp["corner"], inp["state"],
                                                loads, case_name=name,
                                                **extract_kwargs)
                        matrix.combined.append(CaseResult(
                            axle=axle, corner=corner_key, case_name=name,
                            forces=cf, loads=loads, rod_kind=rod_kind,
                            note=(f"{case.name} with the bump's "
                                  f"{d_Fz:.0f} N vertical increment on top")))
                    except Exception as e:
                        matrix.failures[f"{axle}/{name}"] = str(e)

            # ---- the bump swept over every instant of every speed ----
            #
            # This is what makes the envelope conservative for the LINKS. The
            # two blocks above are both single instants -- the one where the
            # contact patch peaks -- and the link-borne force does not peak
            # there. See the module docstring for the 1.22-1.27x it costs.
            if sweep is not None:
                steady_pool = [r for r in matrix.results
                               if r.axle == axle and r.case_name != BUMP_CASE_NAME]
                try:
                    matrix.swept[axle] = _swept_bump_envelope(
                        inp["corner"], inp["state"], extract_kwargs, sweep,
                        get_path(cfg, "load_cases.bump.speed_sweep_m_s"),
                        caliper_on_upright=caliper_on_upright,
                        drive_reacts_on_upright=drive_on_upright[axle],
                        steady=steady_pool if include_combined else ())
                except Exception as e:
                    # Recorded rather than raised so a GUI still renders the
                    # rest, but recorded LOUDLY: `failures` is on every report
                    # and the batch tests assert it is empty. An envelope that
                    # quietly dropped its swept pool would understate every
                    # link and look complete doing it -- the exact failure this
                    # module was fixed for.
                    matrix.failures[f"{axle}/{BUMP_CASE_NAME} swept"] = str(e)

    # Merge provenance across results, de-duplicated: the same placeholder is
    # reported once for the run, not once per case.
    for r in matrix.results + matrix.combined:
        for n in r.forces.provenance.notes:
            if (n.key, n.value) in seen_notes:
                continue
            seen_notes.add((n.key, n.value))
            matrix.provenance.notes.append(n)

    return matrix


def _bump_result(cfg, axle: str, results=None):
    """Peak-CONTACT-PATCH point of the swept bump, plus a note saying which speed.

    SELECTED ON CONTACT PATCH, WHICH IS WHAT THIS ONE INSTANT CAN HONESTLY CARRY
    ---------------------------------------------------------------------------
    This result becomes the single `bump` column of the per-component tables, so
    it has to be one instant of one speed, and one instant cannot be the worst
    for everything. Contact patch is the quantity it IS the worst for: it
    governs the tire, wheel, bearings and upright, and it rises monotonically
    with speed, so the chosen instant is the top of the sweep and nothing about
    the choice is subtle.

    It used to select on `BumpResult.link_borne_force` instead, on the argument
    that the links are what the table is for. Two things were wrong with that.
    `link_borne_force` is `Fz - m_u*(a_u + g)` sampled at the CONTACT-PATCH peak
    -- `bump_case` reports one instant -- and the link-borne force does not peak
    there, so the number being maximised was not the link peak at all. Taken at
    its OWN peak instant instead, the link-borne force keeps rising up the whole
    sweep -- the front flat from 20 to 30 m/s, the rear still climbing at 30.
    The apparent peak near 11 m/s and the falloff above it, which this docstring
    used to explain as the tire starting to skip, were artefacts of the
    mis-sampling and nothing else. And because the scalar was wrong, the SPEED
    it selected was wrong too: 11 m/s instead of 30.

    The links are now covered where they can be covered properly -- by the
    swept envelope, which resolves the instant per attachment point. See
    `_swept_bump_envelope`.

    Args:
        results: a prebuilt `build_bump_envelope` list, to avoid re-running 11
            quarter-car simulations for a caller that already has them.
    """
    from ..core.config import build_bump_envelope, get_path

    if results is None:
        results = build_bump_envelope(cfg, axle)
    speeds = get_path(cfg, "load_cases.bump.speed_sweep_m_s")
    i = int(np.argmax([r.Fz_peak for r in results]))
    r = results[i]
    height = get_path(cfg, "load_cases.bump.height_m") * 1000.0
    length = get_path(cfg, "load_cases.bump.length_m") * 1000.0
    note = (f"worst contact patch at {speeds[i]:.0f} m/s of the swept "
            f"{height:.0f} mm x {length:.0f} mm bump; peak contact patch "
            f"{r.Fz_peak:.0f} N ({r.load_factor:.2f}x static), of which "
            f"{r.link_borne_force:.0f} N reaches the links AT THAT INSTANT -- "
            f"the links peak at a different instant, which only the swept "
            f"envelope sees")
    return r, note


_BASIS_FZ = 1000.0
"""Contact-patch load the linear basis case is extracted at, newtons. Any value
would do -- the extraction is linear -- and a round one keeps the scale factors
in the sweep readable when they are printed during debugging."""


def _swept_bump_envelope(corner, state, extract_kwargs, results,
                         speeds: Sequence[float], *,
                         caliper_on_upright: bool,
                         drive_reacts_on_upright: bool,
                         steady: Sequence[CaseResult] = ()
                         ) -> Dict[str, Dict[str, dict]]:
    """Worst force at each attachment point over EVERY instant of EVERY speed.

    Returns {component: {point: row}} in newtons, each row carrying the vector
    at that point's own worst instant and a `Case` naming the variant and speed
    it came from. Points peak at different instants -- that is the whole reason
    this exists -- so the returned rows do NOT describe one force set and must
    never be assembled into one.

    TWO BASIS EXTRACTIONS INSTEAD OF 550,000
    -----------------------------------------
    11 speeds x 25001 instants x (bump alone + one variant per steady case) is
    far more solves than a structural review can wait for -- the naive loop did
    not finish in two minutes even subsampled. It does not need them: extraction
    is linear in the applied load (pinned by
    `test_forces_scale_linearly_with_load`), so two extractions span everything
    the bump can do to this corner:

        E1  Fz = _BASIS_FZ, a_u = -g   pure contact patch; the unsprung weight
                                       and its inertia cancel exactly
        E2  Fz = 0,         a_u = 0    pure unsprung weight term, -m_u*g

    and then, with `beta = (g + a_u[j])/g` folding weight and d'Alembert
    inertia into one factor,

        bump alone at instant j   (Fz[j]/_BASIS_FZ)*E1 + beta[j]*E2
        bump + steady at j        steady + ((Fz[j]-static)/_BASIS_FZ)*E1
                                         + (beta[j] - 1)*E2

    the `-1` and the `-static` both removing what the steady case already
    carries -- one unsprung weight and one static wheel load -- exactly as the
    single-instant combined case does.

    This is EXACT here, not a small-angle argument, for two reasons worth
    stating because both are properties of `extract_components` that a future
    edit could break: the `braking = abs(brake_torque) >= abs(drive_torque)`
    branch does not depend on Fz, so it cannot flip part-way along a trace; and
    the spindle torque `T_required = M_cp . spin` is independent of Fz because
    `cp - wc` is (0, 0, -r) and its cross product with a vertical force is zero,
    so a vertical increment carries no torque and lands identically on either
    side of that branch. Measured residual is ~1e-13 N.

    Because that exactness is load-bearing, it is CHECKED rather than trusted:
    each variant's winning instant is re-extracted directly and compared. If the
    linearity ever stops holding this raises, and `compute_force_matrix` turns
    it into a recorded failure. Pinned by
    `test_swept_basis_reconstruction_matches_direct_extraction`.
    """
    if len(speeds) != len(results):
        # Zipping them would truncate silently and label every remaining row
        # with the wrong speed -- a wrong number that still reads as a result.
        raise ValueError(f"{len(speeds)} speeds against {len(results)} bump "
                         f"results; they index each other")

    base = results[0].to_corner_loads(
        caliper_on_upright=caliper_on_upright,
        drive_reacts_on_upright=drive_reacts_on_upright)
    g = float(base.gravity)
    # The static wheel load is a property of the quarter car, not of the speed,
    # so any result in the sweep gives the same number.
    static = (results[0].Fz_peak / results[0].load_factor
              if results[0].load_factor else 0.0)

    def _solve(loads, name: str):
        return extract_components(corner, state, loads, case_name=name,
                                  **extract_kwargs)

    def _bump_only(Fz: float, a_u: float):
        """The load a pure bump applies: vertical at the patch, nothing else."""
        return replace(base, Fx=0.0, Fy=0.0, Fz=Fz, Mx=0.0, My=0.0, Mz=0.0,
                       brake_torque=0.0, drive_torque=0.0,
                       unsprung_accel=np.array([0.0, 0.0, a_u]))

    E1 = _solve(_bump_only(_BASIS_FZ, -g), "basis: contact patch")
    E2 = _solve(_bump_only(0.0, 0.0), "basis: unsprung weight")

    keys = [(comp, point) for comp, pts in E1.components.items() for point in pts]
    A = np.array([E1.components[c][p] for c, p in keys], dtype=float)
    B = np.array([E2.components[c][p] for c, p in keys], dtype=float)

    variants: List[Tuple[str, Optional[CaseResult]]] = [(BUMP_CASE_NAME, None)]
    variants += [(f"{BUMP_CASE_NAME} + {s.case_name}", s) for s in steady]

    n = len(keys)
    cols = np.arange(n)
    best = np.full(n, -np.inf)
    best_vec = np.zeros((n, 3))
    best_case = [""] * n

    for speed, r in zip(speeds, results):
        # SAME INDEX, ALWAYS. `Fz_contact[j]` goes with `a_unsprung[j]`.
        #
        # Combining each array's own extreme describes an instant that never
        # happened, and it is not even a conservative shortcut: on the front's
        # 30 m/s trace the patch peaks 1.7 ms AFTER the unsprung acceleration
        # does, and that acceleration is +299 m/s^2 there -- the mass moving up
        # with the wheel, unloading the links -- so the mismatched pair gives
        # 0.64x the real rod load. Borrow from an instant where a_u is negative
        # instead and the same shortcut overstates it. The direction depends on
        # which instant you stole from, which is exactly why the index is kept.
        Fz = np.asarray(r.result.Fz_contact, dtype=float)
        a_u = np.asarray(r.result.a_unsprung, dtype=float)
        beta_bump = (g + a_u) / g

        for name, s in variants:
            if s is None:
                alpha, beta = Fz / _BASIS_FZ, beta_bump
                S = np.zeros((n, 3))
            else:
                alpha, beta = (Fz - static) / _BASIS_FZ, beta_bump - 1.0
                S = np.array([s.forces.components[c][p] for c, p in keys],
                             dtype=float)

            # (instants, points, 3). ~18 MB per variant at 25001 instants,
            # freed each pass -- cheap next to re-solving the corner.
            V = S + alpha[:, None, None] * A + beta[:, None, None] * B
            nets = np.linalg.norm(V, axis=2)
            j = np.argmax(nets, axis=0)          # each point's OWN instant
            peak = nets[j, cols]

            # Checked at the instant carrying this variant's largest force --
            # one extraction, placed where drift would do the most damage.
            worst_j = int(j[int(np.argmax(peak))])
            loads_j = (_bump_only(float(Fz[worst_j]), float(a_u[worst_j]))
                       if s is None else
                       replace(s.loads,
                               Fz=s.loads.Fz + float(Fz[worst_j]) - static,
                               unsprung_accel=np.array(
                                   [0.0, 0.0, float(a_u[worst_j])])))
            _verify_linear_basis(_solve(loads_j, "linearity check"),
                                 keys, V[worst_j], name)

            won = peak > best
            best = np.where(won, peak, best)
            best_vec = np.where(won[:, None], V[j, cols], best_vec)
            label = f"{name} swept @ {speed:g} m/s"
            best_case = [label if w else c for w, c in zip(won, best_case)]

    out: Dict[str, Dict[str, dict]] = {}
    for (comp, point), vec, net, case in zip(keys, best_vec, best, best_case):
        out.setdefault(comp, {})[point] = {
            "Point": point, "Case": case, "Fx": float(vec[0]),
            "Fy": float(vec[1]), "Fz": float(vec[2]), "Net": float(net)}
    return out


def _verify_linear_basis(direct: ComponentForces, keys, rebuilt: np.ndarray,
                         name: str) -> None:
    """Confirm the two-basis reconstruction reproduces a real solve.

    `direct` is `extract_components` run on the actual load at one instant;
    `rebuilt` is what the basis arithmetic said that instant would be. They
    agree to ~1e-13 N today.

    Raises rather than warns, because the reconstruction is the only thing
    standing between this envelope and 550,000 real solves. If it ever drifts,
    every swept number drifts with it and nothing about the output would look
    wrong -- the failure mode this module already paid for once.
    """
    resid = 0.0
    scale = 0.0
    for i, (comp, point) in enumerate(keys):
        got = np.asarray(direct.components[comp][point], dtype=float)
        resid = max(resid, float(np.linalg.norm(rebuilt[i] - got)))
        scale = max(scale, float(np.linalg.norm(got)))
    if resid > 1e-6 + 1e-9 * scale:
        raise ValueError(
            f"'{name}': the linear basis reconstruction is off by "
            f"{resid:.3e} N against a direct extraction (peak {scale:.0f} N). "
            f"Extraction is no longer linear in the applied vertical load, so "
            f"the swept envelope cannot be trusted -- see _swept_bump_envelope "
            f"for the two properties that made it exact.")

#!/usr/bin/env python3
"""Run the suspension force-extraction pipeline from one vehicle workbook.

    python run_sim.py NFR27_vehicle.xlsx
    python run_sim.py NFR27_vehicle.xlsx -o results.csv

Reads one Excel workbook containing front + rear suspension geometry and
every vehicle parameter the pipeline needs (see README.md for the exact
format), runs the four standard structural load cases (braking, outside
cornering, inside cornering, acceleration) at all four corners, and writes
one CSV of per-component, per-attachment-point forces.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from forcessim.core.config import (
    build_brake_bias, build_caliper_on_upright, build_corner_geometry,
    build_drive_split, build_vehicle_params,
)
from forcessim.core.vehicle_xlsx import VehicleXlsxError, load_vehicle_xlsx
from forcessim.suspension.component_forces import extract_components
from forcessim.suspension.geometry import (
    DOUBLE_WISHBONE_PUSHROD, FIVE_LINK_PULLROD, build_corner, load_hardpoints_csv,
)
from forcessim.suspension.kinematics import _nominal_spindle_axis, corner_state
from forcessim.vehicle.load_cases import STANDARD_CASES, corner_loads_for_case

CORNERS = ("FL", "FR", "RL", "RR")
AXLE_OF = {"FL": "front", "FR": "front", "RL": "rear", "RR": "rear"}
_TOPOLOGIES = {"double_wishbone": DOUBLE_WISHBONE_PUSHROD, "five_link": FIVE_LINK_PULLROD}
_ROD_POINTS = {"pushrod": ("IB_Push", "OB_Push"), "pullrod": ("IB_Pull", "OB_Pull")}

OUTPUT_FIELDS = ["Corner", "Case", "Component", "Point", "Fx", "Fy", "Fz", "Net"]


def _mirror_points(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Negate y (ISO left) on every point -- the sheet, mirrored about the
    vehicle centreline. See `run_sim.py`'s module docstring / README for why
    FL/RL are built this way: there is only one hardpoint sheet per axle
    (the right-hand corner), and mirroring it is the same technique this
    project's own tests (`test_mirrored_corners_carry_identical_link_loads`)
    validate against.
    """
    return {k: np.array([p[0], -p[1], p[2]]) for k, p in pts.items()}


def _corner_geometry_and_state(cfg: dict, corner_key: str):
    """(CornerGeometry, CornerState, raw hardpoints dict) for one corner.

    FR/RR are the exact, byte-validated construction (`build_corner_geometry`
    reads the sheet directly). FL/RL have no separate hardpoint sheet, so the
    same sheet is mirrored about the centreline -- both the link/actuator
    points AND the raw hardpoints dict `extract_components` reads for the
    caliper/bearing seats must be mirrored together, or the caliper and
    bearing geometry disagrees with the mirrored spindle axis.
    """
    axle = AXLE_OF[corner_key]
    sus = cfg["suspension"][axle]

    if corner_key in ("FR", "RR"):
        # `build_corner_geometry` already reads the hardpoints file and
        # stores the raw points on the corner (`build_corner` sets
        # `CornerGeometry.points`), so there is no need to read the file a
        # second time here.
        corner = build_corner_geometry(cfg, axle)
    else:
        hardpoints = load_hardpoints_csv(Path(sus["hardpoints_file"]))
        mirrored = _mirror_points(hardpoints)
        corner = build_corner(
            corner_key, mirrored, _TOPOLOGIES[sus["topology"]], sus["actuation"],
            _ROD_POINTS[sus["actuation"]],
            rod_pickup=sus.get("rod_pickup", "upper_wishbone"))
        wc = cfg["wheels"]["centre_m"][axle]
        corner.wheel_centre = np.array([wc[0], -wc[1], wc[2]])
        corner.spindle_axis = _nominal_spindle_axis(
            np.radians(sus["static_camber_deg"]), np.radians(sus["static_toe_deg"]),
            left=True)
        corner.loaded_radius = cfg["wheels"]["loaded_radius_m"][axle]

    state = corner_state(corner, float(corner.wheel_centre[2]), 0.0)
    return corner, state, corner.points


def run(cfg: dict) -> List[dict]:
    """cfg -> one row per (corner, case, component, attachment point)."""
    params = build_vehicle_params(cfg)
    drive_front, drive_on_upright = build_drive_split(cfg)
    caliper_on_upright = build_caliper_on_upright(cfg)
    brake_bias = build_brake_bias(cfg)
    if brake_bias is None:
        brake_bias = 0.67

    loaded_radius = {k: cfg["wheels"]["loaded_radius_m"][AXLE_OF[k]] for k in CORNERS}
    unsprung_mass = {k: cfg["mass"]["unsprung_kg_per_corner"][AXLE_OF[k]] for k in CORNERS}

    # Geometry does not depend on the load case, so build all four corners
    # once, up front.
    geometry = {k: _corner_geometry_and_state(cfg, k) for k in CORNERS}

    rows: List[dict] = []
    for case in STANDARD_CASES:
        # `corner_loads_for_case` takes one `drive_reacts_on_upright` bool for
        # ALL four corners, but the two axles can have genuinely different
        # values (a hub-motor front + inboard-diff rear, say) -- so it is
        # called once per axle and the relevant two corners kept from each
        # call, exactly as the old GUI's `app/model.py::compute_loads` did.
        loads_by_corner = {}
        for axle in ("front", "rear"):
            loads = corner_loads_for_case(
                params, case, brake_bias_front=brake_bias,
                drive_fraction_front=drive_front, loaded_radius=loaded_radius,
                caliper_on_upright=caliper_on_upright,
                drive_reacts_on_upright=drive_on_upright[axle],
                unsprung_mass=unsprung_mass)
            for k in (("FL", "FR") if axle == "front" else ("RL", "RR")):
                loads_by_corner[k] = loads[k]

        for corner_key in CORNERS:
            axle = AXLE_OF[corner_key]
            corner, state, hardpoints = geometry[corner_key]
            cf = extract_components(
                corner, state, loads_by_corner[corner_key], case_name=case.name,
                inner_bearing_offset=cfg["bearings"]["inner_offset_m"],
                outer_bearing_offset=cfg["bearings"]["outer_offset_m"],
                rotor_effective_radius=cfg["brakes"]["rotor_effective_radius_m"][axle],
                axially_located=cfg["bearings"]["axially_located"],
                hardpoints=hardpoints)

            for component in cf.components:
                for point, v in cf.table(component, lbf=True).items():
                    rows.append({
                        "Corner": corner_key, "Case": case.name,
                        "Component": component, "Point": point,
                        "Fx": round(v["Fx"], 3), "Fy": round(v["Fy"], 3),
                        "Fz": round(v["Fz"], 3), "Net": round(v["Net"], 3),
                    })
    return rows


def write_output_csv(rows: List[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the suspension force-extraction pipeline from one "
                    "vehicle workbook and write a per-component forces CSV.")
    parser.add_argument("input_xlsx", type=Path,
                        help="vehicle workbook (.xlsx): front + rear geometry "
                             "and vehicle parameters (see README.md for the "
                             "format)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output CSV path (default: "
                             "<input file>_forces.csv, next to the input)")
    args = parser.parse_args(argv)

    output = args.output or args.input_xlsx.with_name(
        args.input_xlsx.stem + "_forces.csv")

    if not args.input_xlsx.is_file():
        print(f"error: input file not found: {args.input_xlsx}", file=sys.stderr)
        return 1

    try:
        cfg = load_vehicle_xlsx(args.input_xlsx)
        rows = run(cfg)
    except (VehicleXlsxError, KeyError, ValueError) as exc:
        # KeyError/ValueError also cover errors raised deeper in the physics
        # pipeline (e.g. a topology missing a required hardpoint, or an
        # invalid `caliper_mount`/`architecture` value) -- those already
        # carry a clear, specific message (see core/config.py,
        # suspension/geometry.py), so the fix here is just to print it
        # instead of a full traceback, not to reword it.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_output_csv(rows, output)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

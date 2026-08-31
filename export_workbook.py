#!/usr/bin/env python3
"""Run the full force-extraction pipeline and write the team's xlsx workbook.

    python export_workbook.py NFR27_vehicle.xlsx
    python export_workbook.py NFR27_vehicle.xlsx -o results.xlsx

Unlike `run_sim.py` (the four steady cases, all four corners, one flat CSV),
this writes the multi-sheet workbook the team's own spreadsheet is shaped
like: one tab per component (five headline cases -- braking, outside/inside
cornering, acceleration, bump), one tab per bump-on-top-of-a-steady-case
combination, a swept-bump tab, an Envelope tab (worst case per attachment
point, bump swept included), and a Key tab decoding every attachment-point
code. See `suspension/export.py` and `suspension/force_matrix.py`.

Only front/rear are computed (not all four corners): braking and
acceleration are left/right symmetric, and cornering's inside/outside pair
IS the left/right distinction -- see the Key sheet's "Load case" section, or
`suspension/force_matrix.py`'s module docstring.

This is slower than `run_sim.py` -- the bump case runs a 2-DOF quarter-car
simulation at 11 speeds per axle (~10-15 s each) plus the swept-bump
envelope's vectorised sweep, so a full run takes on the order of a minute.

The default output name is `<input file>_forces.xlsx`, NOT `<input
file>.xlsx` -- the input is itself an .xlsx now (see `core/vehicle_xlsx.py`),
and reusing its stem with just the suffix swapped would silently overwrite
the input the first time someone ran this without `-o`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from forcessim.core.vehicle_xlsx import VehicleXlsxError, load_vehicle_xlsx
from forcessim.suspension.export import write_csv, write_workbook
from forcessim.suspension.force_matrix import compute_force_matrix


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline (including the bump case and "
                    "envelope) and write an xlsx or long-format csv workbook.")
    parser.add_argument("input_xlsx", type=Path,
                        help="vehicle workbook (.xlsx): front + rear geometry "
                             "and vehicle parameters (see README.md for the "
                             "format)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output path (default: <input file>_forces.xlsx, "
                             "next to the input). Extension picks the "
                             "format: .xlsx or .csv.")
    parser.add_argument("--units", choices=["lbf", "N"], default="lbf",
                        help="reporting units (default: lbf, matching the "
                             "team's spreadsheet)")
    parser.add_argument("--frame", choices=["iso8855", "solidworks"],
                        default="solidworks",
                        help="reporting axis triad (default: solidworks, "
                             "matching the team's spreadsheet)")
    parser.add_argument("--no-bump", action="store_true",
                        help="skip the bump case, bump-on-top combinations "
                             "and swept envelope (steady cases only, ~100x "
                             "faster) -- component sheets and Envelope will "
                             "be missing the bump case and its rows")
    args = parser.parse_args(argv)

    output = args.output or args.input_xlsx.with_name(
        args.input_xlsx.stem + "_forces.xlsx")

    if not args.input_xlsx.is_file():
        print(f"error: input file not found: {args.input_xlsx}", file=sys.stderr)
        return 1

    # The one output path that must never be allowed, regardless of how it
    # was spelled (relative vs. absolute, a redundant `-o` restating the
    # default): it would silently replace the vehicle description with a
    # results file of the same name.
    if output.resolve() == args.input_xlsx.resolve():
        print(f"error: output path {output} is the same file as the input; "
              f"pass -o to write somewhere else.", file=sys.stderr)
        return 1

    try:
        cfg = load_vehicle_xlsx(args.input_xlsx)
        matrix = compute_force_matrix(cfg, include_bump=not args.no_bump,
                                      include_combined=not args.no_bump)
    except (VehicleXlsxError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if matrix.failures:
        print("warning: some cases did not compute:", file=sys.stderr)
        for key, why in matrix.failures.items():
            print(f"  {key}: {why}", file=sys.stderr)

    lbf = args.units == "lbf"
    if output.suffix.lower() == ".csv":
        write_csv(matrix, output, lbf=lbf, frame=args.frame)
    else:
        write_workbook(matrix, output, lbf=lbf, frame=args.frame)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# forcesSim

A terminal tool that reads one vehicle workbook -- front and rear suspension geometry plus every vehicle parameter the model needs -- and writes the 3D force on every suspension component (upright, wishbones, rods, hub, rocker) at every attachment point.

Two entry points, same input workbook:

- **`run_sim.py`** -- fast. The four steady-state structural cases (braking, acceleration, inside/outside cornering) at all four corners, one flat CSV out.
- **`export_workbook.py`** -- slower (~30-60 s, a 2-DOF quarter-car bump simulation runs at 11 speeds per axle), but complete: adds the bump case, every "bump on top of a steady case" combination, the swept-bump envelope, and writes the team's multi-tab xlsx (or an equivalent long-format CSV) -- one sheet per component, an Envelope sheet (worst case per attachment point across everything, including the swept bump), and a Key sheet decoding every attachment-point code.

## Quickstart

Requires Python 3.9 or newer. Sets up a virtual environment, installs the runtime dependencies (numpy, scipy, pyyaml, openpyxl), and runs the shipped NFR27 example.

**macOS / Linux** (Terminal):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python run_sim.py NFR27_vehicle.xlsx
```

**Windows** (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python run_sim.py NFR27_vehicle.xlsx
```

> If PowerShell blocks the activate script with an execution-policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or activate with `.venv\Scripts\activate.bat` from `cmd.exe` instead.

Either way, this writes `NFR27_vehicle_forces.csv` next to the input (the default output name is always the input filename's stem plus `_forces.csv`). Next time, you only need to activate the venv (`source .venv/bin/activate` / `.venv\Scripts\Activate.ps1`) before running -- no need to reinstall. General form:

```bash
python run_sim.py <your_vehicle.xlsx> [-o output.csv]
```

Pass `-o`/`--output` to name the output file yourself instead of taking the default.

For the full workbook (bump case, envelope, Key sheet -- see above), run `export_workbook.py` on the same input instead:

```bash
python export_workbook.py NFR27_vehicle.xlsx
```

This writes `NFR27_vehicle_forces.xlsx` next to the input (same `_forces` naming as `run_sim.py`, not just the input's stem -- the input is itself an .xlsx now, and reusing its name with only the suffix unchanged would silently overwrite it the first time someone forgot `-o`; the tool refuses outright if `-o` is ever pointed back at the input). `-o output.xlsx` names it yourself; `-o output.csv` writes the long-format CSV equivalent instead of an xlsx. `--units N` reports newtons instead of the default lbf; `--frame iso8855` reports in the model's internal frame instead of the default SolidWorks triad. `--no-bump` skips the bump case entirely for a ~100x faster run when you only need the four steady cases in this richer layout.

## The input workbook format

One `.xlsx` file, two sheets. Column order within a sheet does not matter -- both are read by header name, from row 1. A row with a blank `Name` is ignored (use it for a spacer).

### `Geometry` sheet

Columns: `Axle, Name, X_mm, Y_mm, Z_mm, Notes`. One row per suspension hardpoint.

Coordinates are millimetres, in the SolidWorks convention this project has always used: origin at the wheelbase midpoint / vehicle centreline / ground plane, `-X` = right (lateral), `+Y` = up, `+Z` = forward. Only the **right side** is entered -- FL/RL are generated automatically by mirroring the front/rear sheets about the centreline.

Which point names are required depends on the suspension topology:

| Topology | Actuation | Required points |
|---|---|---|
| `double_wishbone` | `pushrod` | `IB_UppLead`, `IB_UppTrail`, `OB_UppPnt`, `IB_LowLead`, `IB_LowTrail`, `OB_LowPnt`, `IB_TiePnt`, `OB_TiePnt`, `IB_Push`, `OB_Push` |
| `five_link` | `pullrod` | `IB_UppLead`, `IB_UppTrail`, `OB_UppPnt`, `IB_LowLead`, `OB_Low_Lead`, `IB_LowTrail`, `OB_Low_Trail`, `IB_TiePnt`, `OB_TiePnt`, `IB_Pull`, `OB_Pull` |

Optional points, added as a whole group or left out entirely:

- `Rocker_Frame`, `Rocker_Axis`, `Rocker_Spring`, `Spring_Chassis` -- rocker pivot. Without these the corner is direct-acting and no `rocker` rows appear in the output.
- `BC_Upper`, `BC_Lower` -- real caliper mount bolts (otherwise synthesized).
- `BPad_Center` -- real brake pad friction centroid (otherwise reconstructed from the caliper mount bolts).
- `IB_Bearing`, `OB_Bearing` -- real wheel bearing seats (otherwise falls back to the `bearings.inner_offset_m`/`outer_offset_m` rows on the `Parameters` sheet, below).

Easiest path for a new car: **copy `NFR27_vehicle.xlsx`'s `Geometry` sheet and replace the numbers** -- it already has both topologies filled in correctly, including every optional group.

The wheel centre point is **not** a workbook input -- it's derived automatically from wheelbase, track width and loaded radius.

### `Parameters` sheet

Columns: `Axle, Name, Value, Unit`. One row per scalar vehicle parameter. `Name` is a dotted key (full list below), `Axle` is blank for a whole-car parameter or `front`/`rear` for one that differs by axle (two rows, same `Name`).

Every row below is required.

| Name | Axle | Unit |
|---|---|---|
| `mass.vehicle_kg` | - | kg (dry mass, no driver) |
| `mass.driver_kg` | - | kg |
| `mass.cg_height_m` | - | m |
| `mass.weight_dist_front` | - | fraction, 0-1 |
| `dimensions.wheelbase_m` | - | m |
| `dimensions.track_front_m` | - | m |
| `dimensions.track_rear_m` | - | m |
| `chassis.torsional_stiffness_Nm_per_deg` | - | N·m/deg |
| `tires.vertical_rate_Npm` | - | N/m |
| `aero.ClA` | - | lift coefficient x area, m² |
| `aero.CdA` | - | drag coefficient x area, m² |
| `aero.balance_front` | - | fraction of downforce on the front axle |
| `aero.cp_height_m` | - | m |
| `aero.air_density_kg_m3` | - | kg/m³ |
| `brakes.bias_front` | - | fraction of brake force at the front axle |
| `brakes.caliper_mount` | - | `upright` or `inboard` |
| `powertrain.architecture` | - | `awd_quad_hub`, `rwd_hub`, `rwd_diff`, `fwd_diff`, or `awd_hybrid` |
| `bearings.inner_offset_m` | - | m (fallback only -- sheet's `IB_Bearing` point wins if present) |
| `bearings.outer_offset_m` | - | m (fallback only -- sheet's `OB_Bearing` point wins if present) |
| `bearings.axially_located` | - | `inner`, `outer`, `split`, or `UNCONFIRMED` |
| `suspension.topology` | front & rear | `double_wishbone` or `five_link` |
| `suspension.actuation` | front & rear | `pushrod` or `pullrod` |
| `suspension.rod_pickup` | front & rear | `upper_wishbone`, `lower_wishbone`, or `upright` |
| `suspension.static_camber_deg` | front & rear | deg (negative = top-in) |
| `suspension.static_toe_deg` | front & rear | deg (sheet's own sign convention) |
| `suspension.spring_rate_Npm` | front & rear | N/m, at the spring |
| `suspension.arb_wheelrate_Npm` | front & rear | N/m at the wheel, 0 if no ARB |
| `mass.unsprung_kg_per_corner` | front & rear | kg, one corner |
| `wheels.loaded_radius_m` | front & rear | m |
| `brakes.rotor_effective_radius_m` | front & rear | m |

`run_sim.py` needs only the rows above. `export_workbook.py` additionally needs these, for the bump case's 2-DOF quarter-car model (`--no-bump` skips them, and skips the check for them):

| Name | Axle | Unit |
|---|---|---|
| `dampers.rate_reference` | - | must be `at_damper` -- the only reference point the model converts from |
| `dampers.knee_velocity_m_per_s` | - | m/s, at the damper, where the digressive damper curve's slope changes |
| `dampers.high_speed.bump_Ns_per_m` | front & rear | N/(m/s), at the damper |
| `dampers.high_speed.rebound_Ns_per_m` | front & rear | N/(m/s), at the damper |
| `dampers.low_speed.bump_Ns_per_m` | front & rear | N/(m/s), at the damper |
| `dampers.low_speed.rebound_Ns_per_m` | front & rear | N/(m/s), at the damper |
| `load_cases.bump.height_m` | - | m, half-sine bump height |
| `load_cases.bump.length_m` | - | m, half-sine bump longitudinal extent |
| `load_cases.bump.speed_sweep_m_s` | - | m/s, **semicolon**-separated list of speeds to sweep, all in one cell (e.g. `4.0;6.0;8.0`) |

A missing row makes either script fail immediately, naming the exact sheet/`Axle`/`Name` you need to add.

## The output CSV format (`run_sim.py`)

Columns: `Corner, Case, Component, Point, Fx, Fy, Fz, Net`.

- **Corner**: `FL`, `FR`, `RL`, `RR`.
- **Case**: one of the four load cases below.
- **Component** / **Point**: which part, and which attachment point on it (table below).
- **Fx, Fy, Fz, Net**: force in **lbf**, in the **ISO 8855** frame (a standard axis convention: x forward, y left, z up). `Net` is the vector magnitude.

### Load cases

| Case | Longitudinal | Lateral | What it represents |
|---|---|---|---|
| `braking` | -1.85 g | 0 | maximum straight-line braking |
| `acceleration` | +1.40 g | 0 | maximum straight-line acceleration/traction |
| `outside_cornering` | -0.24 g | +1.59 g | max cornering, this corner on the OUTSIDE of the turn (heavily loaded) |
| `inside_cornering` | -0.24 g | -1.59 g | max cornering, same corner on the INSIDE of the turn (lightly loaded -- can unload a link) |

### Components and attachment points

| Component | Points |
|---|---|
| `upright` | `UBJ` upper ball joint - `LBJ` lower ball joint (only if the lower arm carries the rod) - `RL`/`FL` rear/front lower link - `TR` tie rod - `BC` brake caliper mount - `OB`/`IB` outer/inner wheel bearing - `UM` unsprung weight + inertia - `PR` pushrod/pullrod (only if it mounts directly on the upright) |
| `wishbone_upper` | `OBJ` outboard ball joint - `FBJ`/`RBJ` front/rear chassis pickup - `PR` rod (only if the rod mounts on this arm) |
| `wishbone_lower` | `RL`/`FL` rear/front link (two-force), or `OBJ`/`FBJ`/`RBJ`/`PR` if the rod mounts on this arm instead |
| `rod` | `OP` outboard point - `IP` inboard point |
| `tie_rod` | `OP` outboard point - `IP` inboard point |
| `hub` | `CP` contact patch - `OB`/`IB` outer/inner wheel bearing - `BR` brake rotor |
| `rocker` | `RD` rod - `SP` spring/damper - `PV` pivot (only present when the corner has a rocker) |

## The workbook (`export_workbook.py`)

Only front/rear are computed, not all four corners -- braking and acceleration are left/right symmetric, and cornering's inside/outside pair IS the left/right distinction (the outside case is the loaded corner of a real turn). This matches the shape of the team's own `Suspension Component Forces` spreadsheet, and the Key sheet's "Load case" section says so explicitly.

Sheets, in order:

- **Key** -- every attachment-point code, load case, and reporting convention, decoded. Read this first.
- One sheet per component (**Uprights**, **Wishbones Upper**, **Wishbones Lower**, **Push-Pullrods**, **Tie Rods**, **Rockers**, **Hubs**) -- a block per axle per headline case (the four steady cases plus `bump`, the single instant that produces the worst CONTACT-PATCH load -- see the Key sheet; it is not the instant that is worst for every member, which is what the next two items are for).
- **Bump + Braking** / **Bump + Outside Cornering** / **Bump + Inside Cornering** / **Bump + Acceleration** -- the bump's vertical load increment superposed on each steady case, every component at once. A bump happens WHILE the car is braking or cornering, which is the one legitimate superposition of two load cases; see the Key sheet.
- **Bump Swept** -- the bump case is a transient, so no single instant is worst for every attachment point. This sheet walks every instant of every speed in `load_cases.bump.speed_sweep_m_s`, alone and on top of each steady case, and reports each point's own worst instant.
- **Envelope** -- worst magnitude per attachment point across everything above (steady cases, bump-on-top combinations, and the swept bump), with the case that produced it. This is the number that sizes a part; the component sheets alone understate several members by 20-30% because they stop at one bump instant.

`write_csv` (`suspension/export.py`) is the same data flattened to one long-format CSV, for a dependency-free fallback or a pivot table -- pass `-o *.csv` to `export_workbook.py` to get it instead of the xlsx.

## What this tool intentionally does NOT do

- **No GUI.** The old Streamlit app is gone. This is a terminal tool, driven entirely by one input workbook and one output file.
- **No straight-line acceleration / gear-ratio simulator.** That was a separate, less-validated feature and was dropped to keep this tool single-purpose. `src/forcessim/core/config.py` still has two leftover functions from that work, `build_tire`/`build_combined_tire` -- they are never called by either entry point and are not reachable through the input workbook (it has no tire-fit fields), and `build_combined_tire` specifically will raise `ModuleNotFoundError` if called directly, since the tire module it depends on was removed as out of scope. Harmless dead code, left in place rather than edited into the validated pipeline file.
- **No automated test suite.** The pytest suite that used to check this pipeline was removed as part of this simplification. If someone changes the physics code under `src/forcessim/`, there is nothing here to catch a regression -- check any change's output against known-good numbers (e.g. a saved copy of `NFR27_vehicle_forces.csv`, or a saved copy of the xlsx) before trusting it.

## Repo layout

- `run_sim.py` -- CLI entry point: the four steady cases, all four corners, one flat CSV out
- `export_workbook.py` -- CLI entry point: the full pipeline (steady cases + bump + envelope), the team's multi-tab xlsx (or long-format CSV) out
- `NFR27_vehicle.xlsx` -- shipped example input, real NFR27 numbers
- `pyproject.toml` -- package name, dependencies, Python version
- `src/forcessim/core/` -- workbook parsing (`vehicle_xlsx.py`) and config-to-dataclass building (`config.py`, including the bump case's `build_quarter_car`/`build_bump_envelope`), plus ISO 8855 frame and unit conventions
- `src/forcessim/suspension/` -- the force-extraction pipeline itself: hardpoint geometry, the corner pose solver, the 6x6 link solver, the per-component/per-point breakdown (`component_forces.py`), the batch layer that runs every case and resolves the envelope (`force_matrix.py`), the workbook/CSV writer (`export.py`), and the attachment-point/case/convention glossary that feeds the Key sheet (`glossary.py`)
- `src/forcessim/tire/` -- fitted tire models (the aligning-moment model feeds the tie-rod's Mz load)
- `src/forcessim/vehicle/` -- vehicle-level load transfer, the four standard structural load cases, and the 2-DOF quarter-car bump model (`quarter_car.py`)

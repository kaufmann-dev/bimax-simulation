# BIMAX

Explore jaw movements in 3D from a binary bone volume. BIMAX is a Linux desktop app for separating the upper and lower jaws, defining geometric cuts, and comparing movement scenarios in slice and 3D views. The headless `bimax-cli` lets AI agents and scripts plan, preview and export from the same projects.

It supports reusable Le Fort I, II and III cuts, editable modified II and III variants, bilateral sagittal split osteotomy (BSSO), and genioplasty. Nested upper-jaw cuts can be combined, such as Le Fort III + I. This is a geometric exploration tool. Bone-interference detection, joint biomechanics, soft-tissue prediction, and clinical feasibility assessment are outside its scope. It does not predict fractures or bite.

## Install

You need Linux with a graphical desktop. BIMAX uses Qt's X11 backend, so an X11 session or XWayland is required. The installer places BIMAX below `~/.local` and does not need `sudo`.

On Fedora, install the prerequisites once:

```sh
sudo dnf install curl desktop-file-utils gtk-update-icon-cache shared-mime-info tar uv
```

Install or update to the latest release:

```sh
curl -fsSL https://raw.githubusercontent.com/kaufmann-dev/bimax-simulation/main/install | bash
```

The installer downloads Python 3.14 if needed, installs BIMAX into its own environment with the versions pinned in `uv.lock`, and adds the `bimax` and `bimax-cli` commands, application launcher, icon, AppStream metadata, and `.bimax` file association. Run the same command again to update after a new release. Confirm that the user copy wins:

```sh
command -v bimax
# Expected: /home/your-user/.local/bin/bimax
```

If the installer warns that `~/.local/bin` is not first in `PATH`, add it to your shell profile and open a new terminal:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

To remove the command, launcher, icon, metadata, and file association:

```sh
curl -fsSL https://raw.githubusercontent.com/kaufmann-dev/bimax-simulation/main/install | bash -s -- --uninstall
```

### Install from a checkout

To install a specific source revision instead, run the installer from a clone. It installs that checkout rather than the latest release, and `./install --uninstall` removes it:

```sh
git clone https://github.com/kaufmann-dev/bimax-simulation.git
cd bimax-simulation
./install
```

## Start BIMAX

Open BIMAX from the application launcher, double-click a `.bimax` project, or start it from a terminal:

```sh
bimax
```

Prepare a 3D NIfTI bone mask (`.nii` or `.nii.gz`) containing only `0` for background and `1` for bone. Raw CT images and DICOM folders need preprocessing before import.

Open a volume directly, or use **Import volume** in the app:

```sh
bimax /path/to/bone.nii.gz
```

## Create a simulation

1. **Orient**: import your bone mask, then align the whole skull with pitch, roll, and yaw. This saved view orientation applies to all scenarios and standardized image exports without changing cuts, movements, or prepared geometry. The separate head reference frame defines movement axes and must be set before placing cuts or movement pivots.
2. **Separate** the skull and upper teeth from the mandible and lower teeth. Paint seeds, preview the separation, correct fused contacts with assignment or barrier tools, then accept it.
3. **Cut** by selecting shared cuts for the active scenario. Edit each cut’s planes once, preview the combination, confirm any new pieces, and accept it. Cut edits affect every scenario using that cut.
4. **Move** the jaws together or adjust each upper-jaw fragment, distal mandible, and chin individually. Set pivots, translations, and rotations; duplicate scenarios to compare alternatives. When BSSO is selected, choose the left or right proximal segment to rotate it independently around a manually picked condyle pivot. These controls have no translation or inherited jaw movement.

The scenario selector above the workflow tabs chooses a saved cut combination and its movements. Segmentation and the seven cut definitions are shared across the project. Select at most one variant per Le Fort level; combined upper-jaw cuts must be nested and leave nonempty pieces. In Le Fort III + I, moving the midface carries the maxilla, and the maxilla control adds relative movement.

Each cut mask and each used combination is prepared once. Switching prepared scenarios loads saved fragment assignments and display meshes, including after reopening a project or evicting geometry from memory. A new combination needs preparation once; editing a cut invalidates only combinations using it. Anatomy or reference changes invalidate all dependent geometry.

Press **F11** to expand the 3D view and **1** for the right-profile view. Use **Ctrl+Z** to undo and **Ctrl+Shift+Z** to redo edits.

Drag in the 3D view to orbit while keeping the head upright. Middle-drag pans and the mouse wheel zooms.

## Proximal rotations and symmetry

In **Move**, select **Left proximal segment** or **Right proximal segment** after selecting and reviewing BSSO. Pick a point on that segment's condyle to place its visible pivot, then adjust pitch, roll, and yaw. The pivot is a geometric approximation of the joint center and stays fixed in baseline space. Only that proximal fragment and its attached landmarks rotate. Linked-jaw, distal-mandible, and chin movements do not carry the proximal segments. **Reset selected movement** zeros the rotation while retaining the pivot; **Reset pivot** clears both. Rotation-only edits reuse prepared geometry.

Use the **Symmetry measurements** controls to create, edit, or delete named landmarks and measurement definitions. Landmarks store a baseline planning point and its fragment identity. Pick a named landmark on a fragment or edit its R/A/S coordinates, then reference those names in a measurement. Baseline and moved results appear together.

First set an explicit reference plane using **Set reference offset** or **Set reference at next pick**. The plane is parallel to the planning Anterior/Superior plane at the selected Right coordinate; the scan origin is never assumed to be the anatomical midline. The reference is shared across scenarios.

- **Midline deviation** reports `Right − reference Right` in millimetres, positive toward the right, plus its absolute distance.
- **Paired symmetry** reports left minus right for Superior height, Anterior position, and absolute lateral distance from the reference plane, in millimetres.
- **Ramus inclination** uses a condyle and jaw-angle landmark on each side. Attach each to its corresponding proximal segment or the distal mandible. The vector runs from jaw angle to condyle. Frontal inclination is `atan2(outward lateral, Superior)`; sagittal inclination is `atan2(Anterior, Superior)`, in degrees. Positive angles indicate outward or anterior inclination. Differences are left minus right.

Measurements use the planning frame and are independent of camera position and whole-skull display orientation. Missing or unsuitable landmark references, coincident points, and undefined inclination projections produce explicit errors. Definitions are never silently retargeted: update or remove them before deleting referenced landmarks or removing their fragments from the selected cuts. Shared cut edits retain landmarks referenced by measurements; review their placement after changing the geometry.

## Save and export

**Save** (`Ctrl+S`) stores the volume, anatomy edits, shared cuts, scenarios, landmarks, symmetry measurements, and prepared geometry in a portable `.bimax` project. Reopen it with **Open project** or:

```sh
bimax /path/to/simulation.bimax
```

Projects use `.bimax` format version 5 and persist symmetry references, measurement definitions, proximal pivots and rotations. Earlier formats are rejected; this app does not provide a migration path. Undo history and background jobs are transient.

Overwriting an existing project creates a timestamped `.backup-<UTC timestamp>.bimax` copy first. Saves use atomic replacement and reject external changes detected since the project was opened or saving began. Use another filename for Save As when resolving a conflict.

**Export fragment labelmaps** writes separate binary `.nii.gz` files on a shared physical grid, plus a `manifest.json` describing the fragments and transforms. Review and accept the active cut combination, and assign or explicitly exclude all bone voxels before exporting.

**Export scenario images** writes right, left, front and bird's-eye PNGs plus a labeled contact sheet for the active scenario. Bird's-eye looks down toward the face from 40° above the front. The images use the visible moved bone surfaces and the corrected head reference frame; they do not model soft-tissue appearance.

**Export scenario report (PDF)** writes a landscape A4 plan summary of the active scenario for surgeons and planners, set in black with the bundled Inter font. Before writing it asks for optional patient name, date of birth, case ID, surgeon and notes. These are printed in the page headers but never saved in the project. The overview page shows before and after right and front views next to the planned movements. Each segment is one row, indented under the segment it moves with, and the columns are AP (+ forward), vertical (+ down) and transverse (+ right) in millimetres, then pitch (+ counterclockwise), roll (+ right side down) and yaw (+ front to the left) in degrees, from the patient's perspective. Landmark movement and symmetry measurements follow. Tables that don't fit continue on a measurements page. Two more pages show before and after for right, front, left, both front obliques and inferior, each pair with the same camera, scale and crop. The combination must be accepted first. Images show the visible structures, and the report names any hidden ones.

**Export aligner movements** writes the active scenario's jaw movements as `movements.toml` for the [aligners](https://github.com/kaufmann-dev/opensource-aligners) app, ready to replace the `[surgery.maxilla]` and `[surgery.mandible]` tables of its `plan.toml` (with `[surgery] mode = "manual"`).
- **Input:** the `out/surgery/landmarks.json` of the aligners case. It holds the dental landmarks (incisor midline, canines, first molars) and the aligners head frame. The bone volume must come from the same CT.
- **Which fragment moves each landmark:** maxilla landmarks move with the Le Fort fragment carrying the upper teeth, or stay with the fixed skull when no Le Fort cut is selected. Mandible landmarks move with the distal mandible.
- **Values:** each landmark's displacement from the scanned occlusion, in millimetres along the aligners head axes: `forward_mm`, `up_mm` and `left_mm`. Note that `left_mm` is positive toward the patient's left, unlike the report's transverse column.
- **Rejected:** a combination that is not accepted, landmarks farther than 3 mm from their jaw bone, and dentitions split across fragments by a cut.

## Command line for agents and scripts

`bimax-cli` plans, previews and exports from a `.bimax` project without a display or a running app, so an AI agent can drive it from a terminal. List the commands with `bimax-cli --help`, and each command's options with `bimax-cli COMMAND --help`. Every command takes the project path and prints one JSON document on stdout. On failure it prints `{"error": "..."}` and exits with status 1.

- **Inspect:** `inspect` lists scenarios, the seven shared cuts and their geometry, movements, pivots, landmarks, fragments, confirmations and acceptance. `measurements --scenario NAME` adds the symmetry results.
- **Edit:** `create-scenario`, `select-scenario`, `select-cuts`, `replace-cut`, `set-movement`, `reset-pivot`, `set-proximal`, `reset-proximal`, `set-symmetry`, `set-measurement`, `remove-measurement`, `set-orientation`, `set-landmark`, `remove-landmark` and `confirm-fragment`.
  - They apply the desktop app's validation.
  - Scenario-owned edits need `--scenario NAME`.
  - `batch PROJECT edits.json` applies a JSON list of these operations with one save, or none of them if any fails.
- **Review:**
  - `prepare` builds the scenario's fragments and lists empty fragments and missing confirmation points.
  - `preview --out sheet.png` renders the contact sheet (right, left, front, bird's-eye), one 3D view, or an axial, coronal or sagittal slice at `--position` mm.
  - `pick --view V --x X --y Y` returns the fragment and baseline point at normalized top-left image coordinates, also within a contact-sheet tile. Use it for confirmation points, pivots and landmarks.
  - `accept` checks and accepts the combination.
  - Unaccepted previews are marked `unreviewed`.
- **Export:** `export-labelmaps`, `export-images`, `export-report` and `export-aligners` write the same files as the desktop exports. Case details for the report are passed as options and never saved.

Points are baseline planning Right/Anterior/Superior millimetres. Rotations are pitch, roll and yaw in degrees, applied as `Rz(yaw) Ry(roll) Rx(pitch)` about the pivot. Distal movements keep their parent-relative inheritance; proximal rotations have none.

Edits save the project in place, atomically and without backups; copy the project first if you want to keep a version. A save is refused if the file changed since the command read it. The desktop app likewise refuses to overwrite a project the CLI changed while it was open, so reopen it to see the changes. Previews and exports do not change the project. Images render offscreen, through EGL when there is no display server.

## Run from source

From a checkout, run the current source without installing it:

```sh
uv sync --frozen
uv run bimax
```

## Publish a release

Set the new version in `pyproject.toml` and the first `<release>` in `packaging/linux/app.bimax.simulator.metainfo.xml`, then run `uv lock`. After the commit is on `main`, tag and push it:

```sh
git tag v0.2.0
git push origin v0.2.0
```

The Release workflow verifies that the tag matches both versions and publishes a GitHub release. The piped installer always installs the latest release.

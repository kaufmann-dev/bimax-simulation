# BIMAX

Explore jaw movements in 3D from a binary bone volume. BIMAX is a Linux desktop app for separating the upper and lower jaws, defining geometric cuts, and comparing movement scenarios in slice and 3D views.

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

The installer downloads Python 3.12 if needed, installs BIMAX into its own environment with the versions pinned in `uv.lock`, and adds the `bimax` and `bimax-mcp` commands, application launcher, icon, AppStream metadata, and `.bimax` file association. Run the same command again to update after a new release. Confirm that the user copy wins:

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

Projects use `.bimax` format version 5 and persist symmetry references, measurement definitions, proximal pivots and rotations. Earlier formats are rejected; this app does not provide a migration path. Editing sessions, jobs, and preview IDs are transient.

Overwriting an existing project creates a timestamped `.backup-<UTC timestamp>.bimax` copy first. Saves use atomic replacement and reject external changes detected since the project was opened or saving began. Use another filename for Save As when resolving a conflict.

**Export fragment labelmaps** writes separate binary `.nii.gz` files on a shared physical grid, plus a `manifest.json` describing the fragments and transforms. Review and accept the active cut combination, and assign or explicitly exclude all bone voxels before exporting.

**Export scenario images** writes right, left, front and bird's-eye PNGs plus a labeled contact sheet for the active scenario. Bird's-eye looks down toward the face from 40° above the front. The images use the visible moved bone surfaces and the corrected head reference frame; they do not model soft-tissue appearance.

## Local MCP control

Configure an MCP client with the installed stdio server. The official Python SDK is pinned to `mcp==2.3.0`; diagnostics go to stderr. For this checkout's installation:

```json
{
  "mcpServers": {
    "bimax-simulation": {
      "command": "/home/david/.local/bin/bimax-mcp"
    }
  }
}
```

On another machine, replace the command with that user's absolute `~/.local/bin/bimax-mcp` path. BIMAX and the MCP client must run as the same user in the same graphical desktop environment. The private Unix socket lives under `$XDG_RUNTIME_DIR/bimax` (or `/run/user/<uid>/bimax`); its directory is owner-only and socket permissions are 0600.

The server discovers running instances, attaches when exactly one exists, and launches the installed `~/.local/bin/bimax` when none exists. When several instances exist, call `discover_instances` and choose an `instance_id` with `connect_instance`.

A typical interaction is: connect → inspect state → begin a session → open an absolute version 5 `.bimax` path → edit a named scenario → preview → revise → explicitly save → end the session. Mutations require the current `session_id`, `document_id`, and `revision`. Scenario-owned edits require explicit scenario names. Stale commands are rejected; inspect state to get fresh guards.

Typed tools cover scenarios, shared cut replacement, atomic movements, independent proximal rotations and resets, orientation, named landmark creation/update/removal, the symmetry reference, measurement definitions and baseline/moved results, fragment confirmations, explicit combination acceptance, undo, redo, opening and saving. Public points use baseline planning Right/Anterior/Superior coordinates in millimetres. Rotations use named pitch, roll and yaw in degrees, in engine order `Rz(yaw) Ry(roll) Rx(pitch)`. Distal movements retain parent-relative inheritance; proximal rotations have none.

`preview` returns a job ID immediately. Poll `job_status` for progress and the real PNG image content, or use `cancel_job`. The default 1024-pixel contact sheet contains right, left, front and bird's-eye views; individual views and axial, coronal or sagittal slices are available. Slice positions are in millimetres along the planning normal rotated by the saved display orientation. `pick_image_point` takes a preview ID, view name and normalized top-left image coordinates and returns a fragment identity and baseline point. For a contact sheet, coordinates are relative to its individual view. Picking rejects old document revisions.

Provisional movement previews are marked **unreviewed** and do not accept cuts. Supply confirmation points and call `accept_combination` explicitly. Saving a provisional scenario persists its edits without accepting it; provisional display geometry remains transient. Manual image exports remain 2048 pixels per view.

During a session, the **AI editing** banner shows the current scenario. Project changes, painting, planning controls, saving, undo and redo are read-only; navigation, slice browsing, zoom and visibility inspection remain available. **Take control** revokes the session and cancels active work while preserving completed edits and undo history. Explicit session completion, disconnect, shutdown, or five minutes without a request or active job releases the lock. Releasing it never saves automatically. Saving requires an explicit absolute filename; opening another project is refused while unsaved changes exist.

## Run from source

From a checkout, run the current source without installing it:

```sh
uv sync --frozen
uv run bimax
```

## Publish a release

Set the new version in `pyproject.toml` and the first `<release>` in `packaging/linux/app.bimax.simulator.metainfo.xml`. After the commit is on `main`, tag and push it:

```sh
git tag v0.1.0
git push origin v0.1.0
```

The Release workflow verifies that the tag matches both versions and publishes a GitHub release. The piped installer always installs the latest release.

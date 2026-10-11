# BIMAX

Explore jaw movements in 3D from a binary bone volume. BIMAX is a Linux desktop app for separating the upper and lower jaws, defining geometric cuts, and comparing movement scenarios in slice and 3D views.

It supports Le Fort I, II and III cut plans, bilateral sagittal split osteotomy (BSSO), and genioplasty. This is a geometric simulation: it does not predict fractures, bite, or soft-tissue behavior.

## Install for your user

You need Linux with a graphical desktop. BIMAX uses Qt's X11 backend, so an X11 session or XWayland is required. The installer places BIMAX below `~/.local` and does not need `sudo`.

On Fedora, install the prerequisites once:

```sh
sudo dnf install desktop-file-utils git-core gtk-update-icon-cache shared-mime-info uv
```

Clone the project, then install or replace the current user copy:

```sh
git clone https://github.com/kaufmann-dev/bimax-simulation.git
cd bimax-simulation
./install
```

The installer downloads Python 3.12 if needed, installs BIMAX into its own environment with the versions pinned in `uv.lock`, and adds the `bimax` command, application launcher, icon, AppStream metadata, and `.bimax` file association. The checkout is not needed at runtime. For a later update, pull the desired source revision and run `./install` again. Confirm that the user copy wins:

```sh
command -v bimax
# Expected: /home/your-user/.local/bin/bimax
```

If the installer warns that `~/.local/bin` is not first in `PATH`, add it to your shell profile and open a new terminal:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

To remove the command, launcher, icon, metadata, and file association, run:

```sh
./install --uninstall
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

1. **Import** your bone mask and adjust the head reference frame before placing cuts or movement pivots.
2. **Separate** the skull and upper teeth from the mandible and lower teeth. Paint seeds, preview the separation, correct fused contacts with assignment or barrier tools, then accept it.
3. **Cut** the anatomy using a cut plan. Adjust the planes, preview the fragments, confirm the required pieces, and accept the result.
4. **Move** the jaws together or adjust the upper jaw, distal mandible, and chin individually. Set pivots, translations, and rotations; duplicate scenarios to compare alternatives. The BSSO proximal pieces remain fixed.

Scenarios can share a cut plan. Create an independent cut plan when you want to compare different cuts rather than just different movements.

Press **F11** to expand the 3D view and **1** for the right-profile view. Use **Ctrl+Z** to undo and **Ctrl+Shift+Z** to redo edits.

Drag in the 3D view to orbit while keeping the head upright. Middle-drag pans and the mouse wheel zooms.

## Save and export

**Save** (`Ctrl+S`) stores the volume, anatomy edits, cut plans, and scenarios in a portable `.bimax` project. Reopen it with **Open project** or:

```sh
bimax /path/to/simulation.bimax
```

**Export fragment labelmaps** writes separate binary `.nii.gz` files on a shared physical grid, plus a `manifest.json` describing the fragments and transforms. Review and accept the active cut plan, and assign or explicitly exclude all bone voxels before exporting.

**Export scenario images** writes right, left, front and bird's-eye PNGs plus a labeled contact sheet for the active scenario. Bird's-eye looks down toward the face from 40° above the front. The images use the visible moved bone surfaces and the corrected head reference frame; they do not model soft-tissue appearance.

## Run from source

From a checkout, run the current source without installing it:

```sh
uv sync --frozen
uv run bimax
```

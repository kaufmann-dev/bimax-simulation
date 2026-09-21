# BIMAX

Explore jaw movements in 3D from a binary bone volume. BIMAX is a Linux desktop app for separating the upper and lower jaws, defining geometric cuts, and comparing movement scenarios in slice and 3D views.

It supports Le Fort I, II and III cut plans, bilateral sagittal split osteotomy (BSSO), and genioplasty. This is a geometric simulation: it does not predict fractures, bite, or soft-tissue behavior.

## Get started

You need Linux with a graphical desktop, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The project requires Python 3.12; uv can install it and the project dependencies automatically. The app uses Qt's X11 backend, so an X11 session or XWayland is required.

```sh
git clone https://github.com/kaufmann-dev/bimax-simulation.git
cd bimax-simulation
uv sync --frozen
./bimax
```

Prepare a 3D NIfTI bone mask (`.nii` or `.nii.gz`) containing only `0` for background and `1` for bone. Raw CT images and DICOM folders need preprocessing before import.

Open a volume directly, or use **Import volume** in the app:

```sh
./bimax /path/to/bone.nii.gz
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
./bimax /path/to/simulation.bimax
```

**Export fragment labelmaps** writes separate binary `.nii.gz` files on a shared physical grid, plus a `manifest.json` describing the fragments and transforms. Review and accept the active cut plan, and assign or explicitly exclude all bone voxels before exporting.

**Export scenario images** writes right, left, front and bird's-eye PNGs plus a labeled contact sheet for the active scenario. Bird's-eye looks down toward the face from 40° above the front. The images use the visible moved bone surfaces and the corrected head reference frame; they do not model soft-tissue appearance.

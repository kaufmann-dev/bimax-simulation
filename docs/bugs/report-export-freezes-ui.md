# Scenario report export freezes the UI

- Fixed: 2026-10-15 12:22:28 CEST (+0200)
- Commit before fix: 05f8b27a6888a1f9c621ad228339b175ad248ca7

## Symptom

Clicking **Export scenario report (PDF)…** on a real project (622 × 672 × 615 voxels) made GNOME show "BIMAX is not responding". The export eventually finished and wrote the PDF.

## Root cause

`report.report_data` ran on the Qt main thread and computed fragment centroids with `scipy.ndimage.center_of_mass` over the full label volume (about 257M voxels), plus two full-volume `label_counts` passes for fragment and anatomy volumes. Profiling on the real project: `center_of_mass` 11.8 s, each label count about 0.95 s, data collection about 15 s in total. Rendering the 14 views took 0.4 s and PDF writing 0.7 s.

## Changes

- The report was rewritten as a surgeon-facing summary: operations, segment movements in surgical terms, landmark movement, symmetry, and before/after images. Fragment volumes, centroids, segmentation counts, matrices and cut geometry were removed, so report data no longer touches any voxel array.
- Measured after the change on the same project: data collection 0.01 s, and the whole export about 1 s.
- Lesson: anything in an export that runs on the main thread must not scan full volumes. Heavy array work belongs in the `Jobs` worker.

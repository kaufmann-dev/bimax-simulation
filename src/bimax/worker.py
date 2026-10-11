"""Private subprocess entry point, never a network service."""

import json
import pickle
import sys
from pathlib import Path


def anatomy_counts(labels):
    """Count labels off the GUI thread with bounded temporary allocations."""
    from .engine import label_counts

    return label_counts(labels, 4)


def execute(operation, doc, options):
    from .engine import default_cuts, load_volume, separate, validate_fragment_preview
    from .prepared import anatomy_display, load_geometry, prepare_geometry
    from .storage import export_labelmaps, load_project, save_project
    from .surface_cache import cached_surfaces

    def progress(fraction, message):
        print("BIMAX_PROGRESS " + json.dumps({"message": message, "fraction": fraction}), flush=True)

    def surfaces(labels):
        return cached_surfaces(labels, doc.geometry, progress=lambda message: progress(0, message))

    if operation == "load_prepared":
        try:
            return load_geometry(**options)
        except (OSError, ValueError, KeyError):
            return {"needs_preparation": True, "configuration_key": options["key"], "document_id": options["document_id"]}
    if operation in ("import", "open"):
        doc = load_volume(options["path"], progress=progress) if operation == "import" else load_project(options["path"])
        if not doc.cuts:
            doc.cuts = default_cuts(doc)
        meshes, counts = anatomy_display(doc, progress)
        surface_sets = {False: meshes}
        if doc.cuts_accepted:
            result = prepare_geometry(doc, progress)
            validate_fragment_preview(doc, result["fragments"], result["fragment_counts"], counts)
            doc.fragments, doc.fragments_key = result["fragments"], doc.configuration_key
            surface_sets[True] = result["surfaces"]
        return {"document": doc, "surface_sets": surface_sets,
                "fragment_mode": doc.fragments is not None, "path": options["path"], "anatomy_counts": counts}
    if operation == "separate":
        labels = separate(doc, roi=options.get("roi"), progress=progress)
        return {"proposal": labels, "surfaces": surfaces(labels), "anatomy_counts": anatomy_counts(labels)}
    if operation in ("cuts_preview", "prepare_scene"):
        if operation == "prepare_scene" and not doc.cuts_accepted:
            raise ValueError("Preview and accept the selected cuts before movement.")
        result = prepare_geometry(doc, progress)
        if operation == "prepare_scene":
            validate_fragment_preview(doc, result["fragments"], result["fragment_counts"], result["anatomy_counts"])
        return result
    if operation == "surfaces":
        fragment_mode = doc.fragments is not None and not options.get("anatomy", False)
        if fragment_mode:
            result = prepare_geometry(doc, progress)
            return {"surfaces": result["surfaces"], "fragment_mode": True,
                    "anatomy_counts": result["anatomy_counts"], "configuration_key": doc.configuration_key,
                    "artifacts": doc.artifacts}
        meshes, counts = anatomy_display(doc, progress)
        return {"surfaces": meshes, "fragment_mode": False, "anatomy_counts": counts,
                "configuration_key": doc.configuration_key, "artifacts": doc.artifacts,
                "anatomy_fingerprint": {"revision": doc.anatomy_revision, "digest": doc._anatomy_digest},
                "document_id": doc._identity}
    if operation == "component":
        from .engine import component_indices
        return {"indices": component_indices(doc, options["xyz"]), "label": options["label"]}
    if operation == "save":
        save_project(doc, options["path"])
        return {"path": options["path"], "artifacts": doc.artifacts}
    if operation == "export":
        return {"path": str(export_labelmaps(doc, options["path"])), "artifacts": doc.artifacts}
    raise ValueError(f"Unknown computation: {operation}")


def main():
    path = Path(sys.argv[1])
    try:
        with (path / "input.pickle").open("rb") as stream:
            operation, document, options = pickle.load(stream)
        result = execute(operation, document, options)
    except Exception as error:
        result = {"error": str(error)}
    with (path / "result.pickle").open("wb") as stream:
        pickle.dump(result, stream, protocol=5)


if __name__ == "__main__":
    main()

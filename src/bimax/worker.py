"""Private subprocess entry point, never a network service."""

import json
import pickle
import sys
from pathlib import Path


def execute(operation, doc, options):
    from .engine import build_fragments, default_cuts, load_volume, separate
    from .storage import export_labelmaps, load_project, save_project
    from .surfaces import make_surfaces

    def progress(fraction, message):
        print(
            "BIMAX_PROGRESS " + json.dumps({"message": message, "fraction": fraction}), flush=True
        )

    def surfaces(labels):
        return make_surfaces(labels, doc.geometry, progress=lambda message: progress(0, message))

    if operation in ("import", "open"):
        doc = (
            load_volume(options["path"], progress=progress)
            if operation == "import"
            else load_project(options["path"])
        )
        if not doc.cuts:
            doc.cuts = default_cuts(doc)
        labels = doc.fragments if doc.fragments is not None else doc.anatomy
        return {
            "document": doc,
            "surfaces": surfaces(labels),
            "fragment_mode": doc.fragments is not None,
            "path": options["path"],
        }
    if operation == "separate":
        labels = separate(doc, roi=options.get("roi"), progress=progress)
        return {"proposal": labels, "surfaces": surfaces(labels)}
    if operation in ("cuts_preview", "cuts_accept"):
        labels = build_fragments(doc, validate=operation == "cuts_accept", progress=progress)
        return {"fragments": labels, "surfaces": surfaces(labels)}
    if operation == "surfaces":
        fragment_mode = doc.fragments is not None and not options.get("anatomy", False)
        labels = doc.fragments if fragment_mode else doc.anatomy
        return {
            "surfaces": surfaces(labels),
            "fragment_mode": fragment_mode,
        }
    if operation == "component":
        from .engine import component_indices

        return {"indices": component_indices(doc, options["xyz"]), "label": options["label"]}
    if operation == "save":
        save_project(doc, options["path"])
        return {"path": options["path"]}
    if operation == "export":
        return {"path": str(export_labelmaps(doc, options["path"]))}
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

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
    from copy import deepcopy

    from .engine import build_fragments, default_cuts, label_counts, load_volume, separate
    from .model import CutPlan
    from .storage import export_labelmaps, load_project, save_project
    from .surface_cache import cached_surfaces

    def progress(fraction, message):
        print(
            "BIMAX_PROGRESS " + json.dumps({"message": message, "fraction": fraction}), flush=True
        )

    def surfaces(labels):
        return cached_surfaces(labels, doc.geometry, progress=lambda message: progress(0, message))

    if operation in ("import", "open"):
        doc = (
            load_volume(options["path"], progress=progress)
            if operation == "import"
            else load_project(options["path"])
        )
        if not doc.plan.cuts:
            doc.plan.cuts = default_cuts(doc, doc.plan.lefort_level)
        progress(0, "Building anatomy display")
        surface_sets = {False: surfaces(doc.anatomy)}
        if doc.plan.accepted:
            progress(0, f"Preparing {doc.plan.name}")
            doc.fragments = build_fragments(doc, validate=True, progress=progress)
            doc.fragments_key = doc.configuration_key
            progress(0, "Building fragment display")
            surface_sets[True] = surfaces(doc.fragments)
        return {
            "document": doc,
            "surface_sets": surface_sets,
            "fragment_mode": doc.fragments is not None,
            "path": options["path"],
            "anatomy_counts": anatomy_counts(doc.anatomy),
        }
    if operation == "separate":
        labels = separate(doc, roi=options.get("roi"), progress=progress)
        return {
            "proposal": labels,
            "surfaces": surfaces(labels),
            "anatomy_counts": anatomy_counts(labels),
        }
    if operation == "cuts_preview":
        labels = build_fragments(doc, validate=False, progress=progress)
        return {
            "fragments": labels,
            "surfaces": surfaces(labels),
            "fragment_counts": label_counts(labels, 8),
            "configuration_key": doc.configuration_key,
        }
    if operation == "prepare_plan":
        if not doc.plan.accepted:
            raise ValueError("Preview and accept this cut plan before movement.")
        labels = build_fragments(doc, validate=True, progress=progress)
        return {
            "fragments": labels,
            "surfaces": surfaces(labels),
            "configuration_key": doc.configuration_key,
        }
    if operation == "create_plan":
        level = options["level"]
        cuts = (
            deepcopy(doc.plan.cuts) if level == doc.plan.lefort_level else default_cuts(doc, level)
        )
        for operation_key in ("bsso", "genio"):
            cuts[operation_key] = deepcopy(doc.plan.cuts[operation_key])
        return {
            "plan_id": options["plan_id"],
            "plan": CutPlan(options["name"], lefort_level=level, cuts=cuts),
        }
    if operation == "surfaces":
        fragment_mode = doc.fragments is not None and not options.get("anatomy", False)
        labels = doc.fragments if fragment_mode else doc.anatomy
        return {
            "surfaces": surfaces(labels),
            "fragment_mode": fragment_mode,
            "anatomy_counts": anatomy_counts(doc.anatomy),
            "configuration_key": doc.configuration_key,
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

"""CUDA VQA, fresh APS calibration, and paired retrieval experiments."""

import argparse
import contextlib
import json
import sys

from .common import Query, Settings, write_json


def parser():
    root = argparse.ArgumentParser(description="Minimal CUDA generate-then-retrieve VQA")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-images", help="download and validate MMBench images (CPU)"
    )
    prepare.add_argument("--data", default="mmbench.pkl")
    prepare.add_argument("--output", default="artifacts/rag/images")
    from .data import SOURCE_REVISION

    prepare.add_argument("--source-revision", default=SOURCE_REVISION)
    for name, help_text in [
        ("build-index", "create leakage-safe split, descriptions, and CLIP vectors"),
        ("answer", "answer an image question using three retrieved examples"),
        ("evaluate", "evaluate every held-out record, or a limited smoke run"),
        ("calibrate-aps", "calibrate fresh draft scores on held-out image groups"),
        ("compare-retrieval", "compare draft and APS candidate queries on separate test groups"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--data", default="mmbench.pkl")
        command.add_argument("--manifest", default="artifacts/rag/images/manifest.json")
        command.add_argument("--index", default="artifacts/rag/index")
        command.add_argument("--llava-model", default=Settings.llava_model)
        command.add_argument("--clip-model", default=Settings.clip_model)
        command.add_argument(
            "--description-max-tokens", type=int, default=Settings.description_max_tokens
        )
        command.add_argument(
            "--description-max-chars", type=int, default=Settings.description_max_chars
        )
        if name == "build-index":
            command.add_argument(
                "--rebuild",
                action="store_true",
                help="replace the index; reuse compatible intermediate caches",
            )
        elif name == "calibrate-aps":
            command.add_argument("--alpha", type=float, default=0.1)
            command.add_argument("--calibration-fraction", type=float, default=0.5)
            command.add_argument("--split-seed", type=int, default=43)
            command.add_argument("--output", default="artifacts/rag/aps/calibration.json")
        elif name == "answer":
            command.add_argument("--image", required=True)
            command.add_argument("--question", required=True)
            command.add_argument("--hint", default="")
            command.add_argument("--choice", action="append", required=True, metavar="A=TEXT")
            command.add_argument("--output", help="also save the answer JSON at this path")
        else:
            command.add_argument("--limit", type=int)
            command.add_argument(
                "--output",
                default=(
                    "results/aps_candidate_retrieval"
                    if name == "compare-retrieval"
                    else "results/generate_then_retrieve"
                ),
            )
        if name in {"answer", "evaluate", "compare-retrieval"}:
            command.add_argument("--calibration", default="artifacts/rag/aps/calibration.json")
        if name in {"answer", "evaluate"}:
            command.add_argument("--retrieval-mode", choices=["draft", "aps"], default="draft")
    return root


def parse_choices(values):
    choices = {}
    for value in values:
        label, separator, answer = value.partition("=")
        label, answer = label.strip().upper(), answer.strip()
        if not separator or label in choices:
            raise ValueError("use unique named choices, e.g. --choice 'A=cat' --choice 'B=dog'")
        choices[label] = answer
    return choices


def run(args):
    from .data import load_manifest, load_records, prepare_images

    records = load_records(args.data)
    if args.command == "prepare-images":
        manifest = prepare_images(records, args.output, args.source_revision)
        return {
            "manifest": args.output + "/manifest.json",
            "records": len(manifest["records"]),
            "unique_images": manifest["unique_images"],
            "alignment": manifest["alignment"],
        }, 0

    from .embeddings import ClipEncoder
    from .index import ReferenceIndex, artifact_identity, assert_compatible, build_index
    from .llava_adapter import LlavaAdapter
    from .pipeline import Pipeline, evaluate

    settings = Settings(
        llava_model=args.llava_model,
        clip_model=args.clip_model,
        description_max_tokens=args.description_max_tokens,
        description_max_chars=args.description_max_chars,
    )
    query = (
        Query(args.question, parse_choices(args.choice), args.hint)
        if args.command == "answer"
        else None
    )
    if (
        args.command in {"evaluate", "compare-retrieval"}
        and args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError("--limit must be positive")
    manifest = load_manifest(args.manifest, records)
    llava, clip = LlavaAdapter(settings), ClipEncoder(settings)
    if args.command == "build-index":
        index = build_index(
            records, manifest, args.manifest, args.index, settings, llava, clip, args.rebuild
        )
        return {
            "index": args.index,
            "reference_examples": len(index.entries),
            "evaluation_records": len(index.metadata["split"]["evaluation_ids"]),
            "fingerprint": index.metadata["fingerprint"],
        }, 0
    index = ReferenceIndex.load(args.index, artifact_identity(records, manifest, settings))
    assert_compatible(
        index.metadata, {"runtime": {"llava": llava.signature(), "clip": clip.signature()}}
    )
    pipeline = Pipeline(settings, llava, clip, index)
    if args.command == "calibrate-aps":
        from .aps_experiment import calibrate

        calibration = calibrate(
            records,
            manifest,
            args.manifest,
            pipeline,
            args.output,
            args.alpha,
            args.calibration_fraction,
            args.split_seed,
        )
        split = calibration.artifact["identity"]["split"]
        return {
            "calibration": args.output,
            "fingerprint": calibration.artifact["fingerprint"],
            "qhat": calibration.qhat,
            "alpha": args.alpha,
            "calibration_groups": len(split["calibration_groups"]),
            "test_groups": len(split["test_groups"]),
            "test_records": len(split["test_ids"]),
        }, 0
    if args.command == "compare-retrieval" or args.retrieval_mode == "aps":
        from .aps import Calibration

        pipeline.calibration = Calibration.load(args.calibration, index, settings)
    if args.command == "compare-retrieval":
        from .aps_experiment import compare

        result = compare(records, manifest, args.manifest, pipeline, args.output, args.limit)
        secondary = result["secondary_all_rotations"]
        return result, int(
            any(secondary[arm]["invalid_final_predictions"] for arm in ["draft_rag", "aps_rag"])
        )
    if args.command == "answer":
        result = pipeline.answer(query, args.image, retrieval_mode=args.retrieval_mode)
        result["index_fingerprint"] = index.metadata["fingerprint"]
        if args.output:
            write_json(args.output, result)
        return result, 0 if result["final"]["valid"] else 1
    result = evaluate(
        records, manifest, args.manifest, pipeline, args.output, args.limit, args.retrieval_mode
    )
    return result, 0 if not result["invalid_final_predictions"] else 1


def main():
    args = parser().parse_args()
    try:
        # LLaVA's loader prints to stdout; reserve stdout for machine-readable JSON.
        with contextlib.redirect_stdout(sys.stderr):
            result, status = run(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return status
    except Exception as exc:
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

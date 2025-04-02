"""Preserve local MMBench text while validating and preparing source images."""

import hashlib
import pickle
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .common import SCHEMA_VERSION, Query, fingerprint, read_json, text, write_json

SOURCE = "HuggingFaceM4/MMBench_dev"
SOURCE_REVISION = "2d902e6698db10c7b5addf059f2b247e2f4c4de9"


@dataclass(frozen=True)
class Record:
    id: int
    index: int
    question: str
    hint: str
    choices: dict
    answer: str

    @property
    def base_id(self):
        return self.index % 1_000_000

    @property
    def answer_text(self):
        return self.choices[self.answer]

    def query(self):
        return Query(self.question, dict(self.choices), self.hint)

    def metadata(self):
        return {
            "id": self.id,
            "index": self.index,
            "question": self.question,
            "hint": self.hint,
            "choices": self.choices,
            "answer": self.answer,
        }


def load_records(path):
    # Like the existing experiment, this accepts only a trusted local pickle.
    with Path(path).open("rb") as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset must be a nonempty list of records")
    records = []
    for row in rows:
        record = Record(
            id=int(row.get("id", row["index"])),
            index=int(row["index"]),
            question=text(row.get("question")),
            hint=text(row.get("hint")),
            choices={k: text(row.get(k)) for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if text(row.get(k))},
            answer=text(row.get("answer")),
        )
        record.query()  # Validate options without ever copying stored logits.
        if record.answer not in record.choices:
            raise ValueError("record {} has an invalid gold answer".format(record.id))
        records.append(record)
    if len({r.id for r in records}) != len(records):
        raise ValueError("record IDs must be unique")
    return records


def dataset_fingerprint(records):
    # Includes every preserved field but excludes the unused historical logits.
    return fingerprint([r.metadata() for r in records])


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", text(value)).split())


def signature(question, hint, answer_text):
    return tuple(normalized(x) for x in (question, hint, answer_text))


def source_signature(row):
    label = row.get("label", row.get("answer"))
    if not isinstance(label, str):
        label = "ABCD"[int(label)] if label is not None and 0 <= int(label) < 4 else ""
    if label not in "ABCD" or len(label) != 1 or not text(row.get(label)):
        raise ValueError("source row has an invalid correct answer")
    return signature(row.get("question"), row.get("hint"), row[label])


class AlignmentError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("image alignment failed: {} mismatches".format(len(report["mismatches"])))


def align_sources(records, source_rows):
    """Use fully validated source order, or uniquely determined content matches.

    The mirror has no IDs. A single positional mismatch disables positional
    assignment for the entire dataset. Duplicate content is never guessed at.
    Source distractors need not equal the locally augmented distractors.
    """
    local = [signature(r.question, r.hint, r.answer_text) for r in records]
    source = []
    errors = []
    for i, row in enumerate(source_rows):
        try:
            source.append(source_signature(row))
        except (ValueError, TypeError) as exc:
            errors.append({"source_row": i, "reason": str(exc)})
            source.append(None)
    if errors:
        raise AlignmentError({"mismatches": errors})
    if len(local) == len(source) and local == source:
        return list(range(len(records))), "validated-position"
    candidates = defaultdict(list)
    for i, key in enumerate(source):
        candidates[key].append(i)
    mapping = []
    for record, key in zip(records, local):
        matches = candidates[key]
        if len(matches) != 1:
            errors.append(
                {
                    "record_id": record.id,
                    "reason": "no content match" if not matches else "ambiguous content match",
                    "candidate_source_rows": matches,
                    "question": record.question,
                }
            )
        else:
            mapping.append(matches[0])
    if errors:
        raise AlignmentError({"mismatches": errors})
    return mapping, "unique-content"


def decoded_image(value):
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError("missing image: {}".format(path))
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_hash(image):
    image = decoded_image(image)
    digest = hashlib.sha256()
    digest.update("RGB:{}x{}:".format(*image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def prepare_from_source(records, source_rows, output, source_metadata):
    """Write a manifest only after all text matches and images are verified."""
    output = Path(output)
    report_path = output / "mismatch-report.json"
    try:
        # HF Dataset column removal avoids decoding thousands of images twice.
        text_rows = (
            source_rows.remove_columns("image")
            if hasattr(source_rows, "remove_columns")
            else source_rows
        )
        mapping, method = align_sources(records, text_rows)
        images_dir = output / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        entries, errors = [], []
        for record, source_row in zip(records, mapping):
            try:
                image = decoded_image(source_rows[source_row]["image"])
                digest = image_hash(image)
                relative = "images/{}.png".format(digest)
                target = output / relative
                # Always write a lossless image; do not trust a stale file.
                image.save(target, format="PNG")
                if image_hash(decoded_image(target)) != digest:
                    raise ValueError("image changed during lossless save")
                entries.append(
                    {
                        "id": record.id,
                        "index": record.index,
                        "source_row": source_row,
                        "image_path": relative,
                        "image_sha256": digest,
                    }
                )
            except Exception as exc:
                errors.append(
                    {"record_id": record.id, "source_row": source_row, "reason": str(exc)}
                )
        if errors:
            raise AlignmentError({"mismatches": errors})
    except AlignmentError as exc:
        write_json(report_path, {"source": source_metadata, **exc.report})
        raise ValueError("{}; see {}".format(exc, report_path)) from exc
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint(records),
        "source": source_metadata,
        "alignment": method,
        "records": entries,
        "unique_images": len({e["image_sha256"] for e in entries}),
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    if report_path.exists():
        report_path.unlink()
    return manifest


def prepare_images(records, output, revision=SOURCE_REVISION):
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "prepare-images needs requirements-rag.txt (datasets and pyarrow)"
        ) from exc
    resolved = HfApi().dataset_info(SOURCE, revision=revision).sha
    dataset = load_dataset(SOURCE, split="train", revision=resolved)
    if len(dataset) != 4377:
        report = {"mismatches": [{"reason": "expected 4377 source rows", "actual": len(dataset)}]}
        write_json(Path(output) / "mismatch-report.json", report)
        raise ValueError("unexpected mirror size; see mismatch-report.json")
    return prepare_from_source(
        records,
        dataset,
        output,
        {
            "dataset": SOURCE,
            "revision": resolved,
            "split": "train",
            "num_rows": len(dataset),
            "hf_fingerprint": dataset._fingerprint,
        },
    )


def load_manifest(path, records):
    manifest = read_json(path)
    content = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
        "fingerprint"
    ) != fingerprint(content):
        raise ValueError("invalid or incompatible image manifest; run prepare-images")
    if manifest["dataset_fingerprint"] != dataset_fingerprint(records):
        raise ValueError("image manifest is stale for this dataset; run prepare-images")
    entries = manifest["records"]
    if len(entries) != len(records) or {(e["id"], e["index"]) for e in entries} != {
        (r.id, r.index) for r in records
    }:
        raise ValueError("manifest record IDs do not match dataset")
    for entry in entries:
        image_path = (Path(path).parent / entry["image_path"]).resolve()
        try:
            image_path.relative_to(Path(path).parent.resolve())
        except ValueError as exc:
            raise ValueError("manifest image path escapes its directory") from exc
    return manifest


def manifest_images(manifest, path):
    return {
        e["id"]: {**e, "path": str(Path(path).parent / e["image_path"])}
        for e in manifest["records"]
    }


def verified_image(entry):
    image = decoded_image(entry["path"])
    if image_hash(image) != entry["image_sha256"]:
        raise ValueError("image hash mismatch: {}".format(entry["path"]))
    return image


def make_split(records, manifest):
    """Union rotations and image duplicates before shuffling indivisible groups."""
    parent = {r.base_id: r.base_id for r in records}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    images = {e["id"]: e["image_sha256"] for e in manifest["records"]}
    image_owner = {}
    for record in records:
        digest = images[record.id]
        if digest in image_owner:
            union(record.base_id, image_owner[digest])
        image_owner[digest] = record.base_id
    groups = {r.id: find(r.base_id) for r in records}
    unique = sorted(set(groups.values()))
    if len(unique) < 2:
        raise ValueError("at least two independent image/question groups are required")
    random.Random(42).shuffle(unique)
    n_reference = max(1, min(len(unique) - 1, int(len(unique) * 0.8)))
    reference_groups = set(unique[:n_reference])
    reference = sorted(r.id for r in records if groups[r.id] in reference_groups)
    evaluation = sorted(r.id for r in records if groups[r.id] not in reference_groups)
    representatives = {}
    for record in sorted(records, key=lambda r: r.id):
        if groups[record.id] in reference_groups:
            representatives.setdefault(record.base_id, record.id)
    return {
        "seed": 42,
        "reference_fraction": 0.8,
        "record_groups": {str(k): v for k, v in sorted(groups.items())},
        "reference_groups": sorted(reference_groups),
        "evaluation_groups": sorted(set(unique) - reference_groups),
        "reference_ids": reference,
        "evaluation_ids": evaluation,
        "representative_ids": sorted(representatives.values()),
    }

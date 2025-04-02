"""Leakage-safe reference selection and small NumPy/JSON retrieval artifacts."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .common import SCHEMA_VERSION, file_hash, fingerprint, read_json, write_json
from .data import dataset_fingerprint, make_split, manifest_images, verified_image
from .embeddings import unit
from .prompts import DESCRIPTION_PROMPT, retrieval_text


def save_vectors(path, vectors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".vectors-", suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_vectors(vectors, count):
    if vectors.ndim != 2 or vectors.shape[0] != count or not count or vectors.shape[1] == 0:
        raise ValueError("index vector dimensions do not match metadata")
    if not np.all(np.isfinite(vectors)) or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, atol=1e-5
    ):
        raise ValueError("index vectors must be finite and unit-normalized")


def assert_compatible(metadata, expected):
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(
            "stale/incompatible index ({}); run build-index --rebuild".format(", ".join(mismatches))
        )


def artifact_identity(records, manifest, settings):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint(records),
        "manifest_fingerprint": manifest["fingerprint"],
        "settings": settings.metadata(),
        "split": make_split(records, manifest),
    }


class ReferenceIndex:
    def __init__(self, vectors, metadata):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata = metadata
        self.entries = metadata["entries"]
        validate_vectors(self.vectors, len(self.entries))
        self.ids = np.array([e["id"] for e in self.entries], dtype=np.int64)
        if len(set(self.ids.tolist())) != len(self.ids):
            raise ValueError("index contains duplicate IDs")
        split = metadata["split"]
        if sorted(self.ids.tolist()) != split["representative_ids"]:
            raise ValueError(
                "index must contain exactly one representative per reference base question"
            )
        if set(self.ids.tolist()) & set(split["evaluation_ids"]):
            raise ValueError("evaluation records leaked into the index")
        evaluation_groups = set(split["evaluation_groups"])
        for entry in self.entries:
            if (
                entry["group_id"] in evaluation_groups
                or entry["group_id"] != split["record_groups"][str(entry["id"])]
            ):
                raise ValueError("evaluation image/question group leaked into the index")

    @classmethod
    def load(cls, directory, expected=None):
        directory = Path(directory)
        metadata = read_json(directory / "metadata.json")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("incompatible index schema; run build-index --rebuild")
        if metadata.get("fingerprint") != fingerprint(
            {k: v for k, v in metadata.items() if k != "fingerprint"}
        ):
            raise ValueError("index metadata checksum mismatch; run build-index --rebuild")
        if expected:
            assert_compatible(metadata, expected)
        if file_hash(directory / "vectors.npy") != metadata["vectors_sha256"]:
            raise ValueError("index vector checksum mismatch; run build-index --rebuild")
        return cls(np.load(directory / "vectors.npy", allow_pickle=False), metadata)

    def retrieve(self, vector, k=3):
        vector = unit(vector)
        if vector.ndim != 1 or vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("query embedding dimension differs from the index")
        scores = self.vectors @ vector
        # lexsort's last key is primary: descending similarity, ascending ID.
        order = np.lexsort((self.ids, -scores))[:k]
        return [{**self.entries[i], "score": float(scores[i])} for i in order]


def build_index(records, manifest, manifest_path, directory, settings, llava, clip, rebuild=False):
    directory = Path(directory)
    identity = artifact_identity(records, manifest, settings)
    existing = directory / "metadata.json"
    if existing.exists() and not rebuild:
        # Fail before expensive model loading if the dataset/config has changed.
        cached = ReferenceIndex.load(directory, identity)
        assert_compatible(
            cached.metadata, {"runtime": {"llava": llava.signature(), "clip": clip.signature()}}
        )
        return cached

    images = manifest_images(manifest, manifest_path)
    seen = set()
    # Verify all groups, including evaluation images, before trusting duplicate isolation.
    for entry in images.values():
        if entry["path"] not in seen:
            verified_image(entry)
            seen.add(entry["path"])

    runtime = {"llava": llava.signature(), "clip": clip.signature()}
    description_identity = {
        "dataset": identity["dataset_fingerprint"],
        "manifest": identity["manifest_fingerprint"],
        "model": runtime["llava"],
        "prompt": DESCRIPTION_PROMPT,
        "settings": settings.metadata(),
    }
    cache_dir = directory / "cache"
    descriptions_path = cache_dir / ("descriptions-" + fingerprint(description_identity) + ".json")
    if descriptions_path.exists():
        descriptions = read_json(descriptions_path)
        if descriptions.get("identity") != description_identity:
            raise ValueError("incompatible description cache")
    else:
        descriptions = {"identity": description_identity, "entries": {}}

    by_id = {r.id: r for r in records}
    entries, vectors = [], []
    representatives = identity["split"]["representative_ids"]
    for position, record_id in enumerate(representatives, 1):
        record = by_id[record_id]
        image = verified_image(images[record_id])
        cached = descriptions["entries"].get(str(record_id))
        digest = images[record_id]["image_sha256"]
        if cached is not None:
            if (
                cached["image_sha256"] != digest
                or not cached["description"]
                or len(cached["description"]) > settings.description_max_chars
            ):
                raise ValueError("invalid description cache entry: {}".format(record_id))
            description = cached["description"]
        else:
            # This API accepts only the image: no question, hint, choices, or gold.
            description = llava.describe(image)
            descriptions["entries"][str(record_id)] = {
                "image_sha256": digest,
                "description": description,
            }
            write_json(descriptions_path, descriptions)
        reference_text = retrieval_text(
            record.question, record.hint, description, record.answer_text
        )
        vector_key = fingerprint(
            {
                "clip": runtime["clip"],
                "settings": settings.metadata(),
                "image_sha256": digest,
                "text": reference_text,
            }
        )
        vector_path = cache_dir / "embeddings" / (vector_key + ".npy")
        if vector_path.exists():
            vector = np.load(vector_path, allow_pickle=False)
            validate_vectors(vector[np.newaxis, :], 1)
        else:
            vector = clip.encode(image, reference_text)
            save_vectors(vector_path, vector)
        vectors.append(vector)
        entries.append(
            {
                "id": record.id,
                "base_id": record.base_id,
                "group_id": identity["split"]["record_groups"][str(record.id)],
                "image_sha256": digest,
                "description": description,
                "question": record.question,
                "hint": record.hint,
                "answer_letter": record.answer,
                "answer_text": record.answer_text,
            }
        )
        print(
            "reference {}/{} (record {})".format(position, len(representatives), record_id),
            file=sys.stderr,
            flush=True,
        )

    matrix = np.stack(vectors)
    metadata = {**identity, "runtime": runtime, "entries": entries}
    index = ReferenceIndex(matrix, metadata)
    save_vectors(directory / "vectors.npy", matrix)
    metadata["vectors_sha256"] = file_hash(directory / "vectors.npy")
    metadata["fingerprint"] = fingerprint(metadata)
    write_json(existing, metadata)
    return index

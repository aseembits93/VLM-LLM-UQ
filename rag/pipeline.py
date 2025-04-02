"""One pipeline shared by interactive CLI answers and held-out evaluation."""

import json
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .common import write_json
from .data import decoded_image, image_hash, manifest_images
from .prompts import (
    candidate_retrieval_text,
    draft_prompt,
    fit_examples,
    parse_draft,
    retrieval_text,
)


class Pipeline:
    def __init__(self, settings, llava, clip, index, calibration=None):
        self.settings, self.llava, self.clip, self.index = settings, llava, clip, index
        self.calibration = calibration
        # Greedy image-only captions can be reused across held-out option rotations.
        self._descriptions = {}

    def prepare_query(self, query, image_path, expected_image_hash=None, include_scores=False):
        """Image-only caption and baseline draft, before any retrieval or gold access."""
        started = time.monotonic()
        result = {
            "question": query.question,
            "hint": query.hint,
            "choices": query.choices,
            "image": str(image_path),
            "description": None,
            "description_cached": False,
            "draft": {"raw": None, "letter": None, "answer_text": None, "valid": False},
            "retrieved": [],
            "used_example_ids": [],
            "dropped_example_ids": [],
            "final": {"letter": None, "answer_text": None, "scores": {}, "valid": False},
            "errors": [],
            "prepared": False,
        }
        stage = "image"
        try:
            image = decoded_image(image_path)
            digest = image_hash(image)
            result["image_sha256"] = digest
            if expected_image_hash is not None and digest != expected_image_hash:
                raise ValueError("query image hash differs from the verified manifest")
            stage = "description"
            result["description_cached"] = digest in self._descriptions
            if digest not in self._descriptions:
                description = self.llava.describe(image)
                if len(self._descriptions) >= 256:
                    self._descriptions.pop(next(iter(self._descriptions)))
                self._descriptions[digest] = description
            description = self._descriptions[digest]
            result["description"] = description
            stage = "draft"
            scores = None
            if include_scores:
                if self.settings.draft_max_tokens != 1:
                    raise ValueError("APS requires the one-token baseline draft")
                raw, scores = self.llava.draft_with_scores(
                    image, draft_prompt(query, description), query.choices
                )
                from .aps import ranked_probabilities

                if set(scores) != set(query.choices):
                    raise ValueError("draft scores differ from supplied choices")
                ranked_probabilities(scores)  # Reject missing/non-finite scores immediately.
            else:
                raw = self.llava.generate(
                    image, draft_prompt(query, description), self.settings.draft_max_tokens
                )
            letter = parse_draft(raw, query.choices)
            answer_text = query.choices[letter] if letter else None
            result["draft"] = {
                "raw": raw,
                "letter": letter,
                "answer_text": answer_text,
                "valid": letter is not None,
            }
            if scores is not None:
                result["draft"]["scores"] = scores
            if letter is None:
                result["errors"].append(
                    {
                        "stage": stage,
                        "type": "MalformedDraft",
                        "message": "draft was not a supplied option letter; draft retrieval omits answer text; APS retrieval uses supplied-option scores",
                    }
                )
            result["prepared"] = True
        except Exception as exc:
            result["errors"].append(
                {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            )
        result["preparation_elapsed_seconds"] = time.monotonic() - started
        result["elapsed_seconds"] = result["preparation_elapsed_seconds"]
        return result

    def answer(
        self, query, image_path, expected_image_hash=None, retrieval_mode="draft", prepared=None
    ):
        if retrieval_mode not in {"draft", "aps"}:
            raise ValueError("retrieval mode must be draft or aps")
        if retrieval_mode == "aps" and self.calibration is None:
            raise ValueError("APS retrieval requires a compatible calibration artifact")
        result = (
            deepcopy(prepared)
            if prepared is not None
            else self.prepare_query(
                query, image_path, expected_image_hash, include_scores=retrieval_mode == "aps"
            )
        )
        result["retrieval_mode"] = retrieval_mode
        if not result["prepared"]:
            return result
        started = time.monotonic()
        stage = "image"
        try:
            image = decoded_image(image_path)
            digest = image_hash(image)
            if (
                digest != result["image_sha256"]
                or (expected_image_hash is not None and digest != expected_image_hash)
                or result["question"] != query.question
                or result["hint"] != query.hint
                or result["choices"] != query.choices
            ):
                raise ValueError("prepared query does not match current image/question/choices")
            description = result["description"]
            stage = "retrieval"
            if retrieval_mode == "aps":
                result["prediction_set"] = self.calibration.predict(
                    result["draft"]["scores"], query.choices
                )
                query_text = candidate_retrieval_text(
                    query.question,
                    query.hint,
                    description,
                    result["prediction_set"]["answer_texts"],
                )
            else:
                query_text = retrieval_text(
                    query.question, query.hint, description, result["draft"]["answer_text"]
                )
            result["retrieval_text"] = query_text
            vector = self.clip.encode(image, query_text)
            retrieved = self.index.retrieve(vector, k=3)
            result["retrieved"] = retrieved
            stage = "context"
            prompt, retained = fit_examples(query, description, retrieved, self.llava.fits)
            result["used_example_ids"] = [r["id"] for r in retained]
            result["dropped_example_ids"] = [r["id"] for r in retrieved[len(retained) :]]
            stage = "final"
            letter, scores = self.llava.score_options(image, prompt, query.choices)
            if letter not in query.choices:
                raise ValueError("final scorer returned an unsupplied option letter")
            result["final"] = {
                "letter": letter,
                "answer_text": query.choices[letter],
                "scores": scores,
                "valid": True,
            }
        except Exception as exc:
            result["errors"].append(
                {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            )
        result["answer_elapsed_seconds"] = time.monotonic() - started
        result["elapsed_seconds"] = (
            result["preparation_elapsed_seconds"] + result["answer_elapsed_seconds"]
        )
        return result


def metrics(rows):
    total = len(rows)
    if not total:
        raise ValueError("evaluation cannot be empty")
    draft_correct = sum(
        r["draft"]["valid"] and r["draft"]["letter"] == r["gold_answer"] for r in rows
    )
    final_correct = sum(
        r["final"]["valid"] and r["final"]["letter"] == r["gold_answer"] for r in rows
    )
    corrected = regressed = 0
    failures = Counter()
    for row in rows:
        draft = row["draft"]["valid"] and row["draft"]["letter"] == row["gold_answer"]
        final = row["final"]["valid"] and row["final"]["letter"] == row["gold_answer"]
        corrected += int(final and not draft)
        regressed += int(draft and not final)
        for error in row["errors"]:
            failures[error["stage"] + ":" + error["type"]] += 1
    return {
        "total_records": total,
        "draft_correct": draft_correct,
        "final_correct": final_correct,
        "draft_accuracy": draft_correct / total,
        "final_accuracy": final_correct / total,
        "accuracy_change": (final_correct - draft_correct) / total,
        "accuracy_change_percentage_points": 100 * (final_correct - draft_correct) / total,
        "corrected_answers": corrected,
        "regressed_answers": regressed,
        "invalid_draft_predictions": sum(not r["draft"]["valid"] for r in rows),
        "malformed_drafts": sum(
            any(e["type"] == "MalformedDraft" for e in r["errors"]) for r in rows
        ),
        "invalid_final_predictions": sum(not r["final"]["valid"] for r in rows),
        "records_with_failures": sum(bool(r["errors"]) for r in rows),
        "failure_counts": dict(sorted(failures.items())),
        "records_with_dropped_examples": sum(bool(r["dropped_example_ids"]) for r in rows),
        "elapsed_seconds": sum(r["elapsed_seconds"] for r in rows),
    }


def evaluate(
    records, manifest, manifest_path, pipeline, output, limit=None, retrieval_mode="draft"
):
    if limit is not None and limit <= 0:
        raise ValueError("evaluation limit must be positive")
    if retrieval_mode == "aps":
        if pipeline.calibration is None:
            raise ValueError("APS evaluation requires calibration")
        ids = pipeline.calibration.artifact["identity"]["split"]["test_ids"]
    else:
        ids = pipeline.index.metadata["split"]["evaluation_ids"]
    selected = ids if limit is None else ids[:limit]
    by_id = {r.id: r for r in records}
    images = manifest_images(manifest, manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for position, record_id in enumerate(selected, 1):
            record = by_id[record_id]
            entry = images[record_id]
            # Only after answering do we attach gold for metrics/output.
            result = pipeline.answer(
                record.query(), entry["path"], entry["image_sha256"], retrieval_mode=retrieval_mode
            )
            result.update(
                {
                    "record_id": record.id,
                    "base_id": record.base_id,
                    "gold_answer": record.answer,
                    "gold_answer_text": record.answer_text,
                }
            )
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(result)
            print(
                "evaluation {}/{} (record {}, draft={}, final={})".format(
                    position,
                    len(selected),
                    record_id,
                    result["draft"]["letter"],
                    result["final"]["letter"],
                ),
                file=sys.stderr,
                flush=True,
            )
    summary = {
        **metrics(rows),
        "held_out_records": len(ids),
        "full_split": len(selected) == len(ids),
        "index_fingerprint": pipeline.index.metadata["fingerprint"],
        "dataset_fingerprint": pipeline.index.metadata["dataset_fingerprint"],
        "settings": pipeline.settings.metadata(),
        "runtime": pipeline.index.metadata["runtime"],
        "retrieval_mode": retrieval_mode,
        "calibration_fingerprint": (
            pipeline.calibration.artifact["fingerprint"] if retrieval_mode == "aps" else None
        ),
        "note": "Record-level accuracy includes option rotations; invalid predictions count as incorrect.",
    }
    write_json(output / "metrics.json", summary)
    return summary

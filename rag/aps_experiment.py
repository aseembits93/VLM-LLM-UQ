"""Fresh APS calibration and a paired candidate-versus-draft retrieval experiment."""

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .aps import (
    CANDIDATE_QUERY_VERSION,
    Calibration,
    calibration_identity,
    calibration_score,
    threshold,
)
from .common import fingerprint, read_json, write_json
from .data import manifest_images, verified_image
from .pipeline import metrics


def cached_payload(path, identity):
    if not path.exists():
        return None
    cached = read_json(path)
    if cached.get("identity") != identity or cached.get("checksum") != fingerprint(
        cached["payload"]
    ):
        raise ValueError("stale/corrupt experiment cache: {}".format(path))
    return cached["payload"]


def save_payload(path, identity, payload):
    write_json(path, {"identity": identity, "payload": payload, "checksum": fingerprint(payload)})


def calibrate(records, manifest, manifest_path, pipeline, output, alpha=0.1, fraction=0.5, seed=43):
    identity = calibration_identity(pipeline.index, pipeline.settings, alpha, fraction, seed)
    output = Path(output)
    if output.exists():
        calibration = Calibration.load(output, pipeline.index, pipeline.settings)
        if calibration.artifact["identity"] != identity:
            raise ValueError("calibration parameters changed; use a new --output path")
        return calibration
    cache = output.parent / (output.stem + "-scores")
    by_id = {record.id: record for record in records}
    images = manifest_images(manifest, manifest_path)
    rows = []
    ids = identity["split"]["calibration_ids"]
    for position, record_id in enumerate(ids, 1):
        record, entry = by_id[record_id], images[record_id]
        verified_image(entry)
        cache_path = cache / (str(record_id) + ".json")
        cache_identity = {"calibration_identity": fingerprint(identity), "record_id": record_id}
        prepared = cached_payload(cache_path, cache_identity)
        if prepared is None:
            prepared = pipeline.prepare_query(
                record.query(), entry["path"], entry["image_sha256"], include_scores=True
            )
            if not prepared["prepared"]:
                write_json(cache / (str(record_id) + "-failure.json"), prepared)
                raise ValueError("calibration record {} failed; see {}".format(record_id, cache))
            save_payload(cache_path, cache_identity, prepared)
        # Only after inference: labels are used for calibration, never model input.
        scores = prepared["draft"]["scores"]
        rows.append(
            {
                "record_id": record_id,
                "group_id": pipeline.index.metadata["split"]["record_groups"][str(record_id)],
                "image_sha256": entry["image_sha256"],
                "scores": scores,
                "gold_answer": record.answer,
                "calibration_score": calibration_score(scores, record.answer),
                "draft": prepared["draft"],
            }
        )
        print(
            "calibration {}/{} (record {})".format(position, len(ids), record_id),
            file=sys.stderr,
            flush=True,
        )
    qhat, rank = threshold([row["calibration_score"] for row in rows], alpha)
    artifact = {
        "identity": identity,
        "qhat": qhat,
        "qhat_null_means": "+infinity (all supplied choices)",
        "quantile_rank": rank,
        "records": rows,
        "note": "One representative per calibration image/rotation group; no test labels used. This dataset was previously evaluated, so comparisons are exploratory.",
    }
    artifact["fingerprint"] = fingerprint(artifact)
    calibration = Calibration(artifact, pipeline.index, pipeline.settings)
    write_json(output, artifact)
    return calibration


def paired_metrics(rows):
    if not rows:
        return {"total_records": 0}
    baseline = [row["draft_rag"] for row in rows]
    candidate = [row["aps_rag"] for row in rows]
    b = metrics(baseline)
    a = metrics(candidate)
    improved = harmed = 0
    sets, histogram = [], Counter()
    by_size = {}
    changed_retrieval = 0
    for row in rows:
        base, aps = row["draft_rag"], row["aps_rag"]
        base_correct = base["final"]["valid"] and base["final"]["letter"] == row["gold_answer"]
        aps_correct = aps["final"]["valid"] and aps["final"]["letter"] == row["gold_answer"]
        improved += int(aps_correct and not base_correct)
        harmed += int(base_correct and not aps_correct)
        labels = aps.get("prediction_set", {}).get("labels", [])
        sets.append(labels)
        histogram[str(len(labels))] += 1
        bucket = by_size.setdefault(
            str(len(labels)),
            {
                "count": 0,
                "covered": 0,
                "draft_correct": 0,
                "draft_rag_correct": 0,
                "aps_rag_correct": 0,
            },
        )
        bucket["count"] += 1
        bucket["covered"] += int(row["gold_answer"] in labels)
        bucket["draft_correct"] += int(
            base["draft"]["valid"] and base["draft"]["letter"] == row["gold_answer"]
        )
        bucket["draft_rag_correct"] += int(base_correct)
        bucket["aps_rag_correct"] += int(aps_correct)
        changed_retrieval += int(
            [entry["id"] for entry in base["retrieved"]]
            != [entry["id"] for entry in aps["retrieved"]]
        )
    discordant = improved + harmed
    # Exact two-sided paired test; descriptive in this exploratory comparison.
    p_value = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(improved, harmed) + 1))
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    return {
        "total_records": len(rows),
        "draft_rag": b,
        "aps_rag": a,
        "aps_minus_draft_rag_percentage_points": 100 * (a["final_accuracy"] - b["final_accuracy"]),
        "improved_over_draft_rag": improved,
        "harmed_vs_draft_rag": harmed,
        "exact_mcnemar_two_sided_p": p_value,
        "changed_retrievals": changed_retrieval,
        "prediction_set_coverage": sum(
            row["gold_answer"] in labels for row, labels in zip(rows, sets)
        )
        / len(rows),
        "mean_prediction_set_size": sum(map(len, sets)) / len(rows),
        "invalid_prediction_sets": sum(not labels for labels in sets),
        "prediction_set_size_histogram": dict(sorted(histogram.items())),
        "by_prediction_set_size": dict(sorted(by_size.items())),
    }


def comparison_summary(rows, calibration):
    split = calibration.artifact["identity"]["split"]
    representatives = set(split["test_representative_ids"])
    primary_rows = [row for row in rows if row["record_id"] in representatives]
    primary = paired_metrics(primary_rows)
    if primary_rows:
        differences = []
        for row in primary_rows:

            def correct(arm):
                result = row[arm]
                return int(
                    result["final"]["valid"] and result["final"]["letter"] == row["gold_answer"]
                )

            differences.append(correct("aps_rag") - correct("draft_rag"))
        rng = np.random.RandomState(43)
        samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        primary["paired_bootstrap_95pct_change_interval_pp"] = (
            100 * np.quantile(samples, [0.025, 0.975])
        ).tolist()
        primary["bootstrap_method"] = (
            "10,000 paired representative resamples; seed 43; percentile interval"
        )
    return {
        "primary_group_representatives": primary,
        "secondary_all_rotations": paired_metrics(rows),
        "full_test_split": sorted(row["record_id"] for row in rows) == split["test_ids"],
        "test_groups": len(split["test_groups"]),
        "test_records": len(split["test_ids"]),
        "calibration_groups": len(split["calibration_groups"]),
        "calibration_fingerprint": calibration.artifact["fingerprint"],
        "index_fingerprint": calibration.artifact["identity"]["index_fingerprint"],
        "alpha": calibration.artifact["identity"]["alpha"],
        "qhat": calibration.qhat,
        "candidate_query_version": CANDIDATE_QUERY_VERSION,
        "note": "Exploratory comparison on a previously evaluated dataset. Primary metrics use one lowest-ID representative per test image/rotation group. Rotations are correlated; the secondary McNemar number is descriptive only. APS coverage concerns the pre-retrieval set, not final-answer accuracy. Both arms always retrieve top three and use the identical final prompt template.",
    }


def compare(records, manifest, manifest_path, pipeline, output, limit=None):
    if pipeline.calibration is None:
        raise ValueError("comparison requires APS calibration")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    calibration = pipeline.calibration
    split = calibration.artifact["identity"]["split"]
    ids = split["test_ids"] if limit is None else split["test_ids"][:limit]
    identity = {
        "version": 1,
        "calibration_fingerprint": calibration.artifact["fingerprint"],
        "candidate_query_version": CANDIDATE_QUERY_VERSION,
        "selected_ids": ids,
    }
    output = Path(output)
    run_path = output / "run.json"
    if run_path.exists() and read_json(run_path) != identity:
        raise ValueError("comparison parameters changed; use a different output directory")
    write_json(run_path, identity)
    by_id = {record.id: record for record in records}
    images = manifest_images(manifest, manifest_path)
    rows = []
    started = time.monotonic()
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for position, record_id in enumerate(ids, 1):
            record, entry = by_id[record_id], images[record_id]
            cache_path = output / "cache" / (str(record_id) + ".json")
            cache_identity = {"run": fingerprint(identity), "record_id": record_id}
            row = cached_payload(cache_path, cache_identity)
            if row is None:
                prepared = pipeline.prepare_query(
                    record.query(), entry["path"], entry["image_sha256"], include_scores=True
                )
                arms = {}
                for name, mode in [("draft_rag", "draft"), ("aps_rag", "aps")]:
                    arms[name] = pipeline.answer(
                        record.query(),
                        entry["path"],
                        entry["image_sha256"],
                        retrieval_mode=mode,
                        prepared=prepared,
                    )
                # Gold becomes accessible only after both arms have finished.
                row = {
                    "record_id": record_id,
                    "group_id": pipeline.index.metadata["split"]["record_groups"][str(record_id)],
                    "gold_answer": record.answer,
                    "gold_answer_text": record.answer_text,
                    **arms,
                }
                for arm in arms.values():
                    arm["gold_answer"] = record.answer
                save_payload(cache_path, cache_identity, row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(
                "comparison {}/{} (record {}, set={}, baseline={}, APS={})".format(
                    position,
                    len(ids),
                    record_id,
                    row["aps_rag"].get("prediction_set", {}).get("labels", []),
                    row["draft_rag"]["final"]["letter"],
                    row["aps_rag"]["final"]["letter"],
                ),
                file=sys.stderr,
                flush=True,
            )
    summary = comparison_summary(rows, calibration)
    summary["this_invocation_elapsed_seconds"] = time.monotonic() - started
    write_json(output / "metrics.json", summary)
    return summary

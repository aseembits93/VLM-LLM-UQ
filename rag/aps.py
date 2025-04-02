"""Deterministic APS and group-disjoint calibration for the frozen RAG scorer."""

import math
import random

import numpy as np

from .common import Query, fingerprint, read_json
from .prompts import draft_prompt

APS_METHOD = {
    "version": 1,
    "score": "cumulative probability through candidate, inclusive",
    "ties": "ascending option letter",
    "quantile": "ceil((n+1)*(1-alpha))-th order statistic; infinity if rank > n",
    "inversion": "score <= qhat; include top option if empty",
    "probabilities": "softmax over exactly the supplied option letters",
    "randomized": False,
}
CANDIDATE_QUERY_VERSION = 1


def ranked_probabilities(scores):
    labels = sorted(scores)
    values = np.asarray([scores[label] for label in labels], dtype=np.float64)
    if len(labels) < 2 or values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("APS requires finite scores for at least two supplied options")
    weights = np.exp(values - values.max())
    probabilities = dict(zip(labels, (weights / weights.sum()).tolist()))
    order = sorted(labels, key=lambda label: (-probabilities[label], label))
    cumulative = np.minimum(1.0, np.cumsum([probabilities[label] for label in order]))
    cumulative[-1] = 1.0
    return order, probabilities, dict(zip(order, cumulative.tolist()))


def calibration_score(scores, gold):
    _, _, cumulative = ranked_probabilities(scores)
    if gold not in cumulative:
        raise ValueError("calibration gold is not a supplied option")
    return cumulative[gold]


def threshold(scores, alpha):
    values = sorted(float(score) for score in scores)
    if not 0 < alpha < 1 or not values:
        raise ValueError("APS needs nonempty calibration scores and 0 < alpha < 1")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("APS calibration scores must be finite values in [0, 1]")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    # JSON null encodes +infinity explicitly; clipping the rank would undercover.
    return (values[rank - 1] if rank <= len(values) else None), rank


def prediction_set(scores, qhat):
    if qhat is not None and (not math.isfinite(qhat) or not 0 <= qhat <= 1):
        raise ValueError("invalid APS threshold")
    order, probabilities, cumulative = ranked_probabilities(scores)
    labels = [label for label in order if qhat is None or cumulative[label] <= qhat]
    return labels or order[:1], probabilities


def calibration_split(index_metadata, fraction=0.5, seed=43):
    """Keep the reference index fixed; split only its held-out image groups."""
    if not 0 < fraction < 1:
        raise ValueError("calibration fraction must be between zero and one")
    original = index_metadata["split"]
    groups = sorted(original["evaluation_groups"])
    if len(groups) < 2:
        raise ValueError("at least two held-out image groups are required")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, int(len(groups) * fraction)))
    calibration_groups, test_groups = set(groups[:count]), set(groups[count:])
    mapping = original["record_groups"]
    all_ids = sorted(original["evaluation_ids"])
    representatives = {}
    for record_id in all_ids:
        representatives.setdefault(mapping[str(record_id)], record_id)
    return {
        "seed": seed,
        "calibration_fraction": fraction,
        "unit": "lowest record ID per merged image/rotation group",
        "calibration_groups": sorted(calibration_groups),
        "test_groups": sorted(test_groups),
        "calibration_ids": sorted(representatives[group] for group in calibration_groups),
        "calibration_all_ids": [i for i in all_ids if mapping[str(i)] in calibration_groups],
        "test_ids": [i for i in all_ids if mapping[str(i)] in test_groups],
        "test_representative_ids": sorted(representatives[group] for group in test_groups),
    }


def calibration_identity(index, settings, alpha=0.1, fraction=0.5, seed=43):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    return {
        "version": 1,
        "method": APS_METHOD,
        "alpha": alpha,
        "index_fingerprint": index.metadata["fingerprint"],
        "dataset_fingerprint": index.metadata["dataset_fingerprint"],
        "manifest_fingerprint": index.metadata["manifest_fingerprint"],
        "settings": settings.metadata(),
        "runtime": index.metadata["runtime"],
        "draft_prompt": draft_prompt(
            Query("QUESTION", {"A": "FIRST", "B": "SECOND"}, "HINT"), "DESCRIPTION"
        ),
        "split": calibration_split(index.metadata, fraction, seed),
    }


class Calibration:
    def __init__(self, artifact, index, settings):
        self.artifact = artifact
        if artifact.get("fingerprint") != fingerprint(
            {key: value for key, value in artifact.items() if key != "fingerprint"}
        ):
            raise ValueError("APS calibration checksum mismatch")
        identity = artifact["identity"]
        split = identity["split"]
        expected = calibration_identity(
            index, settings, identity["alpha"], split["calibration_fraction"], split["seed"]
        )
        if identity != expected:
            raise ValueError("stale/incompatible APS calibration; run calibrate-aps again")
        rows = artifact["records"]
        if [row["record_id"] for row in rows] != split["calibration_ids"]:
            raise ValueError("APS calibration must contain exactly its group representatives")
        values = [calibration_score(row["scores"], row["gold_answer"]) for row in rows]
        qhat, rank = threshold(values, identity["alpha"])
        if artifact["qhat"] != qhat or artifact["quantile_rank"] != rank:
            raise ValueError("APS calibration threshold does not match its scores")
        self.qhat = qhat

    @classmethod
    def load(cls, path, index, settings):
        return cls(read_json(path), index, settings)

    def predict(self, scores, choices):
        if set(scores) != set(choices):
            raise ValueError("APS scores must match exactly the supplied choices")
        labels, probabilities = prediction_set(scores, self.qhat)
        return {
            "labels": labels,
            "answer_texts": [choices[label] for label in labels],
            "probabilities": probabilities,
            "qhat": self.qhat,
            "alpha": self.artifact["identity"]["alpha"],
            "calibration_fingerprint": self.artifact["fingerprint"],
        }

"""Compare generic and conformal-set-guided contrastive retrieval queries.

This deliberately isolates one question: does adding the surviving conformal
options to the query retrieve the known supporting passage more often?  It does
not call a generator or claim post-RAG conformal coverage.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


def as_text(value: Any) -> str:
    """Return cleaned text while treating MMBench's NaN fields as empty."""
    return value.strip() if isinstance(value, str) else ""


def softmax(values: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("softmax expects a non-empty one-dimensional sequence")
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def option_probabilities(record: Mapping[str, Any]) -> np.ndarray:
    """Extract the six option probabilities used by the existing demo."""
    logits = np.asarray(record["logits"], dtype=float)
    if logits.ndim != 1 or logits.size < len(OPTION_LABELS):
        raise ValueError("each record needs at least six one-dimensional logits")
    return softmax(logits[: len(OPTION_LABELS)])


def aps_calibration_score(record: Mapping[str, Any]) -> float:
    """Return the APS cumulative mass at the true option."""
    answer = as_text(record.get("answer"))
    if answer not in OPTION_LABELS:
        raise ValueError("answer must be one of A-F")

    probabilities = option_probabilities(record)
    order = np.argsort(probabilities)[::-1]
    cumulative = np.cumsum(probabilities[order])
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return float(cumulative[ranks[OPTION_LABELS.index(answer)]])


def calibrate_qhat(records: Sequence[Mapping[str, Any]], alpha: float) -> float:
    """Calibrate the APS threshold with the finite-sample correction."""
    if not records:
        raise ValueError("calibration records cannot be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    scores = [aps_calibration_score(record) for record in records]
    quantile_level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, quantile_level, method="higher"))


def prediction_set(record: Mapping[str, Any], qhat: float) -> List[str]:
    """Reproduce the demo's APS prediction-set rule."""
    probabilities = option_probabilities(record)
    order = np.argsort(probabilities)[::-1]
    cumulative = np.cumsum(probabilities[order])
    labels = [
        OPTION_LABELS[int(order[index])]
        for index in range(len(order))
        if cumulative[index] <= qhat
    ]
    return labels or [OPTION_LABELS[int(order[0])]]


def generic_query(record: Mapping[str, Any]) -> str:
    """Baseline query: the question alone."""
    return as_text(record.get("question"))


def contrastive_query(
    record: Mapping[str, Any], candidate_labels: Sequence[str]
) -> str:
    """Add only surviving, non-empty answer options to the retrieval query."""
    candidates = [
        (label, as_text(record.get(label)))
        for label in candidate_labels
        if label in OPTION_LABELS and as_text(record.get(label))
    ]
    if len(candidates) < 2:
        raise ValueError("a contrastive query needs at least two non-empty candidates")

    rendered = "\n".join("{}: {}".format(label, text) for label, text in candidates)
    return "{}\nFind evidence that distinguishes these candidate answers:\n{}".format(
        generic_query(record), rendered
    )


class TfidfRetriever:
    """A minimal lexical retriever that keeps the query comparison controlled."""

    def __init__(self, documents: Sequence[str]):
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
        )
        self.document_matrix = self.vectorizer.fit_transform(self.documents)

    def ranks(
        self, queries: Sequence[str], relevant_indices: Sequence[int]
    ) -> List[int]:
        """Return one-indexed ranks of each query's known relevant document."""
        if len(queries) != len(relevant_indices):
            raise ValueError("queries and relevant_indices must have equal length")
        similarities = self.vectorizer.transform(queries) @ self.document_matrix.T
        ranks: List[int] = []
        for row_index, relevant_index in enumerate(relevant_indices):
            scores = similarities.getrow(row_index).toarray()[0]
            ordered = np.argsort(-scores, kind="stable")
            ranks.append(int(np.flatnonzero(ordered == relevant_index)[0]) + 1)
        return ranks


def rank_metrics(ranks: Sequence[int], top_k: Sequence[int]) -> Dict[str, float]:
    rank_array = np.asarray(ranks, dtype=float)
    if rank_array.size == 0:
        raise ValueError("at least one rank is required")
    metrics = {
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(np.mean(rank_array)),
        "median_rank": float(np.median(rank_array)),
    }
    for k in top_k:
        metrics["hit_at_{}".format(k)] = float(np.mean(rank_array <= k))
    return metrics


def plot_results(result: Mapping[str, Any], output_path: Path) -> None:
    """Plot aggregate retrieval quality and paired per-question outcomes."""
    top_k = result["configuration"]["top_k"]
    generic_hits = [100 * result["generic_query"][f"hit_at_{k}"] for k in top_k]
    contrastive_hits = [100 * result["contrastive_query"][f"hit_at_{k}"] for k in top_k]
    positions = np.arange(len(top_k))
    width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    generic_bars = axes[0].bar(
        positions - width / 2,
        generic_hits,
        width,
        label="Question only",
        color="#64748b",
    )
    contrastive_bars = axes[0].bar(
        positions + width / 2,
        contrastive_hits,
        width,
        label="Contrastive",
        color="#2563eb",
    )
    axes[0].bar_label(generic_bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].bar_label(contrastive_bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].set_xticks(positions, [f"Hit@{k}" for k in top_k])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Supporting passage retrieved (%)")
    axes[0].set_title("Evidence retrieval improves")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    comparison = result["paired_rank_comparison"]
    outcome_labels = ["Better", "Tied", "Worse"]
    outcome_counts = [
        comparison["contrastive_better"],
        comparison["tied"],
        comparison["contrastive_worse"],
    ]
    outcome_bars = axes[1].bar(
        outcome_labels,
        outcome_counts,
        color=["#16a34a", "#94a3b8", "#dc2626"],
        width=0.62,
    )
    axes[1].bar_label(outcome_bars, padding=3, fontsize=10)
    axes[1].set_ylim(0, max(outcome_counts) * 1.18)
    axes[1].set_ylabel("Evaluation questions")
    axes[1].set_title("Change in supporting-passage rank")
    axes[1].grid(axis="y", alpha=0.2)

    eligible = result["counts"]["eligible_non_singletons"]
    figure.suptitle(
        f"Conformal-set contrastive retrieval MVP (n={eligible})", fontsize=14
    )
    figure.text(
        0.5,
        0.01,
        "Same corpus, TF-IDF retriever, and retrieval depth; only the query changes.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def record_key(record: Mapping[str, Any], fallback: int) -> int:
    for field in ("id", "index"):
        try:
            return int(record[field])
        except (KeyError, TypeError, ValueError):
            continue
    return fallback


def split_by_id_parity(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Use even IDs for calibration and odd IDs for evaluation."""
    calibration: List[Mapping[str, Any]] = []
    evaluation: List[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        target = calibration if record_key(record, index) % 2 == 0 else evaluation
        target.append(record)
    if not calibration or not evaluation:
        raise ValueError(
            "the parity split must produce non-empty calibration and evaluation sets"
        )
    return calibration, evaluation


def unique_hints(records: Iterable[Mapping[str, Any]]) -> List[str]:
    return sorted(
        {hint for hint in (as_text(record.get("hint")) for record in records) if hint}
    )


def run_experiment(
    records: Sequence[Mapping[str, Any]], alpha: float, top_k: Sequence[int]
) -> Dict[str, Any]:
    """Run the paired query experiment over the MMBench result records."""
    calibration, evaluation = split_by_id_parity(records)
    qhat = calibrate_qhat(calibration, alpha)
    documents = unique_hints(records)
    document_indices = {document: index for index, document in enumerate(documents)}

    eligible: List[Tuple[Mapping[str, Any], List[str], str]] = []
    skipped = {"singleton": 0, "missing_hint": 0, "fewer_than_two_named_candidates": 0}
    evaluation_sets: List[List[str]] = []
    for record in evaluation:
        candidates = prediction_set(record, qhat)
        evaluation_sets.append(candidates)
        hint = as_text(record.get("hint"))
        named_candidates = [label for label in candidates if as_text(record.get(label))]
        if len(candidates) == 1:
            skipped["singleton"] += 1
        elif not hint:
            skipped["missing_hint"] += 1
        elif len(named_candidates) < 2:
            skipped["fewer_than_two_named_candidates"] += 1
        else:
            eligible.append((record, candidates, hint))

    if not eligible:
        raise ValueError("no evaluation records are eligible for contrastive retrieval")

    baseline_queries = [generic_query(record) for record, _, _ in eligible]
    contrastive_queries = [
        contrastive_query(record, candidates) for record, candidates, _ in eligible
    ]
    relevant_indices = [document_indices[hint] for _, _, hint in eligible]
    retriever = TfidfRetriever(documents)
    baseline_ranks = retriever.ranks(baseline_queries, relevant_indices)
    contrastive_ranks = retriever.ranks(contrastive_queries, relevant_indices)
    baseline_metrics = rank_metrics(baseline_ranks, top_k)
    contrastive_metrics = rank_metrics(contrastive_ranks, top_k)

    initial_coverage = np.mean(
        [record["answer"] in candidates for record, candidates, _ in eligible]
    )
    evaluation_coverage = np.mean(
        [
            record["answer"] in candidates
            for record, candidates in zip(evaluation, evaluation_sets)
        ]
    )
    baseline_array = np.asarray(baseline_ranks)
    contrastive_array = np.asarray(contrastive_ranks)
    deltas = {
        metric: float(contrastive_metrics[metric] - baseline_metrics[metric])
        for metric in baseline_metrics
    }

    return {
        "configuration": {
            "alpha": alpha,
            "qhat": qhat,
            "split": "even record id = calibration; odd record id = evaluation",
            "retriever": "word TF-IDF cosine similarity",
            "top_k": list(top_k),
        },
        "counts": {
            "records": len(records),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
            "corpus_documents": len(documents),
            "eligible_non_singletons": len(eligible),
            "skipped": skipped,
        },
        "initial_sets": {
            "evaluation_average_size": float(
                np.mean([len(item) for item in evaluation_sets])
            ),
            "evaluation_true_answer_coverage": float(evaluation_coverage),
            "eligible_true_answer_coverage": float(initial_coverage),
        },
        "generic_query": baseline_metrics,
        "contrastive_query": contrastive_metrics,
        "contrastive_minus_generic": deltas,
        "paired_rank_comparison": {
            "contrastive_better": int(np.sum(contrastive_array < baseline_array)),
            "tied": int(np.sum(contrastive_array == baseline_array)),
            "contrastive_worse": int(np.sum(contrastive_array > baseline_array)),
        },
        "scope_note": (
            "Retrieval-only MVP: supporting-hint rank is measured before adding a "
            "generator or recalibrating evidence-conditioned prediction sets."
        ),
    }


def load_records(path: Path) -> List[Mapping[str, Any]]:
    # Pickle is already the repository's source format. Only load a trusted local file.
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError("expected the input pickle to contain a list of records")
    return records


def parse_top_k(value: str) -> List[int]:
    try:
        top_k = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "top-k must be comma-separated integers"
        ) from error
    if not top_k or top_k[0] < 1:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return top_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("mmbench.pkl"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--top-k", type=parse_top_k, default=parse_top_k("1,3,5"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    result = run_experiment(load_records(args.data), args.alpha, args.top_k)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.plot:
        plot_results(result, args.plot)
    print(rendered)


if __name__ == "__main__":
    main()

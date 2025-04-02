"""Compare normal retrieval with prediction-set-guided retrieval.

The whole experiment is contained in this file so it is easy to follow:

1. Split MMBench into separate calibration and evaluation examples.
2. Use calibration examples to choose the conformal cutoff ``qhat``.
3. Turn each evaluation example's logits into a prediction set.
4. For non-singleton sets, build two search queries:
   a. the original question by itself;
   b. the question plus the answer options that survived in the set.
5. Send both queries through the exact same TF-IDF retriever.
6. Compare where the known supporting passage appears in the results.
7. Save the metrics as JSON and optionally draw a Matplotlib figure.

This is intentionally only a retrieval experiment. It does not ask a model to
answer again after retrieval, so it does not calculate a new prediction set S1.

Type-hint cheat sheet for newer Python readers:

* ``Mapping[str, Any]`` means a dictionary-like MMBench record.
* ``Sequence[float]`` means an ordered collection of numbers.
* ``List[str]`` means a list of strings.
* ``Dict[str, float]`` means a dictionary whose values are numbers.
* ``->`` shows what a function returns; it does not change how the code runs.
"""

# argparse reads command-line flags such as --data and --plot.
import argparse

# json turns the result dictionary into a readable JSON file.
import json

# pickle loads the existing mmbench.pkl file used by this repository.
import pickle

# Path makes file paths easier and safer to work with.
from pathlib import Path

# These typing names document what kind of data each function expects.
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# NumPy supplies array operations, softmax math, sorting, and averages.
import numpy as np

# Matplotlib draws the two result charts.
from matplotlib import pyplot as plt

# TfidfVectorizer is the small lexical search engine used in both conditions.
from sklearn.feature_extraction.text import TfidfVectorizer


# The current demo asks the VLM for one of these six option letters.
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


def as_text(value: Any) -> str:
    """Return cleaned text while treating MMBench's NaN fields as empty."""
    # Some missing MMBench hints are floating-point NaNs instead of strings.
    # Only strings are safe to strip; every other value becomes empty text.
    return value.strip() if isinstance(value, str) else ""


def softmax(values: Sequence[float]) -> np.ndarray:
    """Convert arbitrary model scores into probabilities that sum to one."""
    # Make a floating-point NumPy array so the vector math below is predictable.
    scores = np.asarray(values, dtype=float)
    # A single row of option scores is required for one multiple-choice question.
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("softmax expects a non-empty one-dimensional sequence")
    # Subtracting the largest score prevents exp() from overflowing.
    shifted = scores - np.max(scores)
    # Exponentiation makes every value positive.
    exponentials = np.exp(shifted)
    # Dividing by the total changes the positive values into probabilities.
    return exponentials / exponentials.sum()


def option_probabilities(record: Mapping[str, Any]) -> np.ndarray:
    """Extract the six option probabilities used by the existing demo."""
    # Each record already contains the VLM's raw output scores under "logits".
    logits = np.asarray(record["logits"], dtype=float)
    # A-F require at least six scores in one flat row.
    if logits.ndim != 1 or logits.size < len(OPTION_LABELS):
        raise ValueError("each record needs at least six one-dimensional logits")
    # Keep the six A-F scores and convert them to probabilities.
    return softmax(logits[: len(OPTION_LABELS)])


def aps_calibration_score(record: Mapping[str, Any]) -> float:
    """Return the APS cumulative mass at the true option."""
    # Read the gold answer letter, for example "B".
    answer = as_text(record.get("answer"))
    # A calibration score is impossible if the answer is not one of A-F.
    if answer not in OPTION_LABELS:
        raise ValueError("answer must be one of A-F")

    # Convert this example's six logits into six probabilities.
    probabilities = option_probabilities(record)
    # Sort option positions from the highest probability to the lowest.
    # [::-1] reverses NumPy's default ascending order.
    order = np.argsort(probabilities)[::-1]
    # Add probability mass as we move down the sorted answer list.
    cumulative = np.cumsum(probabilities[order])
    # Create an empty array that will map each option back to its sorted rank.
    ranks = np.empty_like(order)
    # If order is [1, 0, 2], ranks becomes [1, 0, 2].
    ranks[order] = np.arange(len(order))
    # Return the cumulative probability at the position of the true answer.
    return float(cumulative[ranks[OPTION_LABELS.index(answer)]])


def calibrate_qhat(records: Sequence[Mapping[str, Any]], alpha: float) -> float:
    """Calibrate the APS threshold with the finite-sample correction."""
    # We need examples with known answers to select a conformal cutoff.
    if not records:
        raise ValueError("calibration records cannot be empty")
    # Alpha is the requested error level, so it must be a probability.
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    # Calculate one APS score for every calibration example.
    scores = [aps_calibration_score(record) for record in records]
    # This is the standard split-conformal finite-sample quantile level.
    # min() handles tiny datasets where the corrected level could exceed one.
    quantile_level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    # "higher" picks an observed score instead of interpolating two scores.
    return float(np.quantile(scores, quantile_level, method="higher"))


def prediction_set(record: Mapping[str, Any], qhat: float) -> List[str]:
    """Reproduce the demo's APS prediction-set rule."""
    # Start from the VLM's probability for every option.
    probabilities = option_probabilities(record)
    # Visit the most likely option first.
    order = np.argsort(probabilities)[::-1]
    # Track how much total probability is covered after each option.
    cumulative = np.cumsum(probabilities[order])
    # Keep every sorted option whose cumulative mass is below qhat.
    # This deliberately matches the prediction-set rule in app.py.
    labels = [
        OPTION_LABELS[int(order[index])]
        for index in range(len(order))
        if cumulative[index] <= qhat
    ]
    # If nothing passed the cutoff, keep the top answer so the set is never empty.
    return labels or [OPTION_LABELS[int(order[0])]]


def generic_query(record: Mapping[str, Any]) -> str:
    """Baseline query: the question alone."""
    # This is the control condition against which contrastive retrieval is judged.
    return as_text(record.get("question"))


def contrastive_query(
    record: Mapping[str, Any], candidate_labels: Sequence[str]
) -> str:
    """Add only surviving, non-empty answer options to the retrieval query."""
    # Pair each surviving letter with its answer text, dropping missing options.
    candidates = [
        (label, as_text(record.get(label)))
        for label in candidate_labels
        if label in OPTION_LABELS and as_text(record.get(label))
    ]
    # Comparing fewer than two real answers would not be contrastive retrieval.
    if len(candidates) < 2:
        raise ValueError("a contrastive query needs at least two non-empty candidates")

    # Put each candidate on its own line, such as "A: cat" and "C: dog".
    rendered = "\n".join("{}: {}".format(label, text) for label, text in candidates)
    # Keep the original question and append an explicit comparison instruction.
    return "{}\nFind evidence that distinguishes these candidate answers:\n{}".format(
        generic_query(record), rendered
    )


class TfidfRetriever:
    """A minimal lexical retriever that keeps the query comparison controlled."""

    def __init__(self, documents: Sequence[str]):
        # A search engine cannot be fitted without documents.
        if not documents:
            raise ValueError("documents cannot be empty")
        # Store a concrete list because callers may provide another sequence type.
        self.documents = list(documents)
        # TF-IDF gives more weight to informative words than common words.
        self.vectorizer = TfidfVectorizer(
            # "Cat" and "cat" should be the same search term.
            lowercase=True,
            # Ignore common English words such as "the" and "is".
            stop_words="english",
            # Log-scale repeated word counts so repetition does not dominate.
            sublinear_tf=True,
        )
        # Learn the vocabulary and represent every passage as a sparse vector.
        self.document_matrix = self.vectorizer.fit_transform(self.documents)

    def ranks(
        self, queries: Sequence[str], relevant_indices: Sequence[int]
    ) -> List[int]:
        """Return one-indexed ranks of each query's known relevant document."""
        # Each query must be paired with one known supporting-passage index.
        if len(queries) != len(relevant_indices):
            raise ValueError("queries and relevant_indices must have equal length")
        # Turn queries into TF-IDF vectors, then compute all query-document dot
        # products. TF-IDF normalizes vectors, so these are cosine similarities.
        similarities = self.vectorizer.transform(queries) @ self.document_matrix.T
        # This list will hold positions such as 1st, 2nd, or 10th.
        ranks: List[int] = []
        # Evaluate one query and its known supporting passage at a time.
        for row_index, relevant_index in enumerate(relevant_indices):
            # Convert this sparse similarity row into a simple score array.
            scores = similarities.getrow(row_index).toarray()[0]
            # Negating scores lets ascending argsort produce highest-first order.
            # Stable sorting makes tied scores deterministic across runs.
            ordered = np.argsort(-scores, kind="stable")
            # Find the zero-based position of the supporting passage, then add
            # one so a best result is reported as rank 1 rather than rank 0.
            ranks.append(int(np.flatnonzero(ordered == relevant_index)[0]) + 1)
        # Return one supporting-passage rank for every query.
        return ranks


def rank_metrics(ranks: Sequence[int], top_k: Sequence[int]) -> Dict[str, float]:
    """Summarize supporting-passage ranks with familiar retrieval metrics."""
    # NumPy makes the averages and comparisons below concise.
    rank_array = np.asarray(ranks, dtype=float)
    # Metrics are undefined when there are no evaluated questions.
    if rank_array.size == 0:
        raise ValueError("at least one rank is required")
    # MRR rewards putting the supporting passage near the top. Mean and median
    # rank show the typical absolute position; lower rank is better.
    metrics = {
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(np.mean(rank_array)),
        "median_rank": float(np.median(rank_array)),
    }
    # Hit@k is the fraction of questions whose passage appeared in the first k.
    for k in top_k:
        metrics["hit_at_{}".format(k)] = float(np.mean(rank_array <= k))
    # The same metric function is used for both query types.
    return metrics


def plot_results(result: Mapping[str, Any], output_path: Path) -> None:
    """Plot aggregate retrieval quality and paired per-question outcomes."""
    # Read the requested retrieval depths, normally 1, 3, and 5.
    top_k = result["configuration"]["top_k"]
    # Convert generic-query hit rates from fractions to percentages.
    generic_hits = [100 * result["generic_query"][f"hit_at_{k}"] for k in top_k]
    # Convert contrastive-query hit rates from fractions to percentages.
    contrastive_hits = [100 * result["contrastive_query"][f"hit_at_{k}"] for k in top_k]
    # Create one x-axis position for each retrieval depth.
    positions = np.arange(len(top_k))
    # Each pair of bars must fit around one x-axis position.
    width = 0.36

    # Put the aggregate comparison and paired comparison side by side.
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    # Draw the gray baseline bars slightly left of each x position.
    generic_bars = axes[0].bar(
        positions - width / 2,
        generic_hits,
        width,
        label="Question only",
        color="#64748b",
    )
    # Draw the blue contrastive bars slightly right of each x position.
    contrastive_bars = axes[0].bar(
        positions + width / 2,
        contrastive_hits,
        width,
        label="Contrastive",
        color="#2563eb",
    )
    # Print the exact percentage above every gray bar.
    axes[0].bar_label(generic_bars, fmt="%.1f%%", padding=3, fontsize=9)
    # Print the exact percentage above every blue bar.
    axes[0].bar_label(contrastive_bars, fmt="%.1f%%", padding=3, fontsize=9)
    # Name the three x positions Hit@1, Hit@3, and Hit@5.
    axes[0].set_xticks(positions, [f"Hit@{k}" for k in top_k])
    # Percentages always live between zero and one hundred.
    axes[0].set_ylim(0, 100)
    # Explain what the left panel's vertical axis represents.
    axes[0].set_ylabel("Supporting passage retrieved (%)")
    # Give the left panel a short conclusion-oriented title.
    axes[0].set_title("Evidence retrieval improves")
    # Identify the gray and blue conditions.
    axes[0].legend(loc="lower right", frameon=False)
    # Faint horizontal lines make values easier to compare.
    axes[0].grid(axis="y", alpha=0.2)

    # Read the per-question better/tied/worse counts.
    comparison = result["paired_rank_comparison"]
    # These labels become the right panel's three x-axis categories.
    outcome_labels = ["Better", "Tied", "Worse"]
    # Keep the counts in the same order as the labels.
    outcome_counts = [
        comparison["contrastive_better"],
        comparison["tied"],
        comparison["contrastive_worse"],
    ]
    # Use green for improvement, gray for no change, and red for regression.
    outcome_bars = axes[1].bar(
        outcome_labels,
        outcome_counts,
        color=["#16a34a", "#94a3b8", "#dc2626"],
        width=0.62,
    )
    # Print the number of questions above each right-panel bar.
    axes[1].bar_label(outcome_bars, padding=3, fontsize=10)
    # Add headroom so the largest number is not clipped.
    axes[1].set_ylim(0, max(outcome_counts) * 1.18)
    # Explain what is being counted.
    axes[1].set_ylabel("Evaluation questions")
    # Explain that this panel compares the two ranks question by question.
    axes[1].set_title("Change in supporting-passage rank")
    # Add the same subtle horizontal guide lines as the first panel.
    axes[1].grid(axis="y", alpha=0.2)

    # Include the final evaluation sample size in the overall title.
    eligible = result["counts"]["eligible_non_singletons"]
    figure.suptitle(
        f"Conformal-set contrastive retrieval MVP (n={eligible})", fontsize=14
    )
    # State the controlled-experiment assumption at the bottom of the figure.
    figure.text(
        0.5,
        0.01,
        "Same corpus, TF-IDF retriever, and retrieval depth; only the query changes.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    # Reserve room for the title and footnote while fixing subplot spacing.
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    # Create the destination directory when it does not already exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save a crisp PNG with whitespace tightly cropped around the figure.
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    # Release Matplotlib's in-memory figure after saving it.
    plt.close(figure)


def record_key(record: Mapping[str, Any], fallback: int) -> int:
    """Find a stable integer ID that can assign a record to one data split."""
    # Prefer the dataset's ID, but accept its index field as a backup.
    for field in ("id", "index"):
        try:
            # Convert string-like IDs to integers when necessary.
            return int(record[field])
        except (KeyError, TypeError, ValueError):
            # If this field is absent or invalid, try the next field.
            continue
    # The record's list position is the final deterministic fallback.
    return fallback


def split_by_id_parity(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Use even IDs for calibration and odd IDs for evaluation."""
    # Labels from the calibration list are used only to choose qhat.
    calibration: List[Mapping[str, Any]] = []
    # Metrics are computed only from the separate evaluation list.
    evaluation: List[Mapping[str, Any]] = []
    # enumerate() also provides a fallback key when an ID is missing.
    for index, record in enumerate(records):
        # Modulo two is zero for even IDs and one for odd IDs.
        target = calibration if record_key(record, index) % 2 == 0 else evaluation
        # Add this record to exactly one of the two splits.
        target.append(record)
    # Both halves must contain data for the experiment to be meaningful.
    if not calibration or not evaluation:
        raise ValueError(
            "the parity split must produce non-empty calibration and evaluation sets"
        )
    # Return calibration first because it is used first in run_experiment().
    return calibration, evaluation


def unique_hints(records: Iterable[Mapping[str, Any]]) -> List[str]:
    """Build the small retrieval corpus from unique, non-empty MMBench hints."""
    # A set removes duplicate supporting passages. Sorting gives every passage a
    # stable numeric index so tied retrieval results are reproducible.
    return sorted(
        {hint for hint in (as_text(record.get("hint")) for record in records) if hint}
    )


def run_experiment(
    records: Sequence[Mapping[str, Any]], alpha: float, top_k: Sequence[int]
) -> Dict[str, Any]:
    """Run the paired query experiment over the MMBench result records."""
    # Step 1: keep calibration labels separate from evaluation metrics.
    calibration, evaluation = split_by_id_parity(records)
    # Step 2: learn one conformal cutoff from only the calibration half.
    qhat = calibrate_qhat(calibration, alpha)
    # Step 3: use all available hints as the searchable document collection.
    documents = unique_hints(records)
    # Map exact hint text to its position in the sorted document list.
    document_indices = {document: index for index, document in enumerate(documents)}

    # Each eligible item stores its record, prediction set, and supporting hint.
    eligible: List[Tuple[Mapping[str, Any], List[str], str]] = []
    # Count why evaluation examples cannot participate in this retrieval test.
    skipped = {"singleton": 0, "missing_hint": 0, "fewer_than_two_named_candidates": 0}
    # Keep every evaluation prediction set for overall size and coverage metrics.
    evaluation_sets: List[List[str]] = []
    # Inspect each held-out evaluation example exactly once.
    for record in evaluation:
        # S0 is the initial conformal set made from the stored VLM logits.
        candidates = prediction_set(record, qhat)
        # Save S0 even if this question cannot enter the retrieval comparison.
        evaluation_sets.append(candidates)
        # The hint is treated as the known supporting passage for this question.
        hint = as_text(record.get("hint"))
        # E and F are sometimes present in S0 even when their answer text is empty.
        named_candidates = [label for label in candidates if as_text(record.get(label))]
        # A singleton has no two surviving answers to contrast.
        if len(candidates) == 1:
            skipped["singleton"] += 1
        # Without a hint, there is no known relevant passage whose rank we can score.
        elif not hint:
            skipped["missing_hint"] += 1
        # At least two answer texts are needed to write "A versus C."
        elif len(named_candidates) < 2:
            skipped["fewer_than_two_named_candidates"] += 1
        # This question satisfies all requirements for the paired comparison.
        else:
            eligible.append((record, candidates, hint))

    # Stop with a clear error instead of returning meaningless empty averages.
    if not eligible:
        raise ValueError("no evaluation records are eligible for contrastive retrieval")

    # Step 4a: baseline queries contain only the original question.
    baseline_queries = [generic_query(record) for record, _, _ in eligible]
    # Step 4b: experimental queries also name the answers that survived in S0.
    contrastive_queries = [
        contrastive_query(record, candidates) for record, candidates, _ in eligible
    ]
    # Translate every known hint into the retriever's numeric document index.
    relevant_indices = [document_indices[hint] for _, _, hint in eligible]
    # Step 5: fit one retriever, ensuring both query types use the same search index.
    retriever = TfidfRetriever(documents)
    # Retrieve with the question-only queries and record supporting-passage ranks.
    baseline_ranks = retriever.ranks(baseline_queries, relevant_indices)
    # Retrieve again with contrastive queries, changing nothing else.
    contrastive_ranks = retriever.ranks(contrastive_queries, relevant_indices)
    # Step 6: summarize the baseline ranks.
    baseline_metrics = rank_metrics(baseline_ranks, top_k)
    # Summarize the contrastive ranks with the exact same metrics.
    contrastive_metrics = rank_metrics(contrastive_ranks, top_k)

    # Check how often S0 contains the correct answer among eligible questions.
    initial_coverage = np.mean(
        [record["answer"] in candidates for record, candidates, _ in eligible]
    )
    # Also measure S0 coverage across the complete held-out evaluation split.
    evaluation_coverage = np.mean(
        [
            record["answer"] in candidates
            for record, candidates in zip(evaluation, evaluation_sets)
        ]
    )
    # Arrays make paired less-than/equal/greater-than comparisons straightforward.
    baseline_array = np.asarray(baseline_ranks)
    contrastive_array = np.asarray(contrastive_ranks)
    # A positive hit-rate or MRR delta favors contrastive retrieval. A negative
    # mean-rank delta also favors it because a smaller rank is better.
    deltas = {
        metric: float(contrastive_metrics[metric] - baseline_metrics[metric])
        for metric in baseline_metrics
    }

    # Step 7: assemble all settings, counts, and metrics into serializable values.
    return {
        # These fields make the run reproducible and interpretable later.
        "configuration": {
            "alpha": alpha,
            "qhat": qhat,
            "split": "even record id = calibration; odd record id = evaluation",
            "retriever": "word TF-IDF cosine similarity",
            "top_k": list(top_k),
        },
        # These counts show how the original 4,377 records become 142 trials.
        "counts": {
            "records": len(records),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
            "corpus_documents": len(documents),
            "eligible_non_singletons": len(eligible),
            "skipped": skipped,
        },
        # These describe the VLM's uncertainty before any retrieval happens.
        "initial_sets": {
            "evaluation_average_size": float(
                np.mean([len(item) for item in evaluation_sets])
            ),
            "evaluation_true_answer_coverage": float(evaluation_coverage),
            "eligible_true_answer_coverage": float(initial_coverage),
        },
        # Store absolute retrieval results for each query condition.
        "generic_query": baseline_metrics,
        "contrastive_query": contrastive_metrics,
        # Store direct metric differences to simplify comparison and plotting.
        "contrastive_minus_generic": deltas,
        # Compare the two supporting-passage ranks question by question.
        "paired_rank_comparison": {
            "contrastive_better": int(np.sum(contrastive_array < baseline_array)),
            "tied": int(np.sum(contrastive_array == baseline_array)),
            "contrastive_worse": int(np.sum(contrastive_array > baseline_array)),
        },
        # Prevent this retrieval result from being read as an end-to-end guarantee.
        "scope_note": (
            "Retrieval-only MVP: supporting-hint rank is measured before adding a "
            "generator or recalibrating evidence-conditioned prediction sets."
        ),
    }


def load_records(path: Path) -> List[Mapping[str, Any]]:
    """Load the repository's existing list of MMBench result dictionaries."""
    # Pickle can execute embedded Python, so only load this trusted local data file.
    with path.open("rb") as handle:
        # Deserialize the same mmbench.pkl format already consumed by app.py.
        records = pickle.load(handle)
    # The experiment expects one Python dictionary per item in a list.
    if not isinstance(records, list):
        raise ValueError("expected the input pickle to contain a list of records")
    # Return the complete dataset; run_experiment() performs the split.
    return records


def parse_top_k(value: str) -> List[int]:
    """Turn a command-line value such as '1,3,5' into sorted integers."""
    try:
        # A set removes duplicate depths, and sorted() gives a stable order.
        top_k = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        # Replace Python's conversion error with a command-line-friendly message.
        raise argparse.ArgumentTypeError(
            "top-k must be comma-separated integers"
        ) from error
    # Retrieval depth zero or below has no useful meaning.
    if not top_k or top_k[0] < 1:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    # argparse stores this list directly in args.top_k.
    return top_k


def main() -> None:
    """Read CLI arguments, run the experiment, and save requested artifacts."""
    # Use this module's opening explanation as the --help description.
    parser = argparse.ArgumentParser(description=__doc__)
    # The tracked MMBench result pickle is the default input.
    parser.add_argument("--data", type=Path, default=Path("mmbench.pkl"))
    # Alpha=0.1 is the conformal error target used by the existing demo.
    parser.add_argument("--alpha", type=float, default=0.1)
    # Report whether evidence appears in the first 1, 3, and 5 passages by default.
    parser.add_argument("--top-k", type=parse_top_k, default=parse_top_k("1,3,5"))
    # --output is optional; when present, metrics are written as JSON here.
    parser.add_argument("--output", type=Path)
    # --plot is optional; when present, the Matplotlib PNG is written here.
    parser.add_argument("--plot", type=Path)
    # Parse the actual values provided by the user in the terminal.
    args = parser.parse_args()

    # Load the input once and pass all experiment settings to the core function.
    result = run_experiment(load_records(args.data), args.alpha, args.top_k)
    # Pretty-print and sort keys so repeated runs create stable, readable JSON.
    rendered = json.dumps(result, indent=2, sort_keys=True)
    # Only write a JSON file when the caller supplied --output.
    if args.output:
        # Make parent folders such as results/ when needed.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # UTF-8 and a final newline make the output a normal text file.
        args.output.write_text(rendered + "\n", encoding="utf-8")
    # Only create a chart when the caller supplied --plot.
    if args.plot:
        plot_results(result, args.plot)
    # Always print the metrics so a quick run is useful without output files.
    print(rendered)


# This guard runs main() for `python -m experiments.contrastive_retrieval`, but
# avoids running the full experiment when another Python file imports functions.
if __name__ == "__main__":
    main()

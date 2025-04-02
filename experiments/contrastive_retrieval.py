"""Compare normal retrieval with prediction-set-guided retrieval.

The whole experiment is contained in this file so it is easy to follow:

1. Split MMBench into separate calibration and evaluation examples.
2. Use calibration examples to choose the conformal cutoff ``qhat``.
3. Turn each evaluation example's logits into a prediction set.
4. For non-singleton sets, build two search queries:
   a. the original question by itself;
   b. the question plus the answer options that survived in the set.
5. Send both queries through the exact same TF-IDF retriever.
6. Check whether the known supporting passage appears in the first five results.
7. Save the metrics as JSON and optionally draw a Matplotlib figure.

This is intentionally only a retrieval experiment. It does not ask a model to
answer again after retrieval, so it does not calculate a new prediction set S1.

Type-hint cheat sheet for newer Python readers:

* ``Mapping[str, Any]`` means a dictionary-like MMBench record.
* ``Sequence[float]`` means an ordered collection of numbers.
* ``List[str]`` means a list of strings.
* ``Dict[str, Any]`` means a dictionary whose values can have different types.
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

# Matplotlib draws the result chart.
from matplotlib import pyplot as plt

# TfidfVectorizer converts passages and queries into weighted word vectors.
from sklearn.feature_extraction.text import TfidfVectorizer

# NearestNeighbors performs the cosine-similarity search.
from sklearn.neighbors import NearestNeighbors


# The current demo asks the VLM for one of these six option letters.
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
# This MVP evaluates only whether evidence appears in the first five results.
RETRIEVAL_DEPTH = 5


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


class SklearnTfidfRetriever:
    """A scikit-learn lexical retriever used identically for both query types."""

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
        # Ask scikit-learn to compare vectors using cosine distance. "Brute"
        # checks every document, which is simple and fast for this 97-passage corpus.
        self.neighbor_index = NearestNeighbors(metric="cosine", algorithm="brute")
        # Fit the reusable search index on the document vectors.
        self.neighbor_index.fit(self.document_matrix)

    def hit_at_five(
        self, queries: Sequence[str], relevant_indices: Sequence[int]
    ) -> float:
        """Return the fraction of queries that retrieve their evidence in five."""
        # Each query must be paired with one known supporting-passage index.
        if len(queries) != len(relevant_indices):
            raise ValueError("queries and relevant_indices must have equal length")
        # Represent each query with the vocabulary learned from the passages.
        query_matrix = self.vectorizer.transform(queries)
        # Let scikit-learn retrieve the five nearest passages for every query.
        _, ordered_documents = self.neighbor_index.kneighbors(
            query_matrix,
            n_neighbors=min(RETRIEVAL_DEPTH, len(self.documents)),
        )
        # This list records one True/False retrieval outcome per query.
        hits: List[bool] = []
        # Evaluate one query and its known supporting passage at a time.
        for row_index, relevant_index in enumerate(relevant_indices):
            # A hit means the supporting-passage index is among these five results.
            hits.append(relevant_index in ordered_documents[row_index])
        # Averaging booleans gives the fraction of successful retrievals.
        return float(np.mean(hits))


def plot_results(result: Mapping[str, Any], output_path: Path) -> None:
    """Plot the question-only and contrastive Hit@5 percentages."""
    # Convert the two Hit@5 fractions into percentages.
    percentages = [
        100 * result["generic_query"]["hit_at_5"],
        100 * result["contrastive_query"]["hit_at_5"],
    ]
    # Create one compact chart with one bar for each query type.
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    # Gray represents the baseline and blue represents contrastive retrieval.
    bars = axis.bar(
        ["Question only", "Contrastive"],
        percentages,
        color=["#64748b", "#2563eb"],
        width=0.58,
    )
    # Print each exact percentage above its bar.
    axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11)
    # Percentages always live between zero and one hundred.
    axis.set_ylim(0, 100)
    # Explain the one metric shown in the figure.
    axis.set_ylabel("Hit@5: supporting passage retrieved (%)")
    # State the comparison directly in the title.
    axis.set_title("Contrastive queries improve Hit@5")
    # Faint horizontal lines make the two values easier to compare.
    axis.grid(axis="y", alpha=0.2)

    # Include the final evaluation sample size in the overall title.
    eligible = result["counts"]["eligible_non_singletons"]
    figure.suptitle(f"Conformal retrieval MVP (n={eligible})", fontsize=14)
    # State the controlled-experiment assumption at the bottom of the figure.
    figure.text(
        0.5,
        0.01,
        "Same corpus, scikit-learn retriever, and depth; only the query changes.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    # Reserve room for the title and footnote while fixing chart spacing.
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
    records: Sequence[Mapping[str, Any]], alpha: float
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
        # Without a hint, there is no known relevant passage whose hit we can score.
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
    retriever = SklearnTfidfRetriever(documents)
    # Step 6: measure baseline success within the first five passages.
    baseline_hit_at_5 = retriever.hit_at_five(baseline_queries, relevant_indices)
    # Measure contrastive success with the exact same five-passage depth.
    contrastive_hit_at_5 = retriever.hit_at_five(contrastive_queries, relevant_indices)

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
    # A positive difference means contrastive queries retrieved evidence more often.
    hit_at_5_delta = contrastive_hit_at_5 - baseline_hit_at_5

    # Step 7: assemble all settings, counts, and metrics into serializable values.
    return {
        # These fields make the run reproducible and interpretable later.
        "configuration": {
            "alpha": alpha,
            "qhat": qhat,
            "split": "even record id = calibration; odd record id = evaluation",
            "retriever": "scikit-learn TF-IDF + cosine NearestNeighbors",
            "retrieval_depth": RETRIEVAL_DEPTH,
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
        # Hit@5 is the single retrieval metric retained for this MVP.
        "generic_query": {"hit_at_5": baseline_hit_at_5},
        "contrastive_query": {"hit_at_5": contrastive_hit_at_5},
        "contrastive_minus_generic": {"hit_at_5": hit_at_5_delta},
        # Prevent this retrieval result from being read as an end-to-end guarantee.
        "scope_note": (
            "Retrieval-only MVP: supporting-hint Hit@5 is measured before adding a "
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


def main() -> None:
    """Read CLI arguments, run the experiment, and save requested artifacts."""
    # Use this module's opening explanation as the --help description.
    parser = argparse.ArgumentParser(description=__doc__)
    # The tracked MMBench result pickle is the default input.
    parser.add_argument("--data", type=Path, default=Path("mmbench.pkl"))
    # Alpha=0.1 is the conformal error target used by the existing demo.
    parser.add_argument("--alpha", type=float, default=0.1)
    # --output is optional; when present, metrics are written as JSON here.
    parser.add_argument("--output", type=Path)
    # --plot is optional; when present, the Matplotlib PNG is written here.
    parser.add_argument("--plot", type=Path)
    # Parse the actual values provided by the user in the terminal.
    args = parser.parse_args()

    # Load the input once and pass all experiment settings to the core function.
    result = run_experiment(load_records(args.data), args.alpha)
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

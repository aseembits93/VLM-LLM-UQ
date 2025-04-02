import unittest

import numpy as np

from experiments.contrastive_retrieval import (
    TfidfRetriever,
    calibrate_qhat,
    contrastive_query,
    prediction_set,
    run_experiment,
)


def record(record_id, answer, logits, question, hint, option_a, option_b):
    return {
        "id": record_id,
        "answer": answer,
        "logits": np.asarray(logits, dtype=float),
        "question": question,
        "hint": hint,
        "A": option_a,
        "B": option_b,
        "C": "",
        "D": "",
        "E": "",
        "F": "",
    }


class ContrastiveRetrievalTest(unittest.TestCase):
    def test_prediction_set_matches_demo_boundary_rule(self):
        example = record(0, "A", [2, 1, 0, -1, -2, -3], "q", "h", "one", "two")
        qhat = calibrate_qhat([example], alpha=0.5)

        self.assertAlmostEqual(qhat, 0.633691322573722)
        self.assertEqual(prediction_set(example, qhat), ["A"])

    def test_contrastive_query_uses_only_surviving_options(self):
        example = record(0, "A", [1, 1, 0, 0, 0, 0], "Which?", "hint", "red", "blue")

        query = contrastive_query(example, ["B", "A", "F"])

        self.assertIn("Which?", query)
        self.assertIn("B: blue", query)
        self.assertIn("A: red", query)
        self.assertNotIn("F:", query)

    def test_retriever_ranks_matching_document_first(self):
        retriever = TfidfRetriever(["cats purr", "dogs bark"])

        self.assertEqual(retriever.ranks(["which dogs bark"], [1]), [1])

    def test_end_to_end_experiment_reports_paired_queries(self):
        records = [
            record(
                0,
                "F",
                [3, 2, 1, 0, -1, -2],
                "calibration one",
                "red fruit apple",
                "apple",
                "berry",
            ),
            record(
                2,
                "F",
                [2, 3, 1, 0, -1, -2],
                "calibration two",
                "blue fruit berry",
                "apple",
                "berry",
            ),
            record(
                1,
                "A",
                [1, 1, 0, -1, -2, -3],
                "Which fruit?",
                "red fruit apple",
                "apple",
                "berry",
            ),
            record(
                3,
                "B",
                [1, 1, 0, -1, -2, -3],
                "Which fruit?",
                "blue fruit berry",
                "apple",
                "berry",
            ),
        ]

        result = run_experiment(records, alpha=0.5, top_k=[1, 2])

        self.assertEqual(result["counts"]["calibration"], 2)
        self.assertEqual(result["counts"]["evaluation"], 2)
        self.assertEqual(result["counts"]["eligible_non_singletons"], 2)
        self.assertIn("hit_at_1", result["generic_query"])
        self.assertIn("hit_at_2", result["contrastive_query"])


if __name__ == "__main__":
    unittest.main()

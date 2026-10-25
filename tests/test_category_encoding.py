"""Checks for training-only category/subcategory encodings and history matching."""
import unittest

import pandas as pd

from src.category_encoding import (RARE, candidate_history_subcategory_match, encode_categories,
                                   fit_category_vocab)


def make_news(rows):
    frame = pd.DataFrame(rows, columns=["news_id", "category", "subcategory"])
    frame["metadata_missing"] = False
    return frame


class CategoryEncodingTests(unittest.TestCase):
    def setUp(self):
        rows = [(f"N{i}", "sports", "football") for i in range(6)]
        rows += [(f"N1{i}", "news", "video") for i in range(5)]
        rows += [("N20", "video", "video"), ("N21", "sports", "golf"), ("N22", "news", "local")]
        self.train = make_news(rows)
        self.vocab = fit_category_vocab(self.train, min_articles=5)

    def test_category_ids_are_sorted_with_zero_unknown(self):
        self.assertEqual(self.vocab["category"]["category"].tolist(), ["__UNKNOWN__", "news", "sports", "video"])
        self.assertEqual(self.vocab["category"]["category_id"].tolist(), [0, 1, 2, 3])

    def test_pairs_keep_parent_category_and_rare_pairs_merge(self):
        encoded = encode_categories(self.train, self.vocab).set_index("news_id")
        self.assertNotEqual(encoded.at["N10", "subcategory_id"], encoded.at["N20", "subcategory_id"])
        self.assertEqual(encoded.at["N22", "subcategory_backoff"], 1)
        rare = self.vocab["subcategory"].set_index(["category", "bucket"])["subcategory_id"]
        self.assertEqual(encoded.at["N21", "subcategory_id"], rare[("sports", RARE)])
        self.assertEqual(encoded.at["N0", "subcategory_backoff"], 0)

    def test_unseen_values(self):
        later = make_news([("N90", "games", "puzzles"), ("N91", "sports", "cricket"), ("N92", "news", "video")])
        later.loc[2, "metadata_missing"] = True
        encoded = encode_categories(later, self.vocab).set_index("news_id")
        self.assertEqual(encoded.at["N90", "category_id"], 0)
        self.assertEqual(encoded.at["N90", "subcategory_id"], 0)
        self.assertEqual(encoded.at["N91", "subcategory_id"], self.vocab["subcategory"].set_index(
            ["category", "bucket"]).at[("sports", RARE), "subcategory_id"])
        self.assertEqual(encoded.at["N92", "category_unknown"], 1)

    def test_history_match_uses_pairs_and_keeps_order(self):
        news = make_news([("A", "news", "video"), ("B", "video", "video"), ("C", "news", "__UNKNOWN__")])
        events = pd.DataFrame({"event_key": ["e1", "e2"], "history_ids": [["A", "A", "B", "C"], []]})
        candidates = pd.DataFrame({"event_key": ["e1", "e1", "e1", "e2"], "candidate_position": [0, 1, 2, 0],
                                   "news_id": ["B", "A", "C", "A"]})
        out = candidate_history_subcategory_match(events, candidates, news)
        self.assertEqual(out["news_id"].tolist(), ["B", "A", "C", "A"])
        self.assertAlmostEqual(out["candidate_history_subcategory_share"].iloc[0], 1 / 3, places=6)
        self.assertAlmostEqual(out["candidate_history_subcategory_share"].iloc[1], 2 / 3, places=6)
        self.assertEqual(out["candidate_history_subcategory_share"].iloc[2], 0)
        self.assertEqual(out["history_subcategory_missing"].tolist(), [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()

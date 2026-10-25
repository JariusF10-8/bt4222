"""Checks for entity prominence features and weighted entity aggregation."""
import unittest

import numpy as np
import pandas as pd

from src.entity_prominence import (aggregate_entity_vectors, article_prominence_features,
                                   entity_mentions, is_text_start, mention_weights)


def entity(qid, kind, offsets, confidence=1.0):
    return {"WikidataId": qid, "Type": kind, "Confidence": confidence, "OccurrenceOffsets": offsets,
            "SurfaceForms": ["x"] * len(offsets), "Label": qid}


class EntityProminenceTests(unittest.TestCase):
    def setUp(self):
        self.news = pd.DataFrame({
            "news_id": ["N1", "N2", "N3"],
            "title": ["Apple sues Google", "'Trump' visits Paris", "No entities here"],
            "abstract": ["Tim Cook said so.", "", ""],
            "title_entities_parsed": [[entity("Q1", "O", [0]), entity("Q2", "O", [11])],
                                      [entity("Q3", "P", [1]), entity("Q4", "G", [15])], []],
            "abstract_entities_parsed": [[entity("Q5", "P", [0]), entity("Q1", "O", [])], [], []],
        })
        self.mentions = entity_mentions(self.news)

    def test_text_start_allows_leading_punctuation_only(self):
        self.assertTrue(is_text_start("Apple", 0))
        self.assertTrue(is_text_start("'Trump' visits", 1))
        self.assertFalse(is_text_start("The Apple", 4))
        self.assertFalse(is_text_start("Apple", -1))

    def test_article_features(self):
        features = article_prominence_features(self.news, self.mentions).set_index("news_id")
        self.assertEqual(features["has_entity_at_title_start"].tolist(), [1, 1, 0])
        self.assertEqual(features["title_start_type_group"].tolist(), ["organisation", "person", "none"])
        self.assertAlmostEqual(features.at["N2", "title_entity_first_position"], 1 / len("'Trump' visits Paris"))
        self.assertEqual(features.at["N3", "title_entity_absent"], 1)
        self.assertEqual(features.at["N3", "title_entity_first_position"], 1.0)
        self.assertEqual(features.at["N2", "abstract_entity_absent"], 1)

    def test_uniform_weights_give_plain_mean_and_entities_count_once(self):
        index = {"Q1": 0, "Q2": 1, "Q5": 2}
        matrix = np.array([[1, 0], [0, 1], [3, 3]], dtype="float32")
        uniform = mention_weights(self.mentions, abstract_weight=1.0, decay=0.0)
        vectors, missing = aggregate_entity_vectors(["N1", "N2", "N3"], self.mentions, uniform, index, matrix)
        np.testing.assert_allclose(vectors[0], [4 / 3, 4 / 3], rtol=1e-6)
        self.assertEqual(missing.tolist(), [0, 1, 1])
        weighted = mention_weights(self.mentions, abstract_weight=0.5, decay=1.0)
        vectors_w, _ = aggregate_entity_vectors(["N1"], self.mentions, weighted, index, matrix)
        self.assertLess(vectors_w[0][0] - vectors_w[0][1], 1.0)
        self.assertNotAlmostEqual(float(vectors_w[0][0]), float(vectors[0][0]))

    def test_unknown_offsets_are_least_prominent(self):
        unknown = self.mentions.loc[self.mentions["first_offset"].lt(0)]
        self.assertEqual(len(unknown), 1)
        weights = mention_weights(self.mentions, abstract_weight=1.0, decay=1.0)
        self.assertAlmostEqual(float(weights.loc[unknown.index[0]]), float(np.exp(-1.0)))


if __name__ == "__main__":
    unittest.main()

"""Checks for impression-based negative sampling."""
import unittest

import pandas as pd

from src.sampling import sample_negatives


def make_candidates(impressions):
    rows = [
        {"event_key": key, "candidate_position": position, "news_id": f"N{key}{position}", "label": label}
        for key, labels in impressions.items() for position, label in enumerate(labels)
    ]
    frame = pd.DataFrame(rows)
    frame["event_key"] = frame["event_key"].astype("string")
    frame["news_id"] = frame["news_id"].astype("string")
    frame["candidate_position"] = frame["candidate_position"].astype("int32")
    frame["label"] = frame["label"].astype("int8")
    return frame


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.candidates = make_candidates({
            "a": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # two clicks, eight negatives
            "b": [0, 1, 0],                       # fewer than k negatives
            "c": [1, 1],                          # no negatives: dropped
            "d": [0, 0, 0],                       # no clicks: contributes nothing
        })

    def test_groups_hold_each_click_and_k_negatives_from_its_impression(self):
        sampled = sample_negatives(self.candidates, k=4, seed=0)
        self.assertTrue(sampled.groupby("group_id").size().eq(5).all())
        clicks = sampled.loc[sampled["slot"].eq(0)]
        self.assertTrue(clicks["label"].eq(1).all())
        self.assertEqual(sorted(clicks["news_id"]), ["Na0", "Na6", "Nb1"])
        self.assertTrue(sampled.loc[sampled["slot"].gt(0), "label"].eq(0).all())
        self.assertTrue(sampled.groupby("group_id")["event_key"].nunique().eq(1).all())
        self.assertNotIn("c", set(sampled["event_key"]))
        self.assertNotIn("d", set(sampled["event_key"]))

    def test_without_replacement_when_enough_negatives(self):
        sampled = sample_negatives(self.candidates, k=4, seed=0)
        rich = sampled.loc[sampled["event_key"].eq("a") & sampled["slot"].gt(0)]
        self.assertTrue(rich.groupby("group_id")["news_id"].nunique().eq(4).all())
        # Exactly k negatives available: still distinct.
        exact = sample_negatives(make_candidates({"e": [1, 0, 0, 0, 0]}), k=4, seed=3)
        self.assertEqual(exact.loc[exact["slot"].gt(0), "news_id"].nunique(), 4)

    def test_with_replacement_when_too_few_negatives(self):
        sampled = sample_negatives(self.candidates, k=4, seed=0)
        sparse = sampled.loc[sampled["event_key"].eq("b") & sampled["slot"].gt(0)]
        self.assertEqual(len(sparse), 4)
        self.assertTrue(set(sparse["news_id"]) <= {"Nb0", "Nb2"})

    def test_seed_controls_the_draw(self):
        first = sample_negatives(self.candidates, seed=0)
        shuffled = self.candidates.sample(frac=1, random_state=1)
        pd.testing.assert_frame_equal(first, sample_negatives(shuffled, seed=0))
        draws = {tuple(sample_negatives(self.candidates, seed=seed)["news_id"]) for seed in range(5)}
        self.assertGreater(len(draws), 1)


if __name__ == "__main__":
    unittest.main()

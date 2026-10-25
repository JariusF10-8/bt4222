"""Leakage and correctness checks for popularity/CTR features."""
import unittest

import numpy as np
import pandas as pd

from src.history_features import FEATURE_COLUMNS, build_exposure_log, compute_history_features

START = pd.Timestamp("2019-11-09")


def make_partitions(seed=0):
    rng = np.random.default_rng(seed)
    parts = {}
    spans = {"train": (0, 40), "validation": (41, 60), "test": (61, 80)}
    for name, (lo, hi) in spans.items():
        events, candidates = [], []
        for i in range(25):
            key = f"{name}:{i}"
            hours = rng.integers(lo, hi)
            events.append({"event_key": key, "user_id": f"U{rng.integers(5)}",
                           "timestamp": START + pd.Timedelta(hours=int(hours), minutes=int(rng.integers(60)))})
            shown = rng.choice(8, size=4, replace=False)
            labels = rng.permutation([1, 0, 0, 0])
            for position, (article, label) in enumerate(zip(shown, labels)):
                candidates.append({"event_key": key, "candidate_position": position,
                                   "news_id": f"N{article}", "label": int(label)})
        parts[name] = (pd.DataFrame(events), pd.DataFrame(candidates))
    keys = {name: pd.Series({f"N{a}": f"cat/sub{a % 3}" for a in range(8)}) for name in parts}
    return parts, keys


def brute_force(log, row, mode):
    t = log.at[row, "t"]
    upper = t
    if mode == "frozen" and log.at[row, "partition"] != 0:
        upper = min(t, log.loc[log["partition"].eq(log.at[row, "partition"]), "t"].min())
    past = log.loc[log["t"].lt(upper)]
    same_article = past.loc[past["article"].eq(log.at[row, "article"])]
    window = same_article.loc[same_article["t"].ge(t - 24 * 3600)]
    pair = past.loc[past["article"].eq(log.at[row, "article"]) & past["user"].eq(log.at[row, "user"])]
    user = past.loc[past["user"].eq(log.at[row, "user"])]
    return {
        "art_prior_exposures": len(same_article), "art_prior_clicks": int(same_article["label"].sum()),
        "art_exposures_24h": len(window), "art_clicks_24h": int(window["label"].sum()),
        "art_hours_since_first_exposure": (t - same_article["t"].min()) / 3600 if len(same_article) else 0.0,
        "user_article_prior_exposures": len(pair), "user_article_prior_clicks": int(pair["label"].sum()),
        "user_prior_events": user["event"].nunique(), "user_prior_clicks": int(user["label"].sum()),
    }


class HistoryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.parts, self.keys = make_partitions()
        self.log = build_exposure_log(self.parts, self.keys)

    def test_matches_brute_force_in_both_modes(self):
        for mode in ("online", "frozen"):
            features = compute_history_features(self.log, mode=mode)
            for row in range(0, len(self.log), 7):
                expected = brute_force(self.log, row, mode)
                for column, value in expected.items():
                    self.assertAlmostEqual(float(features.at[row, column]), float(value), places=3,
                                           msg=f"{mode} {column} row {row}")

    def test_current_and_future_labels_never_change_features(self):
        cut = int(self.log["t"].quantile(0.5))
        perturbed = self.log.copy()
        later = perturbed["t"].ge(cut)
        perturbed.loc[later, "label"] = 1 - perturbed.loc[later, "label"]
        for mode in ("online", "frozen"):
            base = compute_history_features(self.log, mode=mode, prior_ctr=0.25)
            changed = compute_history_features(perturbed, mode=mode, prior_ctr=0.25)
            earlier = ~later
            pd.testing.assert_frame_equal(base.loc[earlier | self.log["t"].eq(cut)].reset_index(drop=True),
                                          changed.loc[earlier | self.log["t"].eq(cut)].reset_index(drop=True))

    def test_frozen_partitions_ignore_their_own_period(self):
        features = compute_history_features(self.log, mode="frozen", prior_ctr=0.25)
        for name, flipped in [(1, [1, 2]), (2, [2])]:
            perturbed = self.log.copy()
            mask = perturbed["partition"].isin(flipped)
            perturbed.loc[mask, "label"] = 1 - perturbed.loc[mask, "label"]
            changed = compute_history_features(perturbed, mode="frozen", prior_ctr=0.25)
            rows = self.log["partition"].eq(name)
            pd.testing.assert_frame_equal(features.loc[rows], changed.loc[rows])
        validation = self.log["partition"].eq(1)
        train_rows = int(self.log["partition"].eq(0).sum())
        self.assertTrue(features.loc[validation, "user_article_prior_exposures"].le(train_rows).all())
        online = compute_history_features(self.log, mode="online", prior_ctr=0.25)
        self.assertGreater(int(online.loc[validation, "art_prior_exposures"].sum()),
                           int(features.loc[validation, "art_prior_exposures"].sum()))

    def test_output_alignment_and_columns(self):
        features = compute_history_features(self.log)
        self.assertEqual(len(features), len(self.log))
        self.assertEqual(features.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(self.log.attrs["partitions"], ["train", "validation", "test"])
        sizes = self.log.groupby("partition").size().tolist()
        self.assertEqual(sizes, [len(c) for _, c in self.parts.values()])
        labels = pd.concat([c["label"] for _, c in self.parts.values()], ignore_index=True)
        self.assertEqual(self.log["label"].tolist(), labels.tolist())


if __name__ == "__main__":
    unittest.main()

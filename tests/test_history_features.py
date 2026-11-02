"""Leakage and correctness checks for popularity/CTR features."""
import unittest

import numpy as np
import pandas as pd

from src.history_features import (FEATURE_COLUMNS, MODES, build_exposure_log, compute_history_features,
                                  mix_by_impression)

START = pd.Timestamp("2019-11-09")


def make_partitions(seed=0, spans=None):
    rng = np.random.default_rng(seed)
    parts = {}
    spans = spans or {"train": (0, 40), "validation": (41, 60), "test": (61, 80)}
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


def brute_force(log, row, mode, origin):
    t = log.at[row, "t"]
    upper = t
    if mode == "frozen" and log.at[row, "partition"] != 0:
        first = log.loc[log["partition"].eq(log.at[row, "partition"]), "t"].min()
        midnight = (origin + pd.Timedelta(seconds=int(first))).normalize()
        upper = min(t, max(0, int((midnight - origin) / pd.Timedelta(seconds=1))))
    if mode == "nightly":
        midnight = (origin + pd.Timedelta(seconds=int(t))).normalize()
        upper = max(0, int((midnight - origin) / pd.Timedelta(seconds=1)))
    past = log.loc[log["t"].lt(upper)]
    same_article = past.loc[past["article"].eq(log.at[row, "article"])]
    window = same_article.loc[same_article["t"].ge(upper - 24 * 3600)]
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
        self.origin = min(events["timestamp"].min() for events, _ in self.parts.values())

    def test_matches_brute_force_in_every_mode(self):
        for mode in MODES:
            features = compute_history_features(self.log, mode=mode)
            for row in range(0, len(self.log), 7):
                expected = brute_force(self.log, row, mode, self.origin)
                for column, value in expected.items():
                    self.assertAlmostEqual(float(features.at[row, column]), float(value), places=3,
                                           msg=f"{mode} {column} row {row}")

    def test_current_and_future_labels_never_change_features(self):
        cut = int(self.log["t"].quantile(0.5))
        perturbed = self.log.copy()
        later = perturbed["t"].ge(cut)
        perturbed.loc[later, "label"] = 1 - perturbed.loc[later, "label"]
        for mode in MODES:
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

    def test_nightly_never_uses_the_same_day(self):
        perturbed = self.log.copy()
        day = self.log["day_start"]
        for value in day.unique():
            rows = day.eq(value)
            changed_day = perturbed.copy()
            changed_day.loc[rows, "label"] = 1 - changed_day.loc[rows, "label"]
            base = compute_history_features(self.log, mode="nightly", prior_ctr=0.25)
            changed = compute_history_features(changed_day, mode="nightly", prior_ctr=0.25)
            pd.testing.assert_frame_equal(base.loc[rows], changed.loc[rows])

    def test_nightly_equals_frozen_for_a_one_day_test_partition(self):
        parts, keys = make_partitions(seed=3, spans={"train": (0, 40), "validation": (41, 71), "test": (72, 90)})
        # Real data case: an exposure exactly 24 hours before the test midnight cutoff.
        validation_events, validation_candidates = parts["validation"]
        test_events, test_candidates = parts["test"]
        test_events.loc[0, "timestamp"] = START + pd.Timedelta(hours=72, seconds=1)
        validation_events.loc[0, "timestamp"] = START + pd.Timedelta(hours=48)
        shown = test_candidates.loc[test_candidates["event_key"].eq(test_events.loc[0, "event_key"]), "news_id"].to_numpy()
        first = validation_candidates["event_key"].eq(validation_events.loc[0, "event_key"])
        validation_candidates.loc[first, "news_id"] = shown
        log = build_exposure_log(parts, keys)
        test = log["partition"].eq(2)
        nightly = compute_history_features(log, mode="nightly", prior_ctr=0.25)
        frozen = compute_history_features(log, mode="frozen", prior_ctr=0.25)
        pd.testing.assert_frame_equal(nightly.loc[test], frozen.loc[test])
        self.assertFalse(nightly.loc[log["partition"].eq(0)].equals(frozen.loc[log["partition"].eq(0)]))

    def test_mixing_keeps_whole_impressions_together(self):
        train = self.log["partition"].eq(0).to_numpy()
        fresh = compute_history_features(self.log, mode="online").loc[train]
        stale = compute_history_features(self.log, mode="nightly").loc[train]
        events = self.log.loc[train, "event"].to_numpy()
        mixed, stale_rows = mix_by_impression(fresh, stale, events, seed=1)
        self.assertEqual(pd.Series(stale_rows).groupby(events).nunique().max(), 1)
        np.testing.assert_array_equal(mixed.loc[stale_rows].to_numpy(), stale.loc[stale_rows].to_numpy())
        np.testing.assert_array_equal(mixed.loc[~stale_rows].to_numpy(), fresh.loc[~stale_rows].to_numpy())
        self.assertTrue(0 < stale_rows.mean() < 1)
        again, _ = mix_by_impression(fresh, stale, events, seed=1)
        pd.testing.assert_frame_equal(mixed, again)
        self.assertEqual(list(mixed.dtypes), list(fresh.dtypes))

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

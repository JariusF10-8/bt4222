"""Impression-based negative sampling for training; never apply to validation/test.

Compute statistical features (article CTR, popularity, user-category affinity)
on the full train candidates before sampling: sampling fixes the click share
at 1/(k+1), which would inflate any rate computed afterwards. Validation and
test are scored per impression on all candidates (AUC, MRR, nDCG@5/10).
"""
import numpy as np
import pandas as pd

from src.preprocessing import require

OUTPUT_COLUMNS = ["group_id", "event_key", "news_id", "candidate_position", "label", "slot"]


def sample_negatives(candidates, k=4, seed=0):
    """One group per clicked candidate: the click + k non-clicks from the same impression.

    Slot 0 is the click and slots 1..k are negatives. Use the rows directly for
    pointwise BCE, or group by group_id for a listwise softmax (target slot 0).
    Negatives are drawn without replacement when the impression has at least k,
    otherwise with replacement. Clicks in impressions without negatives are
    dropped. Pass seed=base_seed + epoch to draw fresh negatives each epoch.
    """
    require(k >= 1, "k must be at least 1")
    require(candidates["label"].isin([0, 1]).all(), "Labels must be 0 or 1")
    rng = np.random.default_rng(seed)
    # Sort first so the draw depends only on content and seed, not row order.
    order = ["event_key", "candidate_position"]
    positives = candidates.loc[candidates["label"].eq(1)].sort_values(order, kind="stable")
    negatives = candidates.loc[candidates["label"].eq(0)].sort_values(order, kind="stable").reset_index(drop=True)
    counts = negatives.groupby("event_key", sort=False).size()
    starts = counts.cumsum() - counts

    available = positives["event_key"].map(counts).fillna(0).astype("int64")
    dropped = positives.loc[available.eq(0), "event_key"]
    if len(dropped):
        print(f"Dropped {len(dropped):,} clicks in {dropped.nunique():,} impressions without negatives", flush=True)
    positives = positives.loc[available.gt(0)].reset_index(drop=True)
    n = positives["event_key"].map(counts).to_numpy("int64")[:, None]
    offsets = rng.integers(0, n, size=(len(positives), k))
    # Redraw rows with repeats wherever enough distinct negatives exist.
    while True:
        ordered = np.sort(offsets, axis=1)
        repeated = (ordered[:, 1:] == ordered[:, :-1]).any(axis=1) & (n[:, 0] >= k)
        if not repeated.any():
            break
        offsets[repeated] = rng.integers(0, n[repeated], size=(int(repeated.sum()), k))

    group_ids = np.arange(len(positives), dtype="int64")
    clicked = positives.assign(group_id=group_ids, slot=0)
    picks = (positives["event_key"].map(starts).to_numpy("int64")[:, None] + offsets).ravel()
    skipped = negatives.iloc[picks].assign(
        group_id=np.repeat(group_ids, k), slot=np.tile(np.arange(1, k + 1), len(positives)))
    sampled = pd.concat([clicked, skipped], ignore_index=True)
    sampled["slot"] = sampled["slot"].astype("int8")
    return sampled.sort_values(["group_id", "slot"], kind="stable")[OUTPUT_COLUMNS].reset_index(drop=True)

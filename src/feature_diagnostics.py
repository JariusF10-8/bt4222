"""Per-impression ranking metrics for quick feature checks on validation data.

Definitions follow the MIND evaluation script: AUC, MRR, nDCG@5 and nDCG@10
are computed inside each impression and then averaged over impressions. Ties
in AUC count one half; ties in the rank-based metrics are broken at random
with a fixed seed so equal scores never favour clicked candidates.
"""
import numpy as np
import pandas as pd


def impression_metrics(frame, score, group="event_key", label="label", seed=0, per_impression=False):
    """Mean metrics over impressions with at least one click and one non-click.

    With per_impression=True, return one row per impression instead, which
    supports paired comparisons between two scores on the same impressions.
    """
    scores = frame[score].to_numpy(dtype="float64")
    scores = np.where(np.isnan(scores), np.nanmin(scores) if np.isfinite(scores).any() else 0.0, scores)
    data = pd.DataFrame({"g": frame[group].to_numpy(), "y": frame[label].to_numpy(dtype="int64"), "s": scores})
    rng = np.random.default_rng(seed)
    data["tie"] = rng.random(len(data))
    data = data.sort_values(["g", "s", "tie"], ascending=[True, False, False], kind="stable")
    data["rank"] = data.groupby("g", sort=False).cumcount() + 1
    data["avg_rank"] = data.groupby("g", sort=False)["s"].rank(method="average", ascending=True)
    grouped = data.groupby("g", sort=False)
    n = grouped["y"].size()
    pos = grouped["y"].sum()
    valid = (pos > 0) & (pos < n)
    data["rr"] = data["y"] / data["rank"]
    data["gain"] = data["y"] / np.log2(data["rank"] + 1)
    data["pos_rank"] = data["y"] * data["avg_rank"]
    sums = data.groupby("g", sort=False).agg(rr=("rr", "sum"), pr=("pos_rank", "sum"))
    dcg = {k: data.loc[data["rank"].le(k)].groupby("g", sort=False)["gain"].sum().reindex(n.index, fill_value=0.0)
           for k in (5, 10)}
    discount = np.concatenate([[0.0], np.cumsum(1 / np.log2(np.arange(2, 12)))])
    auc = (sums["pr"] - pos * (pos + 1) / 2) / (pos * (n - pos))
    table = pd.DataFrame({"AUC": auc, "MRR": sums["rr"] / pos})
    for k in (5, 10):
        table[f"nDCG@{k}"] = dcg[k] / discount[np.minimum(pos.to_numpy(), k)]
    table = table.loc[valid]
    if per_impression:
        return table
    result = {"impressions": int(valid.sum()), **table.mean().to_dict()}
    return {key: (round(float(v), 4) if isinstance(v, float) else v) for key, v in result.items()}


def paired_difference(frame, score_a, score_b, metric="AUC", **kwargs):
    """Mean per-impression difference (b minus a) with a normal 95% interval."""
    a = impression_metrics(frame, score_a, per_impression=True, **kwargs)[metric]
    b = impression_metrics(frame, score_b, per_impression=True, **kwargs)[metric]
    diff = (b - a.reindex(b.index)).dropna()
    half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    return {"metric": metric, "mean_difference": round(float(diff.mean()), 4),
            "ci_low": round(float(diff.mean() - half), 4), "ci_high": round(float(diff.mean() + half), 4),
            "impressions": int(len(diff))}


def univariate_table(frame, columns, sign=None, **kwargs):
    """Score candidates by one feature at a time; sign=-1 flips features expected to lower clicks."""
    sign = sign or {}
    rows = []
    for column in columns:
        scored = frame[["event_key", "label"]].assign(_s=frame[column].astype("float64") * sign.get(column, 1))
        rows.append({"feature": column, "direction": "lower is better" if sign.get(column, 1) < 0 else "higher is better",
                     **impression_metrics(scored, "_s", **kwargs)})
    return pd.DataFrame(rows)

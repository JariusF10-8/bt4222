"""Article popularity and click-through features from interactions strictly before each impression.

Two modes share one implementation:
- online: every logged exposure and click with time < t, from any partition.
- frozen: training rows use time < t; rows in later partitions only see
  interactions before their partition starts (the interim-report rule).
The impression being scored never contributes, because it has time t.
"""
import numpy as np
import pandas as pd

from src.preprocessing import require

WINDOW_HOURS = (1, 6, 24)
SMOOTHING = {"subcategory": 100.0, "article": 20.0, "user": 20.0}

ARTICLE_COLUMNS = [
    "art_prior_exposures", "art_prior_clicks", "art_prior_ctr",
    *[f"art_{kind}_{h}h" for h in WINDOW_HOURS for kind in ("exposures", "clicks")],
    "art_ctr_24h", "art_hours_since_first_exposure", "art_seen_before",
]
SUBCATEGORY_COLUMNS = ["subcat_prior_ctr"]
USER_COLUMNS = ["user_prior_events", "user_prior_exposures", "user_prior_clicks", "user_prior_ctr"]
USER_ARTICLE_COLUMNS = ["user_article_prior_exposures", "user_article_prior_clicks", "user_article_skipped_before"]
FEATURE_COLUMNS = ARTICLE_COLUMNS + SUBCATEGORY_COLUMNS + USER_COLUMNS + USER_ARTICLE_COLUMNS
FLAG_COLUMNS = ("art_seen_before", "user_article_skipped_before")
FLOAT_COLUMNS = ("art_prior_ctr", "art_ctr_24h", "art_hours_since_first_exposure", "subcat_prior_ctr", "user_prior_ctr")


def _distinct(values):
    """Distinct string keys plus each row's index into them (fast for categoricals)."""
    series = pd.Series(values, copy=False)
    if isinstance(series.dtype, pd.CategoricalDtype):
        codes = series.cat.codes.to_numpy()
        require((codes >= 0).all(), "Missing key in a categorical column")
        return series.cat.categories.astype(str).to_numpy(), codes
    codes, uniques = pd.factorize(series.astype(str))
    return np.asarray(uniques, dtype=object), codes


def _codes(values, table):
    keys, rows = _distinct(values)
    for key in keys:
        if key not in table:
            table[key] = len(table)
    return pd.Series(table).reindex(keys).to_numpy().astype("int64")[rows]


def build_exposure_log(partitions, subcategory_keys=None):
    """Compact integer log with one row per candidate exposure, partitions in time order.

    partitions maps name -> (events, candidates); rows keep each partition's
    candidate order, so features can be re-attached to its candidates table.
    subcategory_keys optionally maps name -> Series(news_id -> raw subcategory key).
    Unknown subcategories use the key "" and fall back to the global rate.
    """
    tables = {"article": {}, "user": {}, "subcat": {"": 0}}
    frames, previous_end, event_offset, origin = [], None, 0, None
    for index, (name, (events, candidates)) in enumerate(partitions.items()):
        require(events["event_key"].is_unique, f"Repeated event_key in {name}")
        event_pos = pd.Series(np.arange(len(events)), index=events["event_key"].astype(str).to_numpy())
        keys, rows = _distinct(candidates["event_key"])
        positions = event_pos.reindex(keys).to_numpy()
        require(not np.isnan(positions).any(), f"Candidates in {name} reference unknown events")
        event_of_row = positions.astype("int64")[rows]
        stamps = pd.to_datetime(events["timestamp"]).to_numpy()
        start, end = stamps.min(), stamps.max()
        require(previous_end is None or previous_end < start, f"{name} overlaps an earlier partition")
        previous_end = end
        origin = start if origin is None else origin
        users = _codes(events["user_id"], tables["user"])
        frame = pd.DataFrame({
            "partition": np.full(len(candidates), index, dtype="int8"),
            "event": event_of_row + event_offset,
            "article": _codes(candidates["news_id"], tables["article"]),
            "user": users[event_of_row],
            "label": candidates["label"].to_numpy(dtype="int64"),
            "timestamp": stamps[event_of_row],
        })
        if subcategory_keys is not None:
            keys, rows = _distinct(candidates["news_id"])
            sub = pd.Series(keys).map(subcategory_keys[name]).fillna("")
            frame["subcat"] = _codes(sub, tables["subcat"])[rows]
        else:
            frame["subcat"] = 0
        frames.append(frame)
        event_offset += len(events)
    log = pd.concat(frames, ignore_index=True)
    log["t"] = ((log.pop("timestamp") - origin) // pd.Timedelta(seconds=1)).astype("int64")
    log.attrs["partitions"] = list(partitions)
    return log


class _GroupStream:
    """Sorted (group, time) keys with cumulative clicks for O(log n) prior counts."""

    def __init__(self, group, t, clicks, span):
        self.span = span
        key = group * span + t
        order = np.argsort(key, kind="stable")
        self.keys = key[order]
        self.cum = np.concatenate([[0], np.cumsum(clicks[order])])

    def counts(self, group, upper, lower=None):
        base = group * self.span
        hi = np.searchsorted(self.keys, base + upper, "left")
        floor = base if lower is None else base + np.maximum(lower, 0)
        lo = np.minimum(np.searchsorted(self.keys, floor, "left"), hi)
        return hi - lo, self.cum[hi] - self.cum[lo]

    def first_time(self, group, upper):
        base = group * self.span
        start = np.searchsorted(self.keys, base, "left")
        hi = np.searchsorted(self.keys, base + upper, "left")
        seen = hi > start
        first = np.where(seen, self.keys[np.minimum(start, len(self.keys) - 1)] - base, -1)
        return seen, first


def query_times(log, mode, training_partition=0):
    require(mode in ("online", "frozen"), "mode must be 'online' or 'frozen'")
    t = log["t"].to_numpy()
    if mode == "online":
        return t
    starts = log.groupby("partition", sort=False)["t"].transform("min").to_numpy()
    return np.where(log["partition"].to_numpy() == training_partition, t, np.minimum(t, starts))


def compute_history_features(log, mode="online", prior_ctr=None, smoothing=None, training_partition=0):
    """Return FEATURE_COLUMNS aligned with the rows of log.

    prior_ctr defaults to the training-partition click rate, one constant.
    Rates shrink towards that constant (subcategory, user) and articles shrink
    towards their subcategory rate, so unseen articles fall back to their topic.
    """
    smoothing = {**SMOOTHING, **(smoothing or {})}
    train = log["partition"].to_numpy() == training_partition
    require(train.any(), "The training partition is needed for the prior click rate")
    clicks = log["label"].to_numpy()
    p0 = float(clicks[train].mean()) if prior_ctr is None else float(prior_ctr)
    t = log["t"].to_numpy()
    q = query_times(log, mode, training_partition)
    span = int(t.max()) + 1
    out = {}

    sub = log["subcat"].to_numpy().astype("int64")
    sub_exp, sub_clk = _GroupStream(sub, t, clicks, span).counts(sub, q)
    m = smoothing["subcategory"]
    sub_rate = np.where(sub == 0, p0, (sub_clk + m * p0) / (sub_exp + m))
    del sub_exp, sub_clk

    art = log["article"].to_numpy().astype("int64")
    stream = _GroupStream(art, t, clicks, span)
    m = smoothing["article"]
    exposures, hits = stream.counts(art, q)
    out["art_prior_exposures"], out["art_prior_clicks"] = exposures, hits
    out["art_prior_ctr"] = (hits + m * sub_rate) / (exposures + m)
    for h in WINDOW_HOURS:
        out[f"art_exposures_{h}h"], out[f"art_clicks_{h}h"] = stream.counts(art, q, t - 3600 * h)
    out["art_ctr_24h"] = (out["art_clicks_24h"] + m * sub_rate) / (out["art_exposures_24h"] + m)
    seen, first = stream.first_time(art, q)
    out["art_hours_since_first_exposure"] = np.where(seen, (t - first) / 3600.0, 0.0)
    out["art_seen_before"] = seen
    out["subcat_prior_ctr"] = sub_rate
    del stream

    user = log["user"].to_numpy().astype("int64")
    stream = _GroupStream(user, t, clicks, span)
    out["user_prior_exposures"], out["user_prior_clicks"] = stream.counts(user, q)
    m = smoothing["user"]
    out["user_prior_ctr"] = (out["user_prior_clicks"] + m * p0) / (out["user_prior_exposures"] + m)
    first_rows = ~pd.Series(log["event"].to_numpy()).duplicated().to_numpy()
    events = _GroupStream(user[first_rows], t[first_rows], np.zeros(int(first_rows.sum()), "int64"), span)
    out["user_prior_events"], _ = events.counts(user, q)
    del stream, events

    pair = pd.factorize(user * (int(art.max()) + 1) + art)[0].astype("int64")
    exposures, hits = _GroupStream(pair, t, clicks, span).counts(pair, q)
    out["user_article_prior_exposures"], out["user_article_prior_clicks"] = exposures, hits
    out["user_article_skipped_before"] = exposures > hits

    frame = pd.DataFrame(index=log.index)
    for column in FEATURE_COLUMNS:
        dtype = "int8" if column in FLAG_COLUMNS else "float32" if column in FLOAT_COLUMNS else "int32"
        frame[column] = np.asarray(out[column]).astype(dtype)
    return frame

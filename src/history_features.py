"""Popularity and click-through features using only interactions before each impression.

Each impression at time t gets a cutoff c <= t, and only interactions before c are counted.
online: c = t.
nightly: c = midnight at the start of the impression's day, in every partition.
frozen: training rows use c = t, later partitions use midnight of their first day.
Time windows (1, 6 and 24 hours) end at the cutoff.
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
    """One row per candidate exposure, partitions in time order and each partition in candidate order."""
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
    stamps = log.pop("timestamp")
    second = pd.Timedelta(seconds=1)
    log["t"] = ((stamps - origin) // second).astype("int64")
    log["day_start"] = ((stamps.dt.normalize() - origin) // second).clip(lower=0).astype("int64")
    log.attrs["partitions"] = list(partitions)
    log.attrs["origin"] = str(pd.Timestamp(origin))
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


MODES = ("online", "nightly", "frozen")


def query_times(log, mode, training_partition=0):
    require(mode in MODES, f"mode must be one of {MODES}")
    t = log["t"].to_numpy()
    if mode == "online":
        return t
    if mode == "nightly":
        return np.minimum(t, log["day_start"].to_numpy())
    starts = log.groupby("partition", sort=False)["day_start"].transform("min").to_numpy()
    return np.where(log["partition"].to_numpy() == training_partition, t, np.minimum(t, starts))


def compute_history_features(log, mode="online", prior_ctr=None, smoothing=None, training_partition=0):
    """Return FEATURE_COLUMNS aligned with the rows of log, with rates smoothed towards the training click rate."""
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
        out[f"art_exposures_{h}h"], out[f"art_clicks_{h}h"] = stream.counts(art, q, q - 3600 * h)
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


def mix_by_impression(fresh, stale, events, stale_share=0.5, seed=42):
    """Give each training impression either its fresh or its stale features; also return a per-row stale flag."""
    require(len(fresh) == len(stale) == len(events), "fresh, stale and events must align")
    require(list(fresh.columns) == list(stale.columns), "fresh and stale need the same columns")
    _, inverse = np.unique(np.asarray(events), return_inverse=True)
    stale_rows = np.random.default_rng(seed).random(inverse.max() + 1)[inverse] < stale_share
    mixed = pd.DataFrame(index=fresh.index)
    for column in fresh.columns:
        mixed[column] = np.where(stale_rows, stale[column].to_numpy(), fresh[column].to_numpy()).astype(fresh[column].dtype)
    return mixed, stale_rows

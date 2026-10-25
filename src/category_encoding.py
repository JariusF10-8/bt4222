"""Category and subcategory encodings fitted on training articles only.

category_id follows the same rule as feature_engineering_zp.ipynb: sorted
training categories get 1..K and 0 is reserved for unknown. Subcategories are
encoded as (category, subcategory) pairs because 13 subcategory names appear
under more than one category. Pairs with fewer than min_articles training
articles share one rare bucket per category.
"""
import numpy as np
import pandas as pd

from src.preprocessing import require

UNKNOWN = "__UNKNOWN__"
RARE = "__RARE__"


def _known(news):
    category = news["category"].astype("string").fillna("")
    return (~news["metadata_missing"].astype(bool)) & category.str.strip().ne("") & category.ne(UNKNOWN)


def fit_category_vocab(train_news, min_articles=5):
    require(min_articles >= 1, "min_articles must be at least 1")
    require(train_news["news_id"].is_unique, "Duplicate news_id in training news")
    known = train_news.loc[_known(train_news), ["category", "subcategory"]].astype(str)
    require(len(known) > 0, "No training articles with a known category")

    categories = sorted(known["category"].unique())
    category_vocab = pd.DataFrame({
        "category_id": np.arange(len(categories) + 1, dtype="int32"),
        "category": [UNKNOWN, *categories],
    })
    category_vocab["train_articles"] = category_vocab["category"].map(
        known["category"].value_counts()).fillna(0).astype("int64")

    pairs = known.loc[known["subcategory"].ne(UNKNOWN)].value_counts(["category", "subcategory"]).rename("train_articles").reset_index()
    pairs["bucket"] = np.where(pairs["train_articles"].ge(min_articles), pairs["subcategory"], RARE)
    buckets = (pairs.groupby(["category", "bucket"], as_index=False)
               .agg(train_articles=("train_articles", "sum"), merged_subcategories=("subcategory", "size"))
               .sort_values(["category", "bucket"], kind="stable").reset_index(drop=True))
    buckets.insert(0, "subcategory_id", np.arange(1, len(buckets) + 1, dtype="int32"))
    unknown_row = pd.DataFrame([{"subcategory_id": 0, "category": UNKNOWN, "bucket": UNKNOWN,
                                 "train_articles": 0, "merged_subcategories": 0}])
    subcategory_vocab = pd.concat([unknown_row, buckets], ignore_index=True)
    subcategory_vocab["subcategory_id"] = subcategory_vocab["subcategory_id"].astype("int32")
    pair_map = pairs[["category", "subcategory", "bucket", "train_articles"]].reset_index(drop=True)
    return {"category": category_vocab, "subcategory": subcategory_vocab,
            "pairs": pair_map, "min_articles": int(min_articles)}


def encode_categories(news, vocab):
    """One row per news_id with integer codes for embedding layers or tree models.

    Unseen categories map to 0. A known category with an unseen or rare
    subcategory maps to that category's rare bucket when the bucket exists,
    otherwise to 0, and subcategory_backoff records that it was not encoded
    individually.
    """
    require(news["news_id"].is_unique, "Duplicate news_id would multiply joins")
    out = news[["news_id"]].copy()
    known = _known(news)
    category = news["category"].astype(str)
    subcategory = news["subcategory"].astype(str)

    cat_ids = dict(zip(category_vocab_names(vocab), vocab["category"]["category_id"]))
    out["category_id"] = np.where(known, category.map(cat_ids).fillna(0), 0).astype("int32")

    sub_vocab = vocab["subcategory"]
    bucket_ids = {(c, b): i for c, b, i in zip(sub_vocab["category"], sub_vocab["bucket"], sub_vocab["subcategory_id"])}
    pair_bucket = {(c, s): b for c, s, b in zip(vocab["pairs"]["category"], vocab["pairs"]["subcategory"], vocab["pairs"]["bucket"])}
    codes, backoff = [], []
    for is_known, c, s in zip(known, category, subcategory):
        if not is_known or c not in cat_ids:
            codes.append(0)
            backoff.append(True)
            continue
        bucket = pair_bucket.get((c, s), RARE)
        code = bucket_ids.get((c, bucket), bucket_ids.get((c, RARE), 0))
        codes.append(code)
        backoff.append(bucket == RARE)
    out["subcategory_id"] = np.asarray(codes, dtype="int32")
    out["category_unknown"] = out["category_id"].eq(0).astype("int8")
    out["subcategory_backoff"] = np.asarray(backoff, dtype="int8")
    return out


def category_vocab_names(vocab):
    return vocab["category"]["category"].tolist()


def subcategory_key(news):
    """Raw (category, subcategory) label used for label-free history matching."""
    known = _known(news) & news["subcategory"].astype(str).ne(UNKNOWN)
    key = news["category"].astype(str) + "/" + news["subcategory"].astype(str)
    return key.where(known, "")


def candidate_history_subcategory_match(events, candidates, news):
    """Share of the impression's supplied history in the candidate's subcategory.

    Uses only the supplied prior history and article metadata, never labels.
    Matching uses raw (category, subcategory) pairs, so it needs no fitted
    vocabulary. Unknown subcategories never count as a match. Rows keep the
    candidates order.
    """
    key_of = dict(zip(news["news_id"].astype(str), subcategory_key(news)))
    hist = events[["event_key", "history_ids"]].explode("history_ids").dropna()
    hist["key"] = hist["history_ids"].astype(str).map(key_of).fillna("")
    hist = hist.loc[hist["key"].ne("")]
    known_total = hist.groupby("event_key").size().rename("history_subcategory_known")
    counts = hist.groupby(["event_key", "key"]).size().rename("candidate_history_subcategory_count").reset_index()

    out = candidates[["event_key", "candidate_position", "news_id"]].copy()
    out["key"] = out["news_id"].astype(str).map(key_of).fillna("")
    out = out.merge(counts, on=["event_key", "key"], how="left", validate="many_to_one", sort=False)
    out = out.merge(known_total, left_on="event_key", right_index=True, how="left", sort=False)
    require(len(out) == len(candidates), "Subcategory match join changed the candidate count")
    out["candidate_history_subcategory_count"] = out["candidate_history_subcategory_count"].fillna(0).astype("int32")
    total = out.pop("history_subcategory_known").fillna(0).astype("int32")
    out["history_subcategory_missing"] = total.eq(0).astype("int8")
    out["candidate_history_subcategory_share"] = np.where(
        total.gt(0), out["candidate_history_subcategory_count"] / total.clip(lower=1), 0.0).astype("float32")
    out.loc[out["key"].eq(""), ["candidate_history_subcategory_count", "candidate_history_subcategory_share"]] = 0
    return out.drop(columns="key")

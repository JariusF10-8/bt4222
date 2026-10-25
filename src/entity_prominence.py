"""Entity prominence from OccurrenceOffsets and position-weighted TransE aggregation.

Offsets are character positions in the raw title or abstract. Title and
abstract positions are kept separate and normalised by that text's length.
Earlier position is treated as a hypothesis about importance, so weighting
schemes are compared rather than assumed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import require

SECTIONS = ("title", "abstract")
TYPE_GROUPS = {"P": "person", "O": "organisation", "G": "place"}
MENTION_COLUMNS = ["news_id", "section", "wikidata_id", "entity_type", "confidence",
                   "first_offset", "occurrences", "text_chars", "position", "at_text_start"]


def is_text_start(text, offset):
    """True when no letter or digit precedes the offset (allows a leading quote)."""
    return offset >= 0 and not any(ch.isalnum() for ch in text[:offset])


def entity_mentions(news):
    """One row per entity annotation, with its earliest position in its own section."""
    rows = []
    for news_id, *fields in zip(news["news_id"], *[news[s] for s in SECTIONS],
                                *[news[f"{s}_entities_parsed"] for s in SECTIONS]):
        texts, parsed = fields[:2], fields[2:]
        for section, text, entities in zip(SECTIONS, texts, parsed):
            chars = len(text)
            for entity in entities:
                offsets = [o for o in entity.get("OccurrenceOffsets") or [] if isinstance(o, int) and o >= 0]
                first = min(offsets) if offsets else -1
                rows.append((
                    news_id, section, str(entity.get("WikidataId") or ""), str(entity.get("Type") or ""),
                    float(entity.get("Confidence", np.nan)), first, len(offsets), chars,
                    min(first / chars, 1.0) if first >= 0 and chars > 0 else np.nan,
                    is_text_start(text, first) if first >= 0 else False,
                ))
    mentions = pd.DataFrame(rows, columns=MENTION_COLUMNS)
    mentions["first_offset"] = mentions["first_offset"].astype("int32")
    return mentions


def article_prominence_features(news, mentions):
    """Per-article prominence features; absent sections get position 1.0 plus a flag."""
    out = news[["news_id"]].copy()
    title = mentions.loc[mentions["section"].eq("title") & mentions["first_offset"].ge(0)]
    lead = title.loc[title["at_text_start"]].sort_values(["news_id", "first_offset"], kind="stable")
    lead = lead.drop_duplicates("news_id").set_index("news_id")["entity_type"]
    out["has_entity_at_title_start"] = out["news_id"].isin(lead.index).astype("int8")
    lead_type = out["news_id"].map(lead).fillna("")
    out["title_start_entity_type"] = lead_type
    out["title_start_type_group"] = np.where(
        lead_type.eq(""), "none", lead_type.map(TYPE_GROUPS).fillna("other"))
    for section in SECTIONS:
        located = mentions.loc[mentions["section"].eq(section) & mentions["position"].notna()]
        earliest = located.groupby("news_id")["position"].min()
        position = out["news_id"].map(earliest)
        out[f"{section}_entity_absent"] = position.isna().astype("int8")
        out[f"{section}_entity_first_position"] = position.fillna(1.0).astype("float32")
    return out


def mention_weights(mentions, abstract_weight=0.5, decay=1.0, use_confidence=False):
    """w = section weight x exp(-decay x normalised position) [x confidence].

    decay=0 and abstract_weight=1 reproduce uniform averaging. Mentions with
    unknown offsets are treated as the least prominent position (1.0).
    """
    section = np.where(mentions["section"].eq("title"), 1.0, abstract_weight)
    position = mentions["position"].fillna(1.0).to_numpy()
    weights = section * np.exp(-decay * position)
    if use_confidence:
        weights = weights * mentions["confidence"].fillna(1.0).to_numpy()
    return pd.Series(weights, index=mentions.index, name="weight")


def load_entity_vectors(path):
    ids, rows = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) > 1:
                ids.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype="float32"))
    require(len({len(r) for r in rows}) == 1, f"Inconsistent vector lengths in {path}")
    require(len(set(ids)) == len(ids), f"Repeated entity IDs in {path}")
    return {entity: i for i, entity in enumerate(ids)}, np.vstack(rows)


def aggregate_entity_vectors(news_ids, mentions, weights, entity_index, matrix):
    """Weighted mean of TransE vectors per article.

    An entity named in both title and abstract counts once, with its largest
    weight. Entities without a pretrained vector are skipped. Returns the
    matrix aligned to news_ids and an int8 flag for articles with no vector.
    """
    frame = mentions[["news_id", "wikidata_id"]].assign(weight=weights.to_numpy())
    frame = frame.loc[frame["wikidata_id"].isin(entity_index) & frame["weight"].gt(0)]
    frame = frame.groupby(["news_id", "wikidata_id"], as_index=False)["weight"].max()
    row_of = pd.Series(np.arange(len(news_ids)), index=pd.Index(news_ids, name="news_id"))
    frame = frame.loc[frame["news_id"].isin(row_of.index)]
    rows = row_of.loc[frame["news_id"]].to_numpy()
    cols = frame["wikidata_id"].map(entity_index).to_numpy()
    w = frame["weight"].to_numpy(dtype="float32")
    total = np.zeros((len(news_ids), matrix.shape[1]), dtype="float32")
    np.add.at(total, rows, w[:, None] * matrix[cols])
    mass = np.bincount(rows, weights=w, minlength=len(news_ids)).astype("float32")
    vectors = np.divide(total, mass[:, None], out=np.zeros_like(total), where=mass[:, None] > 0)
    return vectors, (mass == 0).astype("int8")

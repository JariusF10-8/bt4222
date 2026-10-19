"""Load the group's CSV handoff with explicit types and JSON list decoding."""
import json
from pathlib import Path

import pandas as pd


def find_project_root(start=None):
    """Find the cloned repository from any notebook's working directory."""
    start = Path(start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "src/data_loading.py").is_file():
            return folder
    raise FileNotFoundError("Run from the cloned repository, or set PROJECT_ROOT explicitly.")


def load_partition(partition, processed_dir=None):
    """Return events, candidates, news for a partition listed in the manifest.

    Raw empty strings stay empty; explicit flags record missingness. Lists are
    decoded from JSON, timestamps are timezone-naive, and IDs remain strings.
    """
    base = Path(processed_dir) if processed_dir is not None else find_project_root() / "data/processed/v1"
    manifest = json.loads((base / "manifest.json").read_text())
    if partition not in manifest["partitions"]:
        raise ValueError(f"Partition {partition!r} unavailable; available: {manifest['partitions']}")
    folder = base / partition
    events = pd.read_csv(folder / "events.csv", keep_default_na=False, dtype={
        "event_key": "string", "impression_id": "string", "source_file": "string",
        "user_id": "string", "time": "string", "history": "string",
        "history_length": "int32", "history_missing": "bool",
    }, parse_dates=["timestamp"])
    events["history_ids"] = events["history_ids"].map(json.loads)
    candidates = pd.read_csv(folder / "candidates.csv", keep_default_na=False, dtype={
        "event_key": "string", "news_id": "string", "candidate_position": "int32", "label": "int8",
    })
    news = pd.read_csv(folder / "news.csv", dtype="string", keep_default_na=False)
    for column in news:
        if column.endswith("_missing"):
            if not news[column].isin(["True", "False"]).all():
                raise ValueError(f"Invalid boolean in {column}")
            news[column] = news[column].eq("True")
        elif column.endswith("_length") or column.startswith("num_"):
            news[column] = news[column].astype("int32")
        elif column in ["title_entities_parsed", "abstract_entities_parsed", "title_wikidata_ids", "abstract_wikidata_ids"]:
            news[column] = news[column].map(json.loads)
    return events, candidates, news

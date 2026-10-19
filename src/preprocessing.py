"""Canonical MIND preparation used by the preprocessing notebook."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd

BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLUMNS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
TRAIN_START = pd.Timestamp("2019-11-09")
VALIDATION_START = pd.Timestamp("2019-11-13")
TRAIN_FILE_END = pd.Timestamp("2019-11-15")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_sampling_tsv(path, columns):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required MIND file not found: {path}")
    return pd.read_csv(
        path, sep="\t", header=None, names=columns, dtype="string",
        keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )

def prepare_events(raw, source_file):
    events = raw.copy()
    require(len(events) > 0, f"No events in {source_file}")
    for column in ["impression_id", "user_id", "time", "impressions"]:
        require(
            events[column].fillna("").str.strip().ne("").all(),
            f"Missing {column} in {source_file}; review the raw source.",
        )
    require(
        events["impression_id"].is_unique,
        f"Repeated impression IDs in {source_file}; review before splitting.",
    )
    events["timestamp"] = pd.to_datetime(
        events["time"], format="%m/%d/%Y %I:%M:%S %p", errors="raise"
    )
    require(events["timestamp"].notna().all(), "Missing parsed timestamps")
    events["source_file"] = source_file
    events["event_key"] = events["source_file"] + ":" + events["impression_id"]
    events["history_ids"] = events["history"].fillna("").str.split()
    events["history_length"] = events["history_ids"].str.len().astype("int32")
    require(events["user_id"].str.fullmatch(r"U[0-9]+", na=False).all(), "Invalid user ID")
    require(events["history_ids"].apply(lambda ids: all(re.fullmatch(r"N[0-9]+", item) for item in ids)).all(), "Invalid history news ID")
    events["history_missing"] = events["history_ids"].str.len().eq(0)
    # Recorded times have no documented timezone: do not assume local time.
    return events.sort_values(
        ["timestamp", "impression_id"], kind="stable"
    ).reset_index(drop=True)

def expand_candidates(events):
    # Compact table: do not repeat titles or entire histories per candidate.
    tokens = events["impressions"].str.split().explode()
    valid = tokens.str.fullmatch(r"N[0-9]+-[01]", na=False)
    require(valid.all(), f"Malformed candidate tokens: {tokens[~valid].head().tolist()}")
    parts = tokens.str.rsplit("-", n=1, expand=True)
    candidates = pd.DataFrame({
        "event_key": events.loc[tokens.index, "event_key"].to_numpy(),
        "candidate_position": tokens.groupby(level=0, sort=False).cumcount().to_numpy(),
        "news_id": parts[0].to_numpy(),
        "label": parts[1].astype("int8").to_numpy(),
    })
    candidates["news_id"] = candidates["news_id"].astype("string")
    candidates["candidate_position"] = candidates["candidate_position"].astype("int32")
    return candidates

def referenced_news_ids(events, candidates):
    history_ids = {article for history in events["history_ids"] for article in history}
    return history_ids | set(candidates["news_id"])

def retain_metadata(events, candidates, metadata):
    require(
        metadata["news_id"].fillna("").str.strip().ne("").all(),
        "Missing article IDs in metadata; review the raw source.",
    )
    require(metadata["news_id"].is_unique, "Duplicate article IDs would multiply joins.")
    referenced_ids = referenced_news_ids(events, candidates)
    required = pd.DataFrame({"news_id": pd.Series(sorted(referenced_ids), dtype="string")})
    retained = required.merge(
        metadata, on="news_id", how="left", validate="one_to_one", indicator=True
    )
    retained["metadata_missing"] = retained.pop("_merge").eq("left_only")
    absent = retained["metadata_missing"]
    for column in ["title", "abstract", "url"]:
        retained.loc[absent, column] = ""
    for column in ["category", "subcategory"]:
        retained.loc[absent, column] = "__UNKNOWN__"
    for column in ["title_entities", "abstract_entities"]:
        retained.loc[absent, column] = "[]"

    # Only join the coverage flag here, avoiding repeated text for millions of rows.
    coverage = candidates[["news_id"]].merge(
        retained[["news_id", "metadata_missing"]], on="news_id", how="left",
        validate="many_to_one", sort=False,
    )
    require(len(coverage) == len(candidates), "Article join changed candidate count.")
    require(coverage["metadata_missing"].notna().all(), "Unaccounted candidate metadata.")
    audit = {
        "referenced_articles": len(retained),
        "articles_missing_metadata": int(absent.sum()),
        "candidate_exposures_missing_metadata": int(coverage["metadata_missing"].sum()),
    }
    return retained, audit

def verify_candidate_integrity(events, candidates):
    require(
        not candidates.duplicated(["event_key", "candidate_position"]).any(),
        "Repeated candidate positions within an event.",
    )
    require(
        set(candidates["event_key"]) == set(events["event_key"]),
        "Candidate table lost an event or introduced an unknown event.",
    )
    # Reconstruct complete token sequences: verifies IDs, labels, order and counts.
    ordered = candidates.sort_values("candidate_position", kind="stable")
    tokens = ordered["news_id"] + "-" + ordered["label"].astype("string")
    reconstructed = tokens.groupby(ordered["event_key"], sort=False).agg(" ".join)
    expected = events.set_index("event_key")["impressions"].str.split().str.join(" ")
    require(
        reconstructed.reindex(expected.index).eq(expected).all(),
        "Candidate IDs or labels changed during splitting/expansion/sampling.",
    )


def clean_news(news):
    """Preserve original text and annotations; add flags and parsed features."""
    news = news.copy()
    require(news['news_id'].str.fullmatch(r'N[0-9]+', na=False).all(), 'Invalid news ID')
    for column in NEWS_COLUMNS[1:]:
        missing = news[column].fillna('').str.strip().eq('')
        news[column + '_missing'] = missing
        news[column] = news[column].fillna('')
        if column in ['category', 'subcategory']:
            news.loc[missing, column] = '__UNKNOWN__'
    for field in ['title', 'abstract']:
        def parse(value):
            parsed = json.loads(value) if value.strip() else []
            require(isinstance(parsed, list) and all(isinstance(e, dict) for e in parsed),
                    f'Invalid {field} entity annotation: expected a list of objects')
            return parsed
        # Malformed nonempty JSON fails visibly rather than being silently discarded.
        news[field + '_entities_parsed'] = news[field + '_entities'].map(parse)
        news['num_' + field + '_entities'] = news[field + '_entities_parsed'].str.len().astype('int32')
        news[field + '_wikidata_ids'] = news[field + '_entities_parsed'].map(
            lambda entries: list(dict.fromkeys(e['WikidataId'] for e in entries
                                              if isinstance(e.get('WikidataId'), str) and e['WikidataId'])))
        news[field + '_length'] = news[field].str.split().str.len().astype('int32')
    return news


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame, path, list_columns=()):
    export = frame.copy()
    for column in list_columns:
        export[column] = export[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    export.to_csv(path, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')


def build_dataset(raw_dir, output_dir, *, require_test=True):
    """Validate and publish a new version; existing versions are never overwritten.

    Set require_test=False explicitly to produce train/validation without dev.
    A present but incomplete or malformed dev source always raises an error.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'{output_dir} already exists. Choose a new output version.')
    has_dev = (raw_dir / 'MINDsmall_dev').exists()
    if require_test and not has_dev:
        raise FileNotFoundError('MINDsmall_dev is missing. Add it or explicitly set REQUIRE_TEST=False for a train/validation-only build.')
    if not has_dev:
        warnings.warn('MINDsmall_dev missing: exporting train and validation only; no final test set.', stacklevel=2)
    sources = ['MINDsmall_train'] + (['MINDsmall_dev'] if has_dev else [])
    events_by_source, metadata_by_source, source_files = {}, {}, {}
    for source in sources:
        folder = raw_dir / source
        events_by_source[source] = prepare_events(
            read_sampling_tsv(folder / 'behaviors.tsv', BEHAVIORS_COLUMNS), source + '/behaviors.tsv')
        metadata_by_source[source] = read_sampling_tsv(folder / 'news.tsv', NEWS_COLUMNS)
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in ['.tsv', '.vec']:
                source_files[source + '/' + path.name] = {
                    'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    original = events_by_source['MINDsmall_train']
    require((original['timestamp'].ge(TRAIN_START) & original['timestamp'].lt(TRAIN_FILE_END)).all(),
            'Training source contains dates outside November 9–14, 2019.')
    events_by_partition = {
        'train': original.loc[original['timestamp'].lt(VALIDATION_START)].reset_index(drop=True),
        'validation': original.loc[original['timestamp'].ge(VALIDATION_START)].reset_index(drop=True),
    }
    if has_dev:
        events_by_partition['test'] = events_by_source['MINDsmall_dev']
    previous = None
    for name, events in events_by_partition.items():
        require(len(events) > 0, f'Empty {name} partition')
        if previous is not None:
            require(previous < events['timestamp'].min(), f'Times overlap before {name}')
        previous = events['timestamp'].max()
    keys = pd.concat([events['event_key'] for events in events_by_partition.values()], ignore_index=True)
    require(keys.is_unique, 'Repeated event keys across partitions')
    require(len(keys) == sum(len(events) for events in events_by_source.values()), 'Lost events in split')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.preprocessing-', dir=output_dir.parent))
    summary_rows = []
    try:
        for name, events in events_by_partition.items():
            print(f'Preparing {name}: {len(events):,} events', flush=True)
            candidates = expand_candidates(events)
            verify_candidate_integrity(events, candidates)
            source = 'MINDsmall_dev' if name == 'test' else 'MINDsmall_train'
            news, audit = retain_metadata(events, candidates, metadata_by_source[source])
            news = clean_news(news)
            folder = staging / name
            folder.mkdir()
            # Outcome strings stay out of events; labels live only in candidates.
            write_csv(events.drop(columns=['impressions']), folder / 'events.csv', ['history_ids'])
            write_csv(candidates, folder / 'candidates.csv')
            write_csv(news, folder / 'news.csv', [
                'title_entities_parsed', 'abstract_entities_parsed',
                'title_wikidata_ids', 'abstract_wikidata_ids'])
            clicks = int(candidates['label'].sum())
            summary_rows.append({
                'partition': name, 'earliest_time': events['timestamp'].min(),
                'latest_time': events['timestamp'].max(), 'events': len(events),
                'users': events['user_id'].nunique(), 'candidate_exposures': len(candidates),
                'clicked_candidates': clicks, 'nonclicked_candidates': len(candidates) - clicks,
                'events_with_empty_history': int(events['history_missing'].sum()), **audit,
            })
        summary = pd.DataFrame(summary_rows)
        write_csv(summary, staging / 'split_summary.csv')
        dictionary = Path(__file__).resolve().parents[1] / 'data/data_dictionary.md'
        shutil.copyfile(dictionary, staging / 'data_dictionary.md')
        manifest = {
            'schema_version': 1, 'dataset_version': output_dir.name,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'partitions': list(events_by_partition), 'complete': has_dev,
            'missing_sources': [] if has_dev else ['MINDsmall_dev'],
            'split_policy': {'train': '2019-11-09 <= time < 2019-11-13',
                             'validation': '2019-11-13 <= time < 2019-11-15',
                             'test': 'all MINDsmall_dev events, strictly later than validation'},
            'timestamp_timezone': 'unspecified in source; preserved as timezone-naive',
            'sampling': 'none; all candidates retained in original order',
            'text_length_definition': 'whitespace-separated words; no lowercasing or truncation',
            'csv_encoding': 'utf-8', 'list_encoding': 'JSON',
            'libraries': {'pandas': pd.__version__, 'numpy': np.__version__},
            'preprocessing_sha256': sha256_file(__file__), 'source_files': source_files,
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging)
        raise
    return summary

# Processed MIND CSV schema (version 1)

Use `src.data_loading.load_partition` to restore types. Every CSV has a header,
UTF-8 encoding and standard CSV quoting (unlike the raw TSV reader). Lists are
JSON, not Python repr strings. Empty raw strings stay empty, including literal
text such as `NA`; missingness is recorded with boolean flags. No records are
sampled, and empty histories are retained. IDs are strings, never model inputs
by default. Timestamps have no assumed timezone.

## events.csv — one row per impression

| Column | Type | Meaning |
|---|---|---|
| event_key | string, primary key | `source_file:impression_id`; unique across sources |
| impression_id | string | Original source-local impression ID |
| user_id | string | Original anonymised user ID |
| source_file | string | Origin, e.g. `MINDsmall_train/behaviors.tsv` |
| time | string | Original timestamp text |
| timestamp | datetime | Parsed source time, timezone unspecified |
| history | string | Original space-separated history; empty if unavailable |
| history_ids | JSON list of strings | Supplied history in its original order |
| history_length | int32 | Number of supplied history entries |
| history_missing | bool | No supplied history; does not prove a new user |

## candidates.csv — one row per candidate exposure

| Column | Type | Meaning |
|---|---|---|
| event_key | string, foreign key | Links to events |
| candidate_position | int32 | Original zero-based order within the impression |
| news_id | string, foreign key | Links to news |
| label | int8 | Target: 1 clicked, 0 unclicked |

The primary key is `(event_key, candidate_position)`. The position preserves
source ordering; it is not assumed to be a documented display-rank feature.
Keep whole impressions together for ranking and evaluation.

## news.csv — one row per referenced article, within each partition

| Column(s) | Type | Meaning |
|---|---|---|
| news_id | string, primary key | Article referenced by a candidate or history |
| category, subcategory | string | Source values; `__UNKNOWN__` when blank or metadata absent |
| title, abstract, url | string | Original source text; blanks become empty strings |
| title_entities, abstract_entities | string | Original entity JSON; absent metadata uses `[]` |
| metadata_missing | bool | Referenced article absent from source metadata |
| `<raw_field>_missing` | bool | Empty/blank field after metadata retention, for every raw field except news_id |
| title_entities_parsed, abstract_entities_parsed | JSON list of objects | Parsed annotations; empty fields become []; malformed JSON stops the build |
| title_wikidata_ids, abstract_wikidata_ids | JSON list of strings | Distinct nonempty entity IDs in first-occurrence order |
| num_title_entities, num_abstract_entities | int32 | Number of annotation entries |
| title_length, abstract_length | int32 | Whitespace-separated word count |

Check `metadata_missing` first: individual field flags describe the retained
values, including placeholders, rather than reconstructing absent source rows.
Original text is not lowercased, stemmed, truncated or tokenised for a model.
Train/validation metadata comes only from MINDsmall_train; test metadata comes
only from MINDsmall_dev. Referenced history articles are retained too.

## Other files and model development

`split_summary.csv` contains partition date ranges, event/user/article counts,
candidate exposure/click counts, and missing-history/metadata counts.
`manifest.json` records build time, source checksums, library versions, partition
policy and whether the test source was available. A build without dev is
explicitly incomplete and has no test directory.

Pretrained entity/relation vectors stay in `data/raw/<source>/*.vec`; their
checksums are recorded when available. Match entity IDs to Wikidata IDs, not
news IDs. Relation vectors alone are not a graph edge list.

Never feed label or current-event click totals into a model. Fit vocabularies,
encoders and scalers using training data only. Popularity/CTR features must use
only permitted past interactions, excluding the event being predicted. Preserve
supplied histories; do not reconstruct them using future clicks. Fit article
vocabularies on train/news.csv excluding metadata_missing rows. Use left joins
with `validate="many_to_one"` to preserve candidate rows. Hold test labels aside
until final evaluation. Do not treat candidate position as a feature without
justifying that decision.

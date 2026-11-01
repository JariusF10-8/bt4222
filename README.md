# BT4222 Group 2 — MIND news recommendation

The project follows one shared workflow:

1. **Preprocessing:** produce the canonical CSV dataset and temporal partitions.
2. **Feature engineering:** generate the unified article, impression and candidate features.
3. **Modelling:** train baseline and advanced models using the agreed data versions.

Teammates reuse the same preprocessing and feature bundles, selecting the inputs
appropriate for each model.

## Main notebooks

| Stage | Notebook | Purpose |
|---|---|---|
| Preprocessing | [01_Data_Preprocessing.ipynb](notebooks/01_Data_Preprocessing.ipynb) | Parse, clean, validate and export the shared dataset |
| Feature engineering | [Unified_Feature_Engineering.ipynb](notebooks/features/Unified_Feature_Engineering.ipynb) | Consolidate the group's engineered features and fitted transforms |
| NeuMF | [NeuMF.ipynb](notebooks/models/NeuMF.ipynb) | Train and tune the ID-only baseline, with optional feature-fusion experiments |

Existing completed data bundles can be reused without rerunning preprocessing
or feature engineering.

## Repository layout

Generated data and outputs are shown below for reference; they are ignored by Git.

```text
data/
  raw/
    MINDsmall_train/             Original train TSVs and embedding vectors
    MINDsmall_dev/               Official dev files for the complete handoff
  data_dictionary.md             Shared preprocessing schema
  processed/<version>/          Generated preprocessing bundle
    train/{events,candidates,news}.csv
    validation/{events,candidates,news}.csv
    test/{events,candidates,news}.csv
    manifest.json
    data_dictionary.md
    split_summary.csv

src/
  preprocessing.py              Canonical parsing, cleaning, checks and exports
  data_loading.py                Shared CSV loader and repository discovery
  sampling.py                    Training negative sampling
  ...                            Supporting feature helpers

notebooks/
  01_Data_Preprocessing.ipynb
  eda/
    Dataset_Overview.ipynb
    Dataset_EDA.ipynb
    Categorical_Text_Features.ipynb
  features/
    Unified_Feature_Engineering.ipynb
    ...                          Original member feature notebooks
  models/
    NeuMF.ipynb
    ...                          Other model notebooks

tests/                           Small pipeline and feature-helper tests
figures/                         Generated report plots
outputs/
  features/<feature_version>/    Generated feature bundles
  models/<model>/<run>/          Model checkpoints, metrics and predictions

requirements.txt
```

The original root-level `MINDsmall_train` folder now lives under `data/raw`.
Raw files are not modified. Existing versioned training files can remain in
the repository; additional large data and generated bundles belong in the
shared Drive.

A present `MINDsmall_dev` directory must contain valid `behaviors.tsv` and
`news.tsv` files. Do not create an empty dev folder as a placeholder.

## GitHub and shared Drive

GitHub stores:

- Notebooks, source code, tests and requirements.
- README instructions and feature definitions.
- Small selected results tables and report figures.

The shared Drive stores:

- Complete preprocessed datasets.
- Complete engineered-feature bundles and fitted transforms.
- Model checkpoints, full predictions and experiment outputs.

The repository ignores `data/processed/` and generated `outputs/`.
Do not force-add these large folders to Git.

Small copies of manifests, feature dictionaries and summaries may be committed
under a documentation folder. Keep the originals with their complete Drive
bundles.

### Shared storage layout

The Drive layout uses the same relative `data/` and `outputs/` paths as the
repository:

```text
BT4222 Group Project/
  data/
    processed/v2/
  outputs/
    features/unified_v1/
    models/
      neumf/<timestamped run>/
      <other model>/<run>/
```

The current shared version settings are:

- Preprocessing: `v2`.
- Engineered features: `unified_v1`.

Update these instructions when the group adopts another version. Preprocessing
and feature versions are independent: a new feature build can reuse the same
preprocessing.

Keep each complete version folder together, including its manifests,
dictionaries, fitted transforms, indexes and matrices.

## Local setup and preprocessing

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/01_Data_Preprocessing.ipynb` and run its cells in order.

The standalone preprocessing notebook defaults to:

```python
OUTPUT_DIR = PROJECT_ROOT / "data/processed/v1"
REQUIRE_TEST = False
```

These are notebook defaults, separate from the group's shared `v2` handoff.

When dev is absent, a partial build records `complete: false` and does not
invent a test partition. For the complete group handoff, provide
`MINDsmall_dev` and set `REQUIRE_TEST=True`.

Existing output versions are never overwritten. Choose a new output directory
for a fresh preprocessing build.

### Temporal partitions

- **Train:** November 9–12, 2019 from `MINDsmall_train`.
- **Validation:** November 13–14, 2019 from `MINDsmall_train`.
- **Final test:** all `MINDsmall_dev` events, verified to occur strictly later.

Preprocessing retains complete candidate lists, click labels and source
ordering. No negative sampling is applied during preprocessing.

Histories remain in their supplied order. Missing histories and abstracts are
retained with flags. Text is preserved, and basic whitespace word counts are
added. Missing article metadata receives explicit placeholders.

Invalid IDs, timestamps, candidate tokens, duplicate event/article IDs and
malformed entity JSON stop the build. Candidate reconstruction and metadata
joins are checked before export.

The manifest records checksums, library versions, split dates and completeness.
The output bundle is published only after the available partitions pass checks.

## Unified feature engineering in Colab

Open
`notebooks/features/Unified_Feature_Engineering.ipynb`
in Colab and edit the setup cell before running it.

```python
STORAGE_ROOT = Path(
    '/content/drive/MyDrive/BT4222 Group Project'
)

PROCESSED_VERSION = 'v2'
OUTPUT_VERSION = 'unified_v1'
```

The notebook constructs:

```python
PROCESSED_DIR = STORAGE_ROOT / 'data/processed' / PROCESSED_VERSION
OUTPUT_DIR = STORAGE_ROOT / 'outputs/features' / OUTPUT_VERSION
```

For a fresh feature build, choose an unused `OUTPUT_VERSION`. If the agreed
feature bundle already exists, models can read it directly.

Existing preprocessing is reused. If its `manifest.json` is missing and
`BUILD_PROCESSED_IF_MISSING=True`, the notebook builds preprocessing from the
raw MIND data before constructing features.

Use these settings for the shared group build:

```python
REQUIRE_TEST = True
HISTORY_MODES = ('train_only',)
ALLOW_HELDOUT_FEEDBACK = False
```

Leave optional feature diagnostics and prominence-grid experiments disabled
unless needed.

Run setup, allow Drive access and upload a current repository ZIP when
prompted. Opening a notebook in Colab alone does not make the repository's
`src/` or raw files available.

Raw data defaults to the uploaded repository's `data/raw` folder. If it is
stored elsewhere, configure this before setup:

```python
import os

os.environ['MIND_RAW_DIR'] = (
    '/content/drive/MyDrive/BT4222 Group Project/Microsoft MIND'
)
```

`MIND_RAW_DIR` must contain the `MINDsmall_train` and `MINDsmall_dev` directories
for a complete build, including the entity embedding files used by feature
engineering.

Check the printed paths, then run the remaining cells in order.

### Feature bundle

Each partition contains:

| File | Purpose |
|---|---|
| `news_features.csv` | Article counts, metadata encodings and entity-related scalar features |
| `event_features.csv` | Impression context, history summaries and missingness flags |
| `candidate_features.csv.gz` | Candidate–history matching and time interaction features |
| `candidate_history_train_only.csv.gz` | Historical popularity and CTR features |
| `news_index.csv` | Article row order for the exported matrices |
| `article_entity_vectors.npz` | Article entity-vector variants |
| `article_entity_ids.csv.gz` | Type-specific entity IDs |
| `tfidf_title_abstract.npz` | Sparse title/abstract TF-IDF representations |

The root contains the manifest, feature dictionary, vocabularies, fitted
vectorizer, summaries and merge documentation.

Click labels remain in the preprocessing candidates table. Feature exports
alone are insufficient to train a supervised model.

## Modelling in Colab

All teammates must use the agreed preprocessing version and, where applicable,
the same feature bundle and historical-feedback policy.

### NeuMF baseline

Open `notebooks/models/NeuMF.ipynb` in Colab. A GPU is useful for tuning.

Before running setup, configure:

- `STORAGE_ROOT`: the shared project folder.
- `PROCESSED_VERSION`: the existing complete preprocessing version.
- `FEATURE_VERSION`: the completed feature version for optional fusion.

The notebook constructs:

```text
data/processed/<PROCESSED_VERSION>/
outputs/features/<FEATURE_VERSION>/
outputs/models/neumf/<RUN_NAME>/
```

`RUN_NAME` is timestamped to keep studies separate.

The complete preprocessing handoff must already exist. The standalone NeuMF
notebook does not rebuild preprocessing and does not require a repository ZIP
or raw embedding files.

For the report's ID-only baseline, keep:

```python
RUN_FEATURE_FUSION = False
RUN_ID_MASK_ABLATION = False
RUN_EXTRA_SEEDS = False
EVALUATE_TEST = False
```

The baseline uses `user_id` and `news_id` as categorical inputs and `label` as
the training target. It learns its own ID embeddings. Engineered features enter
only in the optional feature-fusion experiments.

Run training and tuning using validation. After fixing the experiment choices,
set `EVALUATE_TEST=True` immediately before the final evaluation cell.
Do not rerun setup merely to enable final test evaluation.

### Advanced models

Choose the feature subset appropriate for each architecture. Models do not
need to consume every exported feature.

For LSTM or attention-based history encoders, use the ordered `history_ids`
from preprocessing and map them to individual article representations.
Pooled history similarities do not replace the original sequence.

Article matrices must be aligned using `news_index.csv` or the accompanying
article IDs.

## Loading data for features and models

Download the agreed preprocessing version to `data/processed/v2`, or use its
shared Drive location.

From a notebook inside the repository:

```python
from pathlib import Path
import sys

PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / 'src/data_loading.py').is_file()
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_partition

# Local copy:
PROCESSED_DIR = PROJECT_ROOT / 'data/processed/v2'

# Alternatively, after mounting Drive:
# PROCESSED_DIR = Path(
#     '/content/drive/MyDrive/BT4222 Group Project/data/processed/v2'
# )

events, candidates, news = load_partition(
    'train', PROCESSED_DIR
)
```

The loader preserves string IDs and literal text, restores timestamps and
booleans, and decodes JSON histories/entity lists. Empty strings remain empty;
use missingness flags.

See [the data dictionary](data/data_dictionary.md).

CSV quoting is handled normally on export/import. Raw TSV loading uses
`csv.QUOTE_NONE` to preserve literal quotes in MIND text.

For candidate-level features, join compact tables as needed:

```python
examples = candidates.merge(
    events[['event_key', 'user_id', 'timestamp', 'history_length']],
    on='event_key',
    how='left',
    validate='many_to_one',
).merge(
    news[['news_id', 'category', 'title_length', 'abstract_missing']],
    on='news_id',
    how='left',
    validate='many_to_one',
)

y = examples.pop('label')
```

Section 12 of the unified feature notebook provides helpers for attaching
exported features while preserving repeated sampled negatives.

## Training and evaluation rules

- Keep identifiers available for joins and grouped evaluation. Select model
  inputs explicitly rather than passing every remaining column to a model.
  ID-based models use separately encoded user/article IDs.
- Keep `label` as the target. Never use current-impression `num_clicked` or
  `num_unclicked` as predictors.
- Fit vocabularies, encoders, scalers and other learned transforms on training
  only. TF-IDF is fitted on training news excluding missing metadata.
- Historical popularity and CTR features must exclude the current event and
  future outcomes. Use the agreed `train_only` policy for shared comparisons.
- Apply negative sampling only during model training. Feature construction uses
  complete logs, and validation/test ranking uses complete original slates.
- Preserve repeated replacement-sampled negatives when joining features.
- Candidate position identifies source ordering; it is not assumed to be a
  display-rank predictor.
- Group ranking metrics by the complete impression, even when file-reading
  chunks split an impression.
- Use validation for tuning and early stopping. Keep test labels aside until
  the model and experiment choices are fixed.
- For the initial evaluation protocol, do not refit on validation before testing.

Pretrained vectors remain in their raw folders and are loaded when needed.
Entity embeddings join to Wikidata entity IDs, not news IDs. The unified
pipeline uses entity vectors; relation embeddings alone do not supply a
knowledge-graph edge table.

## Checks

Run the small repository tests without downloading the full dataset:

```bash
python -m unittest discover -s tests -v
```

The tests cover preprocessing, sampling and supporting feature helpers,
including temporal boundaries, preserved candidate identities and labels,
CSV round trips, missing data, version handling and historical-feature logic.

Also run the checks included in the unified feature and model notebooks.
Repository helper tests do not replace validation of inline notebook code or
the actual exported bundles.

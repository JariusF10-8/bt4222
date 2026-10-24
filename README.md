# BT4222 Group 2 — MIND news recommendation

One canonical preprocessing notebook produces the shared CSV dataset. Each
teammate starts from the same version and temporal partitions, then develops
features and models independently.

## Repository layout

```text
data/
  raw/MINDsmall_train/           Original train TSVs and embedding vectors
  raw/MINDsmall_dev/             Add the official dev files when available
  data_dictionary.md            Shared output schema
  processed/v1/                 Generated handoff (ignored by Git)
    train/{events,candidates,news}.csv
    validation/{events,candidates,news}.csv
    test/{events,candidates,news}.csv  Only when dev is available
    manifest.json
    data_dictionary.md
    split_summary.csv
src/
  preprocessing.py              Canonical parsing, cleaning, checks and exports
  data_loading.py               Shared CSV loader and repository discovery
notebooks/
  01_Data_Preprocessing.ipynb    Main entry point
  eda/
    Dataset_Overview.ipynb       Raw data and embedding overview
    Dataset_EDA.ipynb            Quality, funnel, sparsity and time analysis
    Categorical_Text_Features.ipynb
  features/                     Feature engineering experiments
  models/                       Modelling experiments
figures/                        Report plots (generated files ignored)
outputs/                        Features, model files and metrics (ignored)
requirements.txt
```

The original root-level MINDsmall_train folder now lives under data/raw.
Raw files are not changed. The training files already included in the repository
can remain versioned; store additional large data and generated handoffs in the
shared Drive. No empty dev folder is supplied: its presence means the pipeline
expects valid behaviors.tsv and news.tsv files.

## Local setup and preprocessing

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/01_Data_Preprocessing.ipynb` and run all cells. The notebook
explicitly sets `REQUIRE_TEST=False` to support the currently available training
files. It emits a warning and records `complete: false` in the manifest when
dev is absent; it does not invent a test partition. Once dev is available, set
`REQUIRE_TEST=True` for the full handoff.

Outputs go to `data/processed/v1`. Existing versions are never overwritten:
choose v2, v3, etc. for subsequent builds. The manifest contains source SHA-256
checksums, the preparation code checksum, library versions, split dates and
completeness. Publication occurs after all available partitions pass checks.

- Train: November 9–12, 2019 from MINDsmall_train.
- Validation: November 13–14, 2019 from MINDsmall_train.
- Final test: all MINDsmall_dev events, required to be strictly later.

No negative sampling is applied. Every event retains all candidate labels and
source ordering. Histories stay in their supplied order. Missing histories and
abstracts are retained with flags. Text is preserved; only basic whitespace word
counts are added. Missing article metadata gets explicit placeholders. Invalid
IDs, timestamps, candidate tokens, duplicate event/article IDs and malformed
entity JSON stop the build. Candidate reconstruction and metadata joins are
checked before export.

## Colab and shared Drive

Clone/open this repository in Colab so `src/` is available. Install its
requirements if the runtime is missing packages. Before the setup cell, configure
paths if your working directory is outside the clone or your raw data is in Drive:

```python
import os
os.environ['BT4222_PROJECT_ROOT'] = '/content/bt4222'
os.environ['MIND_RAW_DIR'] = (
    '/content/drive/MyDrive/BT4222 Group Project/Microsoft MIND'
)
```

The notebooks mount Drive when running in Colab. MIND_RAW_DIR must contain the
MINDsmall_train and optional MINDsmall_dev directories. To export directly to
Drive, change OUTPUT_DIR in the preprocessing notebook to a new version folder
there. Otherwise copy the entire generated version folder to the shared Drive.
Do not share only the CSVs: include the manifest, dictionary and summary.

## Loading data for features and models

Download the agreed version to `data/processed/v1`, or pass its shared Drive
path to the loader. From any notebook inside the repository:

```python
from pathlib import Path
import sys

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / 'src/data_loading.py').is_file())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_partition

events, candidates, news = load_partition(
    'train', PROJECT_ROOT / 'data/processed/v1'
)
```

The loader preserves string IDs and literal text, restores timestamps and
booleans, and decodes JSON histories/entity lists. Empty strings remain empty;
use missingness flags. See [the data dictionary](data/data_dictionary.md).
CSV quoting is handled normally on export/import; raw TSV loading uses
`csv.QUOTE_NONE` to preserve literal quotes in MIND text.

For candidate-level features, join compact tables as needed:

```python
examples = candidates.merge(
    events[['event_key', 'user_id', 'timestamp', 'history_length']],
    on='event_key', how='left', validate='many_to_one',
).merge(
    news[['news_id', 'category', 'title_length', 'abstract_missing']],
    on='news_id', how='left', validate='many_to_one',
)
y = examples.pop('label')  # Keep the target separate from feature inputs.
```

Keep event/user/article identifiers available for joins and grouped evaluation;
do not blindly pass every remaining column to a model. Fit vocabularies,
encoders, scalers and other learned transforms on training only. For TF-IDF,
use train/news.csv excluding metadata_missing articles. Historical popularity
and CTR features must exclude the event being predicted and future outcomes.
Do not use current-event num_clicked/num_unclicked from EDA as features. Candidate
position preserves source order; it is not assumed to be a display-rank feature.
Keep all candidates from an event together for ranking evaluation. Hold test
labels aside until the model is fixed. For the initial experiment, do not refit
on validation before testing.

Pretrained entity/relation vectors remain in their raw folders and are loaded
only when needed. Entity vectors join to Wikidata entity IDs, not news IDs.
Teammates should agree on the same output version for comparable experiments.

## Checks

Run the small pipeline tests without downloading data:

```bash
python -m unittest discover -s tests -v
```

They cover temporal boundaries, source-safe event keys, preserved candidate
order/labels, CSV round trips (including literal quotes and JSON entities),
missing metadata, missing dev data, immutable versions, and failed builds.

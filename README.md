# BT4222 Group 2: MIND news recommendation

We rank every candidate article in a MIND-small impression by its predicted click
chance. Training uses 9 to 12 November 2019, validation uses 13 to 14 November,
and the test set is the `MINDsmall_dev` log of 15 November, scored once.

## Pipeline

Run the notebooks in this order. Each one starts with a markdown section that
explains its purpose, inputs and outputs, and every notebook keeps its saved
output cells. In each model notebook, a note marked **Final test result** next
to the test cell gives the numbers used in the report.

| Step | Notebook | What it does | Main output |
|---|---|---|---|
| 01 | [01_Dataset_EDA.ipynb](notebooks/eda/01_Dataset_EDA.ipynb) | Data quality, sparsity and time patterns in the raw files | EDA tables and plots |
| 02 | [02_Data_Preprocessing.ipynb](notebooks/02_Data_Preprocessing.ipynb) | Parse, clean, validate and split the raw MIND files | `data/processed/v2` |
| 03 | [03_Unified_Feature_Engineering.ipynb](notebooks/features/03_Unified_Feature_Engineering.ipynb) | Build the 185 shared features with leakage checks | `outputs/features/unified_v1` |
| 04a | [04a_NeuMF.ipynb](notebooks/models/04a_NeuMF.ipynb) | Baseline: ID-only NeuMF | test metrics |
| 04b | [04b_CBF.ipynb](notebooks/models/04b_CBF.ipynb) | Baseline: content-based filtering with TF-IDF | test metrics |
| 04c | [04c_UBCF.ipynb](notebooks/models/04c_UBCF.ipynb) | Baseline: user-based collaborative filtering | test metrics |
| 05a | [05a_feature-based_MLP.ipynb](notebooks/models/05a_feature-based_MLP.ipynb) | Advanced model 1: mean-pooled history and engineered features | test metrics |
| 05b | [05b_Candidate_Attention_MLP.ipynb](notebooks/models/05b_Candidate_Attention_MLP.ipynb) | Advanced model 2: candidate-aware attention over the history | test metrics |

Supporting notebooks:

- `notebooks/eda/Dataset_Overview.ipynb` and `notebooks/eda/Categorical_Text_Features.ipynb`: more EDA on the raw files, embeddings, categories and text.
- `notebooks/eda/final_report_section_1_2_data_distributions.ipynb`: the figures in Section 1.2 of the report.
- `notebooks/features/`: each member's original feature notebook and the CTR history check (see `notebooks/features/README.md`). Step 03 combines them into one bundle, so the two advanced models only read the Step 03 output.

## Results on the test set

All models are fitted on the training days only, tuned on validation, and scored
once on the complete test slates (73,152 impressions).

| Model | Validation nDCG@10 | Test AUC | Test MRR | Test nDCG@5 | Test nDCG@10 |
|---|---|---|---|---|---|
| Random order | 0.2703 | 0.5000 | 0.2182 | 0.2229 | 0.2865 |
| UBCF | 0.2764 | 0.4956 | 0.2165 | 0.2201 | 0.2828 |
| CBF | 0.3591 | 0.5763 | 0.2762 | 0.2949 | 0.3569 |
| NeuMF | 0.2993 | 0.5095 | 0.2223 | 0.2275 | 0.2912 |
| Feature-based MLP | 0.3916 | 0.6367 | 0.2981 | 0.3300 | 0.3897 |
| Candidate-aware attention MLP | 0.3942 | 0.6377 | 0.2997 | 0.3290 | 0.3925 |

## Datasets and preprocessing artifacts

The raw MIND-small files are in this repository under `data/raw`. The cleaned
dataset and the feature bundle are too large for GitHub (about 720 MB and
860 MB, with single files over GitHub's 100 MB limit), so they are kept in our
shared Google Drive folder:

**Shared Drive folder** (add a shortcut to My Drive to run the notebooks): https://drive.google.com/drive/folders/1ZMBqrHA3wOM8h1DuBW04gypymjJiPmk1

- Cleaned dataset, Step 02 output (`data/processed/v2`): https://drive.google.com/drive/folders/1FbcOhE7elk3nTzBw9QTGrf-SFXtU3ta9
- Feature bundle, Step 03 output (`outputs/features/unified_v1`): https://drive.google.com/drive/folders/1VdCiWarnWEucRM9C6DAcSGVy5lZm6qsK

```text
BT4222 Group Project/
  data/processed/v2/                 Step 02 output, used by every model
    train/ validation/ test/         events.csv, candidates.csv, news.csv
    manifest.json                    checksums, split dates, row counts
  outputs/features/unified_v1/       Step 03 output, used by both advanced models
    train/ validation/ test/         article, impression and candidate features, TF-IDF and entity matrices
    manifest.json                    feature dictionary, fitted vocabularies and checks
  outputs/models/                    run folders with metrics, checkpoints and predictions
```

`v2` is the shared preprocessing build and `unified_v1` is the shared feature
build. Running Step 02 or Step 03 yourself writes a new version folder (Step 02
defaults to `v1` locally) and never overwrites an existing one.

## How to run in Google Colab

1. Open the shared Drive link above and choose **Add shortcut to Drive**, then
   place it in **My Drive**. Keep the folder name `BT4222 Group Project`,
   because every notebook looks for `/content/drive/MyDrive/BT4222 Group Project`.
2. Open a notebook from GitHub in Colab (each notebook has an **Open in Colab**
   badge in its first cell).
3. Run all cells and allow Drive access when asked. The setup cell prints the
   input and output folders it uses.

Steps 04 and 05 only need the Drive folder. Steps 01, 02 and 03 also need this
repository's code and raw files. In Colab, run these two lines first:

```python
!git clone https://github.com/JariusF10-8/bt4222.git /content/bt4222
%cd /content/bt4222
```

Step 03 can instead ask you to upload the repository ZIP (**Code > Download ZIP**
on GitHub).

A GPU runtime is optional. On a 2-core CPU, CBF takes a few minutes, UBCF about
10 minutes, and each advanced model about 2 hours including tuning and ablations.

To run locally instead, copy `data/processed/v2` and `outputs/features/unified_v1`
from Drive into the same paths in this repository, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Rules every model follows

- Vocabularies, encoders, scalers and other fitted transforms use training data only.
- Click-history features use only clicks from before the impression (`train_only` mode).
- `num_clicked` and `num_unclicked` are removed because they leak the label.
- Negative sampling (1 click and 4 non-clicks) is used only to train NeuMF and the two MLPs.
  Validation and test always score the complete original slates.
- Validation is used for tuning and early stopping. Test is read once, in the last cell.
- No model is refitted on validation before testing.

## Repository layout

```text
data/raw/                 original MIND-small train and dev files
data/data_dictionary.md   column definitions for the processed CSVs
notebooks/                the pipeline notebooks above
src/                      shared preprocessing, loading, sampling and feature code
tests/                    unit tests for the shared code
figures/                  report figures (generated)
outputs/                  generated features and model runs (kept in Drive)
requirements.txt
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover preprocessing, time boundaries, sampling, category encoding,
entity prominence and the click-history features.

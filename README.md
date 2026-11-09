# BT4222 Group 2: MIND news recommendation

We rank every candidate article in a MIND-small impression by its predicted click
chance. Training uses 9 to 12 November 2019, validation uses 13 to 14 November,
and the test set is the `MINDsmall_dev` log of 15 November, scored once.

## Pipeline

The numbering shows the pipeline stages. Each notebook explains its purpose,
inputs and outputs and preserves saved experimental output cells. Model
notebooks highlight report metrics beside the final test cell.

The completed shared data bundles can be reused: running the model notebooks
does not require rebuilding Steps 01–03. The current Step 02 and Step 03 defaults
write separate Colab reproduction-check bundles, leaving the original model
inputs unchanged.

| Step | Notebook | What it does | Main output |
|---|---|---|---|
| 01 | [01_Dataset_EDA.ipynb](notebooks/eda/01_Dataset_EDA.ipynb) | Data quality, sparsity and time patterns in the raw files | EDA tables and plots |
| 02 | [02_Data_Preprocessing.ipynb](notebooks/02_Data_Preprocessing.ipynb) | Parse, clean, validate and split the raw MIND files | Reproduction check: `data/processed/v2_colab_check` |
| 03 | [03_Unified_Feature_Engineering.ipynb](notebooks/features/03_Unified_Feature_Engineering.ipynb) | Build the shared feature bundle, documented by 185 dictionary entries, with leakage checks | Reproduction check: `outputs/features/unified_v1_colab_check` |
| 04a | [04a_NeuMF.ipynb](notebooks/models/04a_NeuMF.ipynb) | Baseline: ID-only NeuMF | test metrics |
| 04b | [04b_CBF.ipynb](notebooks/models/04b_CBF.ipynb) | Baseline: content-based filtering with TF-IDF | test metrics |
| 04c | [04c_UBCF.ipynb](notebooks/models/04c_UBCF.ipynb) | Baseline: user-based collaborative filtering | test metrics |
| 05a | [05a_feature-based_MLP.ipynb](notebooks/models/05a_feature-based_MLP.ipynb) | Advanced model 1: mean-pooled history and engineered features | test metrics |
| 05b | [05b_Candidate_Attention_MLP.ipynb](notebooks/models/05b_Candidate_Attention_MLP.ipynb) | Advanced model 2: candidate-aware attention over the history | test metrics |

Supporting notebooks:

- `notebooks/eda/Dataset_Overview.ipynb` and `notebooks/eda/Categorical_Text_Features.ipynb`: more EDA on the raw files, embeddings, categories and text.
- `notebooks/eda/final_report_section_1_2_data_distributions.ipynb`: the figures in Section 1.2 of the report.
- `notebooks/features/`: each member's original feature notebook and the CTR history check (see `notebooks/features/README.md`). These are supporting analyses, not prerequisites for the main pipeline. The advanced models read the shared preprocessing and unified feature bundles. The separate CTR history diagnostic requires additional exports from `feature_engineering_chris.ipynb`.

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
    manifest.json                    provenance and build settings
    feature_dictionary.csv           feature definitions; fitted transforms and audits alongside
  outputs/models/                    run folders with metrics, checkpoints and predictions
```

`v2` and `unified_v1` are the agreed input bundles for the model notebooks.
The current Colab reproduction defaults are separate:

| Notebook | Reads | Writes |
|---|---|---|
| Step 02 | Repository raw MIND train/dev files | `data/processed/v2_colab_check/` |
| Step 03 | Original `data/processed/v2/` and repository entity embeddings | `outputs/features/unified_v1_colab_check/` |
| Steps 04–05 | Original `data/processed/v2/`; advanced models also use `outputs/features/unified_v1/` | Model-specific run folders under `outputs/models/` |

Step 02 does not automatically replace the inputs consumed by Step 03 or the
models. Step 03 currently requires existing preprocessing because
`BUILD_PROCESSED_IF_MISSING=False`. Before another reproduction build, choose an
unused `PROCESSED_VERSION` in Step 02 or `OUTPUT_VERSION` in Step 03; existing
completed folders are preserved. Do not switch the report models to the check
bundles merely because those bundles were regenerated.

## How to run in Google Colab

1. Open the shared Drive folder above and add a shortcut under **My Drive**.
   The expected folder is
   `/content/drive/MyDrive/BT4222 Group Project`. Keep that name or edit the
   notebook's `STORAGE_ROOT` to its actual mounted location.
2. Open a numbered notebook using its **Open in Colab** badge. A CPU runtime is
   sufficient for setup checks, EDA, preprocessing, feature generation, TF-IDF
   CBF and UBCF. Select a GPU before starting neural-model training when one is
   available.
3. Before running setup, check the notebook's storage paths, version settings
   and output folder. For the reported models, keep `PROCESSED_VERSION='v2'`
   and, where used, `FEATURE_VERSION='unified_v1'`.
4. Run the included setup cells from the top and authorise Drive access.
   **No separate clone command or ZIP upload is required for the numbered
   pipeline.** Steps 01–03 include repository-cloning cells that obtain the
   helper code and raw files. Steps 04–05 read the shared Drive bundles directly
   and do not require a repository clone.
5. Check the printed paths before continuing. Steps 02–03 use the separate
   reproduction folders documented above. Choose another unused output version
   if a previous check build already exists.
6. Run the remaining cells in order for a full experiment. For a setup-only
   check, stop after setup/input loading; this does not verify a complete
   training run. Preserve the completed performance outputs when saving the
   submission notebook. Execute final test evaluation only after model choices
   are fixed; some test cells set the evaluation flag themselves, so do not use
   **Run all** when you intend to skip test evaluation.

The repository is public, so the included clone cells do not require GitHub
authentication. Cloning is still needed internally for Steps 01–03; Drive
access alone does not supply their helper code or raw embeddings. The ZIP
fallback in Step 03 is disabled by default.

The main pipeline's Colab setup has been checked. Steps 01–03 also have completed
Colab reproduction outputs, and NeuMF has a completed T4 run. Setup checks for
the remaining models do not imply that their full saved experiments were rerun
in Colab; retain each notebook's execution/provenance notes. Supporting notebooks
have their own prerequisites and are not covered by these numbered-pipeline
setup instructions.

Full tuning and ablation studies take longer than setup checks. Runtimes depend
on the assigned hardware; a GPU is useful for neural training but is not needed
just to verify paths and input access.

For local model execution, copy `data/processed/v2` and
`outputs/features/unified_v1` from Drive into the same paths in this repository,
then:

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

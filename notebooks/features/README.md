# Member feature notebooks

Each member first built and tested their features in their own notebook.
`03_Unified_Feature_Engineering.ipynb` then rebuilt all of them in one
place, with the same training-only rules and leakage checks, and exported the
shared bundle `outputs/features/unified_v1`. The advanced models read only that bundle.

| Notebook | Features explored |
|---|---|
| `feature_engineering_zp.ipynb` | Basic news features, impression-time context and category interactions |
| `feature_engine_sy.ipynb` | Abstract entity counts, title and abstract TF-IDF, entity-type matching |
| `feature_engineering_ethan.ipynb` | Article entity vectors and history to candidate entity similarity |
| `feature_engineering_johannes.ipynb` | Impression counts and entity confidence |
| `feature_engineering_chris.ipynb` | Category encoding, historical popularity and CTR, entity prominence |
| `history_mode_check_chris.ipynb` | Which version of the CTR history features to train on |

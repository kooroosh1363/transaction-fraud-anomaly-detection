# Data Source

Dataset: public anonymized credit-card fraud benchmark distributed through OpenML under the dataset name `creditcard` (version 1).

Expected source properties used as integrity checks:

- rows: 284,807
- fraud rows (`Class=1`): 492
- target: `Class`
- predictors: `Time`, `Amount`, `V1`–`V28`

The pipeline downloads the dataset through scikit-learn `fetch_openml`, caches it locally under `data/raw/creditcard.csv`, and refuses to proceed if the expected row count, fraud count, or schema does not match.

The raw/cache directory and generated artifacts are not committed to Git.

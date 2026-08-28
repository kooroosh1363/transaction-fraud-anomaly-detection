# Data Dictionary

| Field | Meaning |
| --- | --- |
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset; not a real calendar timestamp |
| `V1`–`V28` | Anonymized transformed numerical variables supplied by the benchmark |
| `Amount` | Transaction amount |
| `Class` | Binary target: 1 = fraud, 0 = legitimate |

Because the `V` variables are anonymized/transformed, feature attribution can be predictive but is not directly business-interpretable without the original feature definitions.

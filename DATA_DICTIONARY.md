# Data Dictionary

| Field | Meaning |
| --- | --- |
| `SequenceIndex` | Preserved source row order created by this project when the retrieved OpenML variant omits `Time`; used only for forward-only splitting and never as a model feature |
| `Time` | Optional source field: elapsed seconds from the first transaction when present in a compatible source variant; not a real calendar timestamp |
| `V1`–`V28` | Anonymized transformed numerical variables supplied by the benchmark |
| `Amount` | Transaction amount |
| `Class` | Binary target: 1 = fraud, 0 = legitimate |

In the CI-validated OpenML retrieval used by this project, `Time` is absent. The fitted models therefore use **V1–V28 + Amount (29 features)**. `SequenceIndex` exists only to preserve forward ordering.

Because the `V` variables are anonymized/transformed, predictive feature attribution would not be directly business-interpretable without the original feature definitions.

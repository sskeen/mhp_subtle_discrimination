# src/notebooks/

IPython notebooks for MHP response annotation and multilabel text classification.

| File | Description |
|------|-------------|
| `annotation_pipeline.ipynb` | Organized iterative, independent co-annotation of audit correspondence field experiment responses. Sampled pilot + parent study data by cycle, computed Cohen's kappa, flagged discrepant tagging decisions for in-person deliberation, and compiled `d_annotated` (n = 1,923) for GPT triangulation. |
| `clr_pipeline.ipynb` | Triangulates human-GPT annotation decisions via "majority vote" approach. Preprocesses and augments training data, subsets sparsely labeled markers for reflexive qual analysis, trains multilabel ModernBERT classifier for configurable targets (`afrm`, `agnt`, `fitt`, `refl`) with inverse-frequency class weights, k-fold cross-validation, and hyperparameter grid search. |

# src/

Source code for MHP subtle discrimination multilabel classification and LLM-human annotation triangulation.

| File | Description |
|------|-------------|
| `annotate.py` | Functions for CHALET-style LLM-human deductive annotation: `load_annotation_config()` for YAML schema parsing, `build_prompt_gpt()` for prompt construction, `code_texts_deductively_gpt()` for OpenAI API inference, `majority_vote_gpt()` for human-GPT triangulation, and inter-rater reliability metrics (Cohen's kappa via `calculate_kappa_by_cycle()`). Includes spaCy NER-based PII redaction and data preprocessing utilities. |
| `notebooks/` | IPython notebooks orchestrating annotation and classification pipelines. See `notebooks/README.md`. |

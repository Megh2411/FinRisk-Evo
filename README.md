# FinRisk-Evo

FinRisk-Evo is a research prototype for comparing narrative passages from consecutive corporate annual reports. It combines PDF text-block filtering, two text-preprocessing streams, an NVDM latent representation, a FinBERT-based encoder, baseline retrieval and reranking, a two-step BiGRU-attention model, and a heuristic Financial Risk Escalation Score (FRES).

> **Research status:** this is a prototype and exploratory case study, not a validated financial forecasting or calibrated risk-prediction system. The manuscript reports an annotation exercise with 150 paragraph pairs; 45 pairs were shared across all three annotators, and nominal Krippendorff's alpha was 0.250. No predictive-accuracy, cross-sector-generalization, calibration-superiority, or financial-outcome claims are made.

## Repository structure

```text
FinRisk-Evo/
├── dashboard/              # Streamlit user interface
├── data/                   # Example output and project data placeholders
├── paper/                  # One canonical manuscript source and compiled draft
├── src/
│   ├── data_prep/          # PDF extraction and text processing
│   ├── models/             # NVDM, BiGRU-attention, taxonomy, FRES scoring
│   ├── rag_engine/         # Baseline vector retrieval, reranking, LLM summary
│   ├── training/           # NVDM, TNCRM and BiGRU training scripts
│   ├── compute_master_scores.py
│   ├── generate_risk_report.py
│   └── run_pipeline.py
├── tests/                  # Lightweight checks; no model-quality claims
├── requirements.txt
└── requirements-dev.txt
```

The Springer Nature manuscript has one canonical source at `paper/main.tex`, one bibliography at `paper/references.bib`, and one compiled working draft at `paper/main.pdf`. Legacy manuscript copies, generated build logs, and unrelated old presentation/project exports have been excluded to avoid duplicate and confusing artifacts.

## Pipeline overview

1. Extract text blocks from baseline and target report PDFs with PyMuPDF.
2. Create an NVDM-oriented bag-of-words stream and a lightly normalized transformer stream.
3. Retrieve up to five baseline candidates from ChromaDB and rerank them with a cross-encoder.
4. Encode each paragraph using the NVDM latent mean and FinBERT CLS embedding (20 + 768 = 788 features).
5. Run the baseline/target pair through the BiGRU-attention model.
6. Apply the fixed similarity gate and pair-local FRES formula.
7. Inspect the output through the Streamlit dashboard or CSV.

The NVDM latent mean is **not** a normalized topic-probability distribution. The risk-category mapping is manually specified and uses the latent representation; it is not a learned topic-to-risk classifier.

## Scoring interpretation

The score implementation is centralized in `src/models/fres_scoring.py`:

- semantic-similarity gate: `0.93`;
- fixed normalization constant: `0.229`;
- shift offset: `1.1`;
- output range: `0–100`.

These constants are implementation heuristics, not independently validated calibration estimates. FRES is a display index, not a probability or a validated measure of absolute financial risk. Scores must not be min-max rescaled within a batch: a paragraph's score should not change because unrelated rows were included.

## Training data limitations

The BiGRU training script creates 400 synthetic samples from ten hand-written baseline/target pairs repeated 40 times, with Gaussian noise added to feature vectors. This is a small synthetic probe set, not 400 independent real-world transitions and not an annotated corpus from actual historical reports. It cannot establish generalization or real-world classification performance. The training objective uses `BCEWithLogitsLoss`, consistent with the model's raw-logit output.

## Reproducibility limitations

- Model checkpoints are not included because `.pt`, `.pkl`, and the FinBERT checkpoint directory are excluded by `.gitignore`. A fresh clone cannot run end-to-end inference until the required artifacts are supplied or retrained.
- The repository does not contain a provenance-complete, independently labelled holdout dataset.
- The annotation agreement is limited; the labels should be treated as exploratory rather than a high-reliability gold standard.
- The code paths have lightweight tests, but end-to-end inference and model predictive quality have not been validated in a clean environment.
- Verify report provenance and redistribution rights before releasing annual-report excerpts or derived text.

## Setup

Use a virtual environment with a Python/PyTorch/Transformers combination compatible with the experiment. Install dependencies with:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

`requirements.txt` specifies minimum versions rather than a fully resolved lockfile, so installs are not guaranteed to recreate the original environment exactly. Record Python, PyTorch, Transformers, CUDA, and hardware versions for any reported experiment.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard requires the NVDM weights and vectorizer, the fine-tuned FinBERT checkpoint directory, and BiGRU weights at the paths expected by the pipeline. These are not included in this repository. Do not claim a fresh-clone inference run works until the artifacts are supplied and tested.

For optional Gemini summaries, set `GEMINI_API_KEY` in your environment. Never commit API keys, tokens, `.env` files, or credentials. Any credential that was previously exposed publicly should be revoked/rotated; deleting it from the current code does not remove it from Git history.

## Tests

```bash
python -m pytest tests/test_fres_scoring.py -q
python -m py_compile src/models/fres_scoring.py src/models/risk_drift.py src/models/bigru_attn.py src/compute_master_scores.py src/training/train_bigru.py
```

The FRES tests validate the formula, similarity gate, range cap, and pair-local normalization behavior. They do **not** validate the predictive quality of the neural models.

## Manuscript

The current JIIS working manuscript is in [`paper/`](paper/). It remains a draft: the authors must verify author declarations and annotation-ethics wording, reproduce any reported calculations from archived artifacts, and freeze the exact tested commit before submission.

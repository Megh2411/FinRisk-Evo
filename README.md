# FinRisk-Evo

FinRisk-Evo is a research prototype for comparing narrative passages from consecutive corporate annual reports. It combines PDF text extraction, dual-stream preprocessing, an NVDM latent representation, a FinBERT-based encoder, baseline retrieval and reranking, a two-step BiGRU-attention model, and a heuristic Financial Risk Escalation Score (FRES).

> **Research status:** This is an exploratory prototype, not a validated financial forecasting or calibrated risk-prediction system. The manuscript reports an annotation exercise with 150 paragraph pairs; 45 pairs were shared across all three annotators, and nominal Krippendorff's alpha was 0.250. The current evidence does not establish predictive accuracy, cross-sector generalization, calibration superiority, or association with financial outcomes.

## Repository structure

- `dashboard/app.py`: Streamlit interface.
- `src/data_prep/`: PDF extraction and dual-stream text processing.
- `src/models/`: NVDM, FinBERT integration, BiGRU-attention, taxonomy, and FRES scoring.
- `src/rag_engine/`: ChromaDB baseline retrieval, cross-encoder reranking, optional Gemini explanations.
- `src/build_baseline_index.py`: builds the named baseline index in `data/04_vector_db_v2`.
- `src/compute_master_scores.py`: batch scoring and CSV diagnostics.
- `src/run_pipeline.py`: pipeline orchestration.
- `src/training/`: model training scripts.
- `tests/`: lightweight automated checks.
- `paper/`: JIIS manuscript source and working draft.

## Pipeline overview

1. Extract text blocks from baseline and target annual-report PDFs.
2. Create an NVDM-oriented text stream and a lightly normalized transformer/retrieval stream.
3. Retrieve up to five baseline candidates from ChromaDB and rerank them with a cross-encoder.
4. Encode each paragraph using the NVDM latent mean and FinBERT CLS embedding (20 + 768 = 788 features).
5. Run the baseline/target pair through the BiGRU-attention model.
6. Apply the fixed similarity gate and pair-local FRES formula.
7. Review the output in the dashboard or a generated CSV.

The NVDM latent mean is **not** a normalized topic-probability distribution. The risk-category mapping is manually specified; it is not a learned topic-to-risk classifier. Retrieval and reranking identify candidate passages, but do not guarantee that passages are financially equivalent or directly comparable.

## FRES interpretation

The score implementation is centralized in `src/models/fres_scoring.py`. Current implementation constants include a semantic-similarity gate of `0.93`, a normalization constant of `0.229`, and a shift offset of `1.1`, with the display score capped to `0–100`.

These are implementation heuristics, not independently validated calibration estimates. FRES is a display index, **not a probability** or a validated measure of absolute financial risk. Do not min-max rescale scores within a batch: a paragraph's score should not change because unrelated rows were included.

## Training data and evaluation limitations

The BiGRU training script creates 400 synthetic samples from ten hand-written baseline/target pairs repeated 40 times, with Gaussian noise added to feature vectors. These are not 400 independent real-world transitions or an annotated corpus of historical reports. This setup cannot establish real-world generalization or classification performance.

The annotation agreement is limited. There is no provenance-complete, independently labelled holdout dataset in the repository, and the current tests do not validate neural-model predictive quality. Manual inspection of a small retrieval smoke test found some questionable baseline matches; retrieval quality therefore requires further evaluation before making reliability claims.

## Setup

Use a Python environment compatible with the project's PyTorch, Transformers, ChromaDB, and sentence-transformers dependencies.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

**macOS/Linux**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The requirements files specify version ranges rather than a fully resolved lockfile, so installations may not reproduce the original environment exactly. Record Python, PyTorch, Transformers, scikit-learn, CUDA, and hardware versions for reported experiments. A scikit-learn version mismatch with the saved vectorizer may produce a warning and should be resolved by using a compatible version or regenerating the vectorizer.

## Required local data and model artifacts

The repository intentionally excludes raw and processed financial reports, model weights, and local vector databases. A fresh clone **cannot run full inference immediately**.

Before indexing and scoring, supply the processed baseline JSON files expected by `src/build_baseline_index.py` in `data/03_processed/`:

- `Mahindra_FY24_dual_stream.json`, whose `document_name` is `Mahindra_FY24`
- `RIL_IAR_2024_dual_stream.json`, whose `document_name` is `RIL_IAR_2024`

The scoring pipeline also expects these local artifacts:

- `src/models/saved_weights/nvdm_weights.pt`
- `src/models/saved_weights/nvdm_vectorizer.pkl`
- `src/models/finbert_tncrm_weights/`
- `src/models/saved_weights/bigru_weights.pt`

Obtain or regenerate these artifacts using the project's data and training process before attempting full inference. Confirm report provenance and redistribution rights before sharing annual-report files or derived text.

## Build the baseline index and score a target

Run commands from the repository root. The index builder writes to the separate local database `data/04_vector_db_v2`; it does not use or overwrite the legacy `data/04_vector_db` directory.

```powershell
python src/build_baseline_index.py
```

The builder expects both baseline JSON files listed above. It tags each baseline as `Baseline` and distinguishes them with `document_tag`.

Example: score up to 200 paragraphs from Mahindra FY25 against Mahindra FY24:

```powershell
python src/compute_master_scores.py `
  "data/03_processed/Mahindra_FY25_dual_stream.json" `
  "data/diagnostics/Mahindra_FY25_scores.csv" `
  --baseline-document-tag Mahindra_FY24 `
  --max-nodes 200
```

For a Reliance target, use the corresponding target JSON and `--baseline-document-tag RIL_IAR_2024`. Make sure the target and baseline are from the same company and the intended reporting periods. Generated databases and diagnostics are local outputs and are ignored by Git.

The scoring script records row status and errors in the CSV. A successful run demonstrates execution, not that every retrieved pair is comparable or that FRES is validated. Inspect matched passages and report retrieval and evaluation limitations.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard requires the local model artifacts listed above and the data/database appropriate to the selected workflow. It should not be described as working from a fresh clone until those files are supplied and the workflow is tested.

Gemini summaries are optional. Set `GEMINI_API_KEY` in your environment to enable them. Do not put keys in source code, commit them, or paste them into issue reports. A previously exposed credential must be revoked. Rewriting Git history does not guarantee that old copies have disappeared from forks, caches, pull requests, or existing clones.

## Tests and checks

Run the lightweight FRES tests:

```bash
python -m pytest tests/test_fres_scoring.py -q
```

Compile the main modules:

```bash
python -m py_compile src/models/fres_scoring.py src/models/risk_drift.py src/models/bigru_attn.py src/compute_master_scores.py src/rag_engine/vector_store.py src/rag_engine/re_ranker.py src/rag_engine/llm_generator.py src/build_baseline_index.py
```

These checks cover basic code execution and FRES formula behavior. They do **not** validate retrieval accuracy, end-to-end reproducibility in a clean environment, calibration, or financial predictive performance.

## Manuscript and submission readiness

The JIIS working manuscript is in `paper/`. Before submission, verify author declarations and annotation-ethics wording, reproduce reported calculations from archived artifacts, confirm references and report permissions, document the exact tested environment, and freeze the exact commit used for the reported results.
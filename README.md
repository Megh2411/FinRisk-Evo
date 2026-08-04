# FinRisk-Evo

FinRisk-Evo is a document-intelligence pipeline for comparing annual reports across years and turning narrative change into a structured Financial Risk Escalation Score (FRES). The project extracts text from PDFs, normalizes it into two processing streams, learns topical and semantic representations, retrieves the closest baseline context, and scores how much risk has drifted from one reporting period to the next.

The current implementation is centered on annual-report comparison, with the dashboard acting as a human-facing layer over the same pipeline used in batch mode.

## What This Project Does

The system is designed to answer a simple but high-value question:

How much has the risk language in a company report changed from the baseline year to the target year, and where is that change concentrated?

To do that, the project:

1. Extracts paragraph-level text from annual report PDFs.
2. Cleans and splits the text into two specialized streams.
3. Builds a targeted corpus of numerically dense or magnitude-bearing sentences.
4. Learns latent topic structure with an NVDM topic model.
5. Fine-tunes a FinBERT-based masked-language model on TNCRM-style financial sentences.
6. Combines topic vectors and semantic embeddings into a temporal sequence model.
7. Retrieves the best baseline match from a persistent vector database.
8. Produces a normalized FRES score, confidence, dispersion, and attention signals.
9. Displays the result in a Streamlit dashboard and can also generate short LLM explanations.

## Repository Layout

- [dashboard/app.py](dashboard/app.py): Streamlit UI for uploading reports and exploring the output.
- [src/run_pipeline.py](src/run_pipeline.py): End-to-end orchestrator for parsing, processing, ingesting, and scoring.
- [src/compute_master_scores.py](src/compute_master_scores.py): Batch scoring entry point that writes `master_fres_scores.csv`.
- [src/generate_risk_report.py](src/generate_risk_report.py): Demo-style report generator for a single target report.
- [src/data_prep/pdf_extractor.py](src/data_prep/pdf_extractor.py): PDF paragraph extraction using PyMuPDF.
- [src/data_prep/text_pipeline.py](src/data_prep/text_pipeline.py): Dual-stream text normalization.
- [src/data_prep/tncrm_dataset.py](src/data_prep/tncrm_dataset.py): TNCRM sentence selection for FinBERT training.
- [src/models/nvdm.py](src/models/nvdm.py): Neural Variational Document Model used for topic structure and dispersion.
- [src/models/bigru_attn.py](src/models/bigru_attn.py): Temporal BiGRU with attention and uncertainty estimation.
- [src/models/risk_drift.py](src/models/risk_drift.py): Master inference pipeline that fuses the model stack.
- [src/rag_engine/vector_store.py](src/rag_engine/vector_store.py): Persistent ChromaDB baseline retrieval.
- [src/rag_engine/re_ranker.py](src/rag_engine/re_ranker.py): Cross-encoder reranker that picks the best baseline paragraph.
- [src/rag_engine/llm_generator.py](src/rag_engine/llm_generator.py): Gemini-based executive-summary generator.
- [src/models/taxonomy.py](src/models/taxonomy.py): Heuristic risk-type labeling by text cues.

## End-To-End Workflow

### 1. PDF extraction

The pipeline starts in [src/data_prep/pdf_extractor.py](src/data_prep/pdf_extractor.py). It uses PyMuPDF to read page blocks from a PDF, keeps only text-like blocks, and stores paragraph nodes in JSON under `data/02_parsed/`.

This step is intentionally conservative: it prefers readable narrative paragraphs over tables, headers, and short fragments so the downstream models receive cleaner financial text.

### 2. Dual-stream preprocessing

[src/data_prep/text_pipeline.py](src/data_prep/text_pipeline.py) converts each extracted paragraph into two views:

- Stream 1 for NVDM: lowercase, punctuation-stripped text optimized for bag-of-words topic modeling.
- Stream 2 for FinBERT and retrieval: minimally cleaned text that preserves sentence order and wording.

This split matters because topic models and transformer models benefit from different text representations. The project does not force a single preprocessing scheme onto all models.

### 3. TNCRM sentence selection

[src/data_prep/tncrm_dataset.py](src/data_prep/tncrm_dataset.py) builds a Targeted Numeric and Change-Related Masking corpus. It selects sentences that contain:

- numeric values,
- percentages,
- trend or magnitude verbs such as rise, decline, surge, fall, increase, and decrease.

The purpose is to concentrate FinBERT post-training on the kind of language that usually signals operational change, financial pressure, or strategic drift.

### 4. Baseline retrieval

[src/rag_engine/vector_store.py](src/rag_engine/vector_store.py) stores the baseline year paragraphs in ChromaDB using `all-MiniLM-L6-v2` embeddings. When the target-year paragraph is analyzed, the vector store retrieves semantically similar baseline candidates from the chosen baseline year.

[src/rag_engine/re_ranker.py](src/rag_engine/re_ranker.py) then reranks those candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` so the final baseline paragraph is selected using a stronger pairwise relevance score rather than embedding similarity alone.

### 5. Temporal scoring

[src/models/risk_drift.py](src/models/risk_drift.py) is the main inference engine. It loads the topic model, the FinBERT checkpoint, and the BiGRU attention model, then fuses them into a single temporal risk estimate.

The scoring logic is:

- represent each paragraph with a topic vector from NVDM,
- represent the same paragraph with a semantic vector from FinBERT,
- concatenate those vectors into a fused representation,
- compare baseline and target representations with a BiGRU over a two-step sequence,
- gate the final risk score using semantic similarity so near-identical statements are suppressed.

The output is a tuple of:

- FRES score,
- attention weight focused on the target-year paragraph,
- confidence estimate from Monte Carlo dropout,
- topic dispersion,
- semantic similarity.

## Model Stack

### NVDM: Neural Variational Document Model

Implemented in [src/models/nvdm.py](src/models/nvdm.py).

Purpose:

- capture latent financial topics from paragraph-level text,
- provide a compact topic vector for each paragraph,
- estimate topic dispersion as a proxy for how spread out or uncertain the latent content is.

Logic:

- input is a bag-of-words vector built from the Stream 1 text,
- the encoder maps sparse counts into a hidden state,
- two linear heads produce `mu` and `logvar` for a variational latent topic space,
- a reparameterized latent vector is decoded back into vocabulary space,
- KL divergence over `mu` and `logvar` is used as topic dispersion.

Why it exists here:

- annual-report narrative often clusters around recurring financial themes,
- a topic model gives the system a structural view that is different from sentence embedding similarity,
- that structure helps the temporal model detect shifts in topic intensity, not just wording changes.

### FinBERT TNCRM checkpoint

Stored in [src/models/finbert_tncrm_weights/](src/models/finbert_tncrm_weights/).

Purpose:

- provide domain-aware semantic embeddings for financial text,
- stay sensitive to risk-bearing wording, numeric statements, and report language patterns.

Logic:

- the project starts from `ProsusAI/finbert`,
- it continues masked-language-model training on the TNCRM corpus built from the extracted financial sentences,
- the resulting checkpoint is used through `AutoModel` to produce a CLS embedding for each paragraph.

Why it exists here:

- generic language models are often too broad for annual-report analysis,
- FinBERT gives the pipeline a finance-specific semantic space,
- the TNCRM step makes that space even more responsive to numeric change and magnitude language.

Note:

- [src/models/finbert_tncrm.py](src/models/finbert_tncrm.py) is currently empty, but the active model is the fine-tuned checkpoint in `finbert_tncrm_weights/`.

### TemporalRiskBiGRU

Implemented in [src/models/bigru_attn.py](src/models/bigru_attn.py).

Purpose:

- compare the baseline and target paragraph representations as a short temporal sequence,
- predict whether the transition indicates low drift or high drift,
- expose attention weights so the target-year contribution is interpretable,
- estimate confidence using Monte Carlo dropout.

Logic:

- the input sequence contains two time steps: baseline and target,
- a bidirectional GRU processes the sequence in both directions,
- an attention layer weights the hidden states,
- a feed-forward head outputs a risk logit,
- repeated stochastic forward passes produce mean score and confidence.

Why it exists here:

- risk change is sequential, not static,
- the model needs to know how the current paragraph differs from the prior-year baseline,
- attention makes the final score easier to inspect.

### MasterRiskPipeline

Implemented in [src/models/risk_drift.py](src/models/risk_drift.py).

Purpose:

- combine NVDM, FinBERT, and the BiGRU into one scoring engine.

Logic:

- `get_fused_vector()` creates one vector from topic structure and one from semantic encoding,
- `evaluate_temporal_risk()` computes cosine similarity between semantic embeddings,
- a high similarity threshold suppresses trivial no-change cases,
- otherwise the BiGRU score is scaled by the similarity gap to create FRES.

This gating is important because the project is meant to highlight escalation, not just any textual difference.

### Auxiliary models

- [src/rag_engine/neural_validator.py](src/rag_engine/neural_validator.py): zero-shot BART-MNLI validator for checking whether external news supports a reported risk claim. This is a supporting module and is not part of the main dashboard path.
- [src/models/taxonomy.py](src/models/taxonomy.py): lightweight heuristic classifier that assigns coarse risk categories such as regulatory shift, ESG drift, operational complexity, or governance variance.
- [src/rag_engine/llm_generator.py](src/rag_engine/llm_generator.py): Gemini 2.5 Flash wrapper for short executive summaries of the baseline-versus-target comparison.

## Training Artifacts

The repository already contains trained weights and derived assets under:

- `src/models/saved_weights/nvdm_weights.pt`
- `src/models/saved_weights/nvdm_vectorizer.pkl`
- `src/models/saved_weights/bigru_weights.pt`
- `src/models/finbert_tncrm_weights/`

These files are what make the scoring pipeline usable without retraining from scratch.

## Data Layout

- `data/01_raw/`: uploaded or source PDFs.
- `data/02_parsed/`: paragraph-level JSON extracted from PDFs.
- `data/03_processed/`: dual-stream JSON and TNCRM corpora.
- `data/04_vector_db/`: persistent ChromaDB baseline store.
- `data/master_fres_scores.csv`: scored output from batch runs.

## How To Run

### 1. Create and activate a virtual environment

From the project root on Windows, use PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, run the `.bat` activation instead from Command Prompt:

```bat
.venv\Scripts\activate.bat
```

If the activation scripts are not present in `.venv\Scripts`, the virtual environment is incomplete. Recreate it with `python -m venv .venv`, then use the activation command above.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit dashboard

Run the app from the project root:

```bash
streamlit run dashboard/app.py
```

If `streamlit` is not on your PATH or the venv is not activated, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The dashboard expects two PDFs:

- a baseline report, typically FY24,
- a target report, typically FY25.

After upload, it runs the full pipeline and refreshes the risk table and 3D scatter plot.

### 4. Optional: run the pipeline from scripts

If you want to work outside the dashboard, you can use the script entry points directly:

- `python src/generate_risk_report.py` to generate a single comparative risk report.
- `python src/compute_master_scores.py` to score processed target text and write `data/master_fres_scores.csv`.

### 5. Optional: enable LLM summaries

The dashboard can generate a short executive summary for each result row using Gemini. Set `GEMINI_API_KEY` in your environment before launching the app if you want that feature enabled.

### When the pipeline runs

The dashboard performs the full workflow automatically after you upload both PDFs:

1. Extract raw text into `data/02_parsed/`.
2. Create dual-stream JSON in `data/03_processed/`.
3. Refresh the baseline vector store in `data/04_vector_db/`.
4. Score the target report and write `data/master_fres_scores.csv`.

## Output Interpretation

The main output fields are:

- FRES: normalized risk escalation score.
- attn_25: attention weight focused on the target-year text.
- confidence: uncertainty estimate from BiGRU dropout sampling.
- dispersion: latent-topic spread from NVDM.
- semantic similarity: cosine similarity between baseline and target FinBERT embeddings.

Higher FRES indicates stronger escalation relative to the baseline paragraph. High attention and high confidence make the result more trustworthy, while high semantic similarity usually means the two paragraphs are describing the same idea and should not be treated as a real escalation.

## Practical Notes

- The pipeline is built for annual-report style documents, not arbitrary PDFs.
- It works best when the baseline and target documents are structurally similar.
- The baseline vector store is persistent, so rerunning the pipeline can reuse previous ingestion unless the collection is cleared.
- The dashboard uses `GEMINI_API_KEY` for the LLM summary generator.

## What Has Been Built

In short, the project turns raw annual-report PDFs into a comparative risk-intelligence system. The extraction layer isolates narrative paragraphs, the preprocessing layer separates topic-oriented and transformer-oriented text, the retrieval layer finds the closest baseline context, and the model stack combines latent topics, finance-specific semantic embeddings, and temporal sequence learning to score risk drift in a way that is both quantitative and inspectable.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("experiments/retrieval_eval")
INPUT = ROOT / "pooled_candidates_for_labeling_with_queries.xlsx"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Workbook not found: {INPUT}")

df = pd.read_excel(INPUT, sheet_name=0)
label_col = "relevance_0_irrelevant_1_partial_2_relevant"

required = {"query_id", "method", "rank", "doc_name", "para_id", label_col}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
if df[label_col].isna().any():
    raise ValueError("Some relevance labels are blank or invalid.")
if not df[label_col].isin([0, 1, 2]).all():
    raise ValueError("Relevance labels must be 0, 1, or 2.")

df["rank"] = pd.to_numeric(df["rank"], errors="raise")
df["relevance"] = df[label_col].astype(int)

# One relevance judgement per unique query-passage pair.
pool = df.drop_duplicates(["query_id", "doc_name", "para_id"])
pooled_counts = {}
for qid, group in pool.groupby("query_id"):
    pooled_counts[qid] = {
        "partial_or_better": int((group["relevance"] >= 1).sum()),
        "fully_relevant": int((group["relevance"] == 2).sum()),
    }

def ndcg_at_k(rels, k=5):
    rels = np.asarray(rels[:k], dtype=float)
    discounts = 1 / np.log2(np.arange(2, len(rels) + 2))
    dcg = np.sum((2**rels - 1) * discounts)
    ideal = np.sort(rels)[::-1]
    idcg = np.sum((2**ideal - 1) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0

rows = []
for (qid, method), group in df.groupby(["query_id", "method"]):
    ranked = group.sort_values("rank").head(5)
    rels = ranked["relevance"].tolist()
    n = len(ranked)
    if n == 0:
        continue

    partial_hits = sum(r >= 1 for r in rels)
    full_hits = sum(r == 2 for r in rels)
    denom_partial = pooled_counts[qid]["partial_or_better"]
    denom_full = pooled_counts[qid]["fully_relevant"]

    rows.append({
        "query_id": qid,
        "method": method,
        "n_results": n,
        "P@5_partial_or_better": partial_hits / 5,
        "P@5_fully_relevant": full_hits / 5,
        "nDCG@5": ndcg_at_k(rels, 5),
        "pooled_Recall@5_partial_or_better":
            partial_hits / denom_partial if denom_partial else np.nan,
        "pooled_Recall@5_fully_relevant":
            full_hits / denom_full if denom_full else np.nan,
    })

per_query = pd.DataFrame(rows)
per_query.to_csv(OUT / "per_query_metrics.csv", index=False)

metric_cols = [
    "P@5_partial_or_better",
    "P@5_fully_relevant",
    "nDCG@5",
    "pooled_Recall@5_partial_or_better",
    "pooled_Recall@5_fully_relevant",
]
summary = per_query.groupby("method")[metric_cols].mean().reset_index()
summary.to_csv(OUT / "aggregate_metrics.csv", index=False)

# Plot nDCG@5 comparison.
plot_df = summary.sort_values("nDCG@5", ascending=True)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(plot_df["method"], plot_df["nDCG@5"], edgecolor="black")
ax.set_xlim(0, 1)
ax.set_xlabel("Mean nDCG@5")
ax.set_title("Retrieval Method Comparison (12-query evaluation)")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "retrieval_ndcg_comparison.png", dpi=220)
plt.close(fig)

print("\nANNOTATIONS")
print("Rows:", len(df))
print("Queries:", df["query_id"].nunique())
print("Labels:", df["relevance"].value_counts().sort_index().to_dict())
print("\nAGGREGATE METRICS (macro-average over queries)")
print(summary.round(3).to_string(index=False))
print("\nSaved:")
print(OUT / "aggregate_metrics.csv")
print(OUT / "per_query_metrics.csv")
print(OUT / "retrieval_ndcg_comparison.png")
print("\nRecall is pooled recall over unique judged passages, not corpus-wide recall.")

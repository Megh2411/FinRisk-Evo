from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "master_fres_scores.csv"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
for col in ["fres", "similarity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["fres", "similarity"])
df["category"] = df["category"].fillna("Unspecified").astype(str)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.hist(df["fres"], bins=16, edgecolor="black")
ax.set_xlabel("FRES score")
ax.set_ylabel("Number of scored pairs")
ax.set_title("Distribution of Pipeline-Generated FRES Scores")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fres_score_distribution.png", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.scatter(df["similarity"], df["fres"], alpha=0.65)
ax.set_xlabel("Similarity score")
ax.set_ylabel("FRES score")
ax.set_title("Similarity and FRES: Exploratory Output Relationship")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fres_vs_similarity.png", dpi=220)
plt.close(fig)

cat = df.groupby("category").agg(n=("fres", "size"), mean_fres=("fres", "mean")).sort_values("n")
fig, ax = plt.subplots(figsize=(8.2, 5.2))
y = range(len(cat))
ax.barh(list(y), cat["n"], edgecolor="black")
ax.set_yticks(list(y), [f"{name} (mean FRES {row.mean_fres:.1f})" for name, row in cat.iterrows()])
ax.set_xlabel("Number of scored pairs")
ax.set_title("Pipeline Output Counts by Assigned Category")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fres_category_counts.png", dpi=220)
plt.close(fig)

print(f"Generated three exploratory figures from {len(df)} scored pairs in {DATA}")

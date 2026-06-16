"""E1: correlation heatmap + top-10 pairs for each dataset."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from src.data import load_all
from src.grouping import correlation_matrix

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_one(d) -> None:
    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(d.X)
    corr = correlation_matrix(X)
    fig, ax = plt.subplots(figsize=(max(8, d.X.shape[1] * 0.35), max(7, d.X.shape[1] * 0.32)))
    sns.heatmap(corr, xticklabels=d.feature_names, yticklabels=d.feature_names,
                cmap="Blues", vmin=0, vmax=1, ax=ax, square=True,
                cbar_kws={"label": "|corr|"})
    ax.set_title(f"{d.name}: |correlation| matrix")
    plt.tight_layout()
    out = FIG_DIR / f"corr_{d.name}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    pairs = []
    n = d.X.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((corr[i, j], d.feature_names[i], d.feature_names[j]))
    pairs.sort(reverse=True)
    print(f"top 10 {d.name} pairs:")
    for r, a, b in pairs[:10]:
        print(f"  {r:.4f}  {a} <-> {b}")


def main() -> None:
    for d in load_all():
        print(f"\n--- {d.name} ---")
        plot_one(d)


if __name__ == "__main__":
    main()

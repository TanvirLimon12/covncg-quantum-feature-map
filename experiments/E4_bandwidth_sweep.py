"""E4: bandwidth sweep c ∈ {0.05..2.0} per dataset. Picks c* by best CV3 MCC."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.data import LOADERS, load
from src.diagnostics import summarise
from src.feature_select import top_k_mi
from src.grouping import group_greedy
from src.kernel import build_gram_matrix

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
TAB_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def sweep(d, bandwidths, n_sub: int, n_feat: int) -> pd.DataFrame:
    sc = MinMaxScaler(feature_range=(0, np.pi))
    X_full = sc.fit_transform(d.X)
    y = d.y
    feat_idx = top_k_mi(X_full, y, k=n_feat)
    print(f"[{d.name}] top-{len(feat_idx)} MI feature idx: {feat_idx}")
    X = X_full[:, feat_idx]

    n_sub = min(n_sub, len(X))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), n_sub, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    groups = group_greedy(X_sub, group_size=3)
    n_qubits_est = sum(2 if len(g) >= 2 else 1 for g in groups)
    print(f"[{d.name}] groups (size=3): {groups}  n_qubits={n_qubits_est}")

    rows = []
    for c in bandwidths:
        K_Q = build_gram_matrix(X_sub, groups, c=c)
        K_C = rbf_kernel(X_sub, gamma=1.0 / (2 * c * c))
        diag = summarise(K_Q, y_sub, K_C_rbf=K_C)
        svm = SVC(kernel="precomputed", C=1.0)
        diag["mcc_cv3"] = float(np.mean(cross_val_score(svm, K_Q, y_sub, cv=3, scoring="matthews_corrcoef")))
        diag["c"] = c
        diag["dataset"] = d.name
        print(f"  [{d.name}] c={c:>5} MCC={diag['mcc_cv3']:.4f} var={diag['off_diag_var']:.3e} "
              f"g={diag['g']:.3f} cond={diag['cond']:.2e}")
        rows.append(diag)
    return pd.DataFrame(rows)


def plot_frontier(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for name, g in df.groupby("dataset"):
        g = g.sort_values("c")
        axes[0].plot(g["c"], g["off_diag_var"], "o-", label=name)
        axes[1].plot(g["c"], g["g"], "o-", label=name)
        axes[2].plot(g["c"], g["mcc_cv3"], "o-", label=name)
    for ax, ylabel, title in zip(axes,
                                 ["off-diag var", "g", "MCC (cv3)"],
                                 ["concentration", "geometric difference", "accuracy"]):
        ax.set_xlabel("c"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xscale("log"); ax.legend()
    axes[0].set_yscale("log")
    axes[1].axhline(1.0, ls="--", color="gray")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    p.add_argument("--n-sub", type=int, default=80)
    p.add_argument("--n-feat", type=int, default=12,
                   help="top-k MI features (caps qubits ≈ 2*ceil(k/3))")
    p.add_argument("--bandwidths", nargs="+", type=float,
                   default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--out-suffix", type=str, default="")
    args = p.parse_args()

    dfs = [sweep(load(name), args.bandwidths, args.n_sub, args.n_feat)
           for name in args.datasets]
    df = pd.concat(dfs, ignore_index=True)
    suf = args.out_suffix
    out_csv = TAB_DIR / f"bandwidth_sweep{suf}.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    plot_frontier(df, FIG_DIR / f"bandwidth_sweep{suf}.png")

    print("\nbest c* per dataset (by MCC cv3):")
    for name, g in df.groupby("dataset"):
        best = g.loc[g["mcc_cv3"].idxmax()]
        print(f"  {name}: c*={best['c']:.3f}  MCC={best['mcc_cv3']:.4f}  "
              f"var={best['off_diag_var']:.3e}  g={best['g']:.3f}")


if __name__ == "__main__":
    main()

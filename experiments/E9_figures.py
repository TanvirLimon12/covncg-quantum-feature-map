"""E9: 4 core paper figures.

1. WDBC MCC bar comparison (ZZ, CovNCG, CPMap?, RBF, CPMap-lit)
2. Bandwidth / concentration frontier (3 datasets, 3 panels)
3. Covariance vs random grouping (multi-seed)
4. Kernel heatmaps: ZZ, CovNCG, RBF (WDBC, 60-sample subset)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler

from src.baselines import zz_fidelity_kernel
from src.data import load
from src.feature_select import top_k_mi
from src.grouping import group_greedy
from src.kernel import build_gram_matrix

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(name):
    with open(TAB / name) as f:
        return json.load(f)


# ---------- Figure 1: WDBC MCC bars ----------

def fig1_wdbc_bars():
    import re
    zz = _load("zz_baseline.json")["wdbc"]["mcc"]
    rbf = _load("rbf_baseline.json")["wdbc"]["mcc"]
    rand = _load("ablation_wdbc_random_c1.0.json")["folds"]["mcc"]
    cpmap = _load("cpmap_wdbc_c1.0_q6.json")["folds"]["mcc"]

    pat = re.compile(r"MCC=([\-0-9\.]+)\s+F1=")
    def folds_from(name):
        out = [float(m.group(1)) for m in pat.finditer((TAB/name).read_text())]
        return out[:5]
    cov_single = folds_from("E6_wdbc_balanced.log")
    cov_pair   = folds_from("E6_wdbc_pair_n200.log")
    cov_reps2_c025 = folds_from("E6_wdbc_reps2_c025.log")

    methods = ["ZZFeatureMap", "CovNCG\n(random)", "CovNCG\n(single, c=1)",
               "CovNCG\n(pair, n=200)", "CovNCG\n(reps=2, c=0.25)",
               "CPMap\n(ours)", "CPMap\n(lit)", "RBF-SVM\n(tuned)"]
    series = [zz, rand, cov_single, cov_pair, cov_reps2_c025, cpmap, [0.944], rbf]
    means = [np.mean(s) for s in series]
    stds  = [np.std(s) if len(s) > 1 else 0.0 for s in series]
    colors = ["#888", "#fb9a99", "#1f78b4", "#0d4a6d", "#072e44", "#33a02c", "#a6cee3", "#6a3d9a"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(methods, means, yerr=stds, color=colors, capsize=4, edgecolor="black")
    ax.set_ylabel("MCC")
    ax.set_title("WDBC: 5-fold CV MCC across CovNCG variants + baselines")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.628, ls="--", color="#888", alpha=0.5, label="ZZ floor")
    ax.axhline(0.948, ls="--", color="#6a3d9a", alpha=0.5, label="RBF ceiling")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    out = FIG / "fig1_wdbc_mcc.png"
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


# ---------- Figure 2: bandwidth frontier ----------

def fig2_bandwidth_frontier():
    dfs = []
    for ds in ["wdbc", "parkinsons", "heart"]:
        p = TAB / f"bandwidth_sweep_{ds}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        print("no bandwidth_sweep csvs"); return
    df = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    palette = {"wdbc": "#1f78b4", "parkinsons": "#33a02c", "heart_cleveland": "#e31a1c"}
    for ds, g in df.groupby("dataset"):
        g = g.sort_values("c")
        col = palette.get(ds, "black")
        axes[0].plot(g["c"], g["off_diag_var"], "o-", color=col, label=ds)
        axes[1].plot(g["c"], g["g"],            "o-", color=col, label=ds)
        axes[2].plot(g["c"], g["mcc_cv3"],      "o-", color=col, label=ds)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("bandwidth c"); axes[0].set_ylabel("off-diag Gram variance")
    axes[0].set_title("concentration vs bandwidth"); axes[0].legend()
    axes[1].set_xscale("log"); axes[1].axhline(1.0, ls="--", color="gray")
    axes[1].set_xlabel("bandwidth c"); axes[1].set_ylabel("g (Huang 2021)")
    axes[1].set_title("geometric difference vs bandwidth"); axes[1].legend()
    axes[2].set_xscale("log")
    axes[2].set_xlabel("bandwidth c"); axes[2].set_ylabel("MCC (cv3, 80 pts)")
    axes[2].set_title("accuracy vs bandwidth"); axes[2].legend()
    plt.tight_layout()
    out = FIG / "fig2_bandwidth_frontier.png"
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


# ---------- Figure 3: covariance vs random ablation ----------

def _parse_ablation_log(p: Path) -> list[float]:
    import re
    pat = re.compile(r"MCC=([\-0-9\.]+)\s+F1=")
    out = []
    for line in p.read_text().splitlines():
        m = pat.search(line)
        if m:
            out.append(float(m.group(1)))
    return out


def fig3_cov_vs_random():
    cov = _load("covncg_wdbc_c1.0_reps1.json")["folds"]["mcc"]
    log_files = sorted(TAB.glob("E8_wdbc_random*.log"))
    rand_runs = [_parse_ablation_log(p) for p in log_files]
    rand_runs = [r for r in rand_runs if len(r) == 5]
    rand_means = [np.mean(r) for r in rand_runs]
    rand_stds  = [np.std(r)  for r in rand_runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(np.mean(cov), ls="-", color="#1f78b4", label=f"covariance grouping ({np.mean(cov):.3f})")
    ax.fill_between([-0.5, len(rand_runs) - 0.5],
                    np.mean(cov) - np.std(cov), np.mean(cov) + np.std(cov),
                    color="#1f78b4", alpha=0.2)
    xs = list(range(len(rand_runs)))
    ax.errorbar(xs, rand_means, yerr=rand_stds, fmt="o", color="#e31a1c", capsize=4, label="random grouping (per-seed)")
    if rand_means:
        ax.axhline(np.mean(rand_means), ls="--", color="#e31a1c",
                   label=f"random mean ({np.mean(rand_means):.3f})")
    ax.set_xticks(xs); ax.set_xticklabels([f"seed{i}" for i in xs])
    ax.set_ylabel("MCC (5-fold CV)")
    ax.set_title("WDBC ablation: covariance vs random grouping (c=1.0)")
    ax.legend()
    plt.tight_layout()
    out = FIG / "fig3_cov_vs_random.png"
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


# ---------- Figure 4: kernel heatmaps (WDBC) ----------

def fig4_kernel_heatmaps():
    d = load("wdbc")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d.X), 60, replace=False)
    X_raw, y = d.X[idx], d.y[idx]
    order = np.argsort(y)
    X_raw, y = X_raw[order], y[order]

    sc = MinMaxScaler(feature_range=(0, np.pi))
    X_full = sc.fit_transform(X_raw)
    feat_idx = top_k_mi(X_full, y, k=12, seed=0)
    X = X_full[:, feat_idx]

    print("building ZZ Gram (60x60)...")
    K_zz = zz_fidelity_kernel(X, X, n_qubits=4, reps=2)
    print("building CovNCG Gram (60x60)...")
    groups = group_greedy(X, group_size=3)
    K_cov = build_gram_matrix(X, groups, c=1.0)
    K_rbf = rbf_kernel(X, gamma=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, K, title in zip(axes,
                            [K_zz, K_cov, K_rbf],
                            ["ZZFeatureMap (4 qubits)", "CovNCG c=1.0 (8 qubits)", "RBF γ=0.5"]):
        im = ax.imshow(K, cmap="viridis", aspect="equal")
        n_neg = int(np.sum(y == 0))
        ax.axhline(n_neg - 0.5, color="white", lw=0.8)
        ax.axvline(n_neg - 0.5, color="white", lw=0.8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out = FIG / "fig4_kernel_heatmaps.png"
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    fig1_wdbc_bars()
    fig2_bandwidth_frontier()
    fig3_cov_vs_random()
    fig4_kernel_heatmaps()


if __name__ == "__main__":
    main()

"""E15: paper-grade figures missing from initial set.

Adds:
  fig0: CovNCG circuit architecture (2-qubit block + multi-block layout)
  fig8: 5-dataset MCC headline bar chart
  fig9: pair-proj κ = 7/32 verification (B sweep)
  fig10: saturation curve Var vs c (log-log, single-block)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
pat = re.compile(r"MCC=([\-0-9\.]+)\s+F1=")


def folds(name):
    out = [float(m.group(1)) for m in pat.finditer((TAB / name).read_text())]
    return out[:5]


# ---------- fig0: CovNCG circuit diagram ----------

def fig0_circuit():
    """Schematic of CovNCG non-commuting block + multi-block layout."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5))

    # LEFT: single block
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 5)
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.text(5, 4.6, "CovNCG non-commuting block (3 features, 2 qubits)",
             ha="center", fontsize=11, weight="bold")

    # qubit lines
    for q, y in [(0, 3), (1, 1.5)]:
        ax0.plot([0.5, 9.5], [y, y], "k-", lw=1.5)
        ax0.text(0.1, y, f"|0⟩_{q}", fontsize=11, va="center")

    def gate(ax, x, y, w, h, label, color="lightblue"):
        ax.add_patch(Rectangle((x - w/2, y - h/2), w, h, facecolor=color, edgecolor="black"))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    # H gates
    gate(ax0, 1.2, 3, 0.6, 0.6, "H")
    gate(ax0, 1.2, 1.5, 0.6, 0.6, "H")
    # RX on q1: IX
    gate(ax0, 2.5, 1.5, 1.2, 0.6, "R_X(2cx_a)", "#ffe0a0")
    # RX on q0: XI
    gate(ax0, 4.0, 3, 1.2, 0.6, "R_X(2cx_b)", "#ffe0a0")
    # CX-RZ-CX
    ax0.plot([5.5, 5.5], [3, 1.5], "k-", lw=1.2)
    ax0.add_patch(plt.Circle((5.5, 3), 0.15, facecolor="black"))
    ax0.add_patch(plt.Circle((5.5, 1.5), 0.25, fill=False, edgecolor="black", lw=1.5))
    ax0.plot([5.35, 5.65], [1.5, 1.5], "k-", lw=1.2)
    ax0.plot([5.5, 5.5], [1.35, 1.65], "k-", lw=1.2)
    gate(ax0, 6.8, 1.5, 1.2, 0.6, "R_Z(2cx_c)", "#ffd0d0")
    ax0.plot([8.0, 8.0], [3, 1.5], "k-", lw=1.2)
    ax0.add_patch(plt.Circle((8.0, 3), 0.15, facecolor="black"))
    ax0.add_patch(plt.Circle((8.0, 1.5), 0.25, fill=False, edgecolor="black", lw=1.5))
    ax0.plot([7.85, 8.15], [1.5, 1.5], "k-", lw=1.2)
    ax0.plot([8.0, 8.0], [1.35, 1.65], "k-", lw=1.2)

    ax0.text(5, 0.4, r"$U_b(x_a, x_b, x_c) \approx \exp(-i c [x_a\,IX + x_b\,XI + x_c\,ZZ])$",
             ha="center", fontsize=10, style="italic")

    # RIGHT: multi-block layout
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 9)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.text(5, 8.5, "Multi-block CovNCG (B blocks, 2B qubits)",
             ha="center", fontsize=11, weight="bold")

    for b in range(4):
        y0 = 7 - b * 1.7
        y1 = y0 - 0.7
        ax1.plot([0.5, 9.5], [y0, y0], "k-", lw=1.2)
        ax1.plot([0.5, 9.5], [y1, y1], "k-", lw=1.2)
        ax1.text(0.0, (y0 + y1) / 2, f"Block {b}", fontsize=9, va="center")
        ax1.add_patch(FancyBboxPatch((2, y1 - 0.15), 6, 1.0,
                                       boxstyle="round,pad=0.05",
                                       facecolor="lightyellow", edgecolor="black"))
        ax1.text(5, (y0 + y1) / 2, f"$U_{{block}}(x_{{3b}}, x_{{3b+1}}, x_{{3b+2}})$",
                 ha="center", va="center", fontsize=10)

    ax1.text(5, 0.5,
             r"$K^B(x,x') = \frac{1}{B}\sum_b K_{block}(x_b, x'_b)$,    "
             r"$\mathrm{Var}[K] = \frac{8\pi^4}{45 B} c^4 + O(c^6)$",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    out = FIG / "fig0_architecture.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- fig8: 5-dataset MCC headline ----------

def fig8_headline_5datasets():
    zz = json.loads((TAB / "zz_baseline.json").read_text())
    rbf = json.loads((TAB / "rbf_baseline.json").read_text())

    keys = {"wdbc": "wdbc", "parkinsons": "parkinsons", "heart": "heart",
            "pneumonia": "pneumonia_mnist", "breast": "breast_mnist"}
    cov_single = {
        "wdbc": folds("E6_wdbc_balanced.log"),
        "parkinsons": folds("E6_parkinsons_c0.5.log"),
        "heart": folds("E6_heart_balanced.log"),
        "pneumonia": folds("E6_pneumonia.log"),
        "breast": folds("E6_breast.log"),
    }
    cov_pair = {
        "wdbc": folds("E6_wdbc_pair_n200.log"),
        "parkinsons": folds("E6_parkinsons_pair_n200.log"),
        "heart": folds("E6_heart_pair_n200.log"),
        "pneumonia": folds("E6_pneumonia_pair_n200.log"),
        "breast": folds("E6_breast_pair_n200.log"),
    }
    cpmap = {
        "wdbc": folds("E10_wdbc.log"),
        "parkinsons": folds("E10_parkinsons.log"),
        "heart": folds("E10_heart.log"),
        "pneumonia": folds("E10_pneumonia.log"),
        "breast": folds("E10_breast.log"),
    }

    datasets = ["wdbc", "parkinsons", "heart", "pneumonia", "breast"]
    methods = ["ZZ", "CovNCG-single", "CovNCG-pair", "CPMap", "RBF"]
    palette = ["#888", "#1f78b4", "#0d4a6d", "#33a02c", "#6a3d9a"]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(datasets))
    w = 0.16

    means = {
        "ZZ": [np.mean(zz[keys[d]]["mcc"]) for d in datasets],
        "CovNCG-single": [np.mean(cov_single[d]) for d in datasets],
        "CovNCG-pair": [np.mean(cov_pair[d]) for d in datasets],
        "CPMap": [np.mean(cpmap[d]) for d in datasets],
        "RBF": [np.mean(rbf[keys[d]]["mcc"]) for d in datasets],
    }
    stds = {
        "ZZ": [np.std(zz[keys[d]]["mcc"]) for d in datasets],
        "CovNCG-single": [np.std(cov_single[d]) for d in datasets],
        "CovNCG-pair": [np.std(cov_pair[d]) for d in datasets],
        "CPMap": [np.std(cpmap[d]) for d in datasets],
        "RBF": [np.std(rbf[keys[d]]["mcc"]) for d in datasets],
    }
    for i, m in enumerate(methods):
        ax.bar(x + (i - 2) * w, means[m], w, yerr=stds[m],
               label=m, color=palette[i], capsize=3, edgecolor="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("MCC (5-fold CV)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("CovNCG vs baselines across 5 medical datasets")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = FIG / "fig8_headline_5datasets.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- fig9: pair-proj κ = 7/32 verification ----------

def fig9_pair_proj_verify():
    Bs = np.array([1, 2, 3, 4])
    c = 0.05
    A_single = 8 * np.pi ** 4 / 45
    A_pair = 7 * np.pi ** 4 / 180
    pred_single = A_single * c ** 4 / Bs
    pred_pair = A_pair * c ** 4 / Bs

    emp_single = np.array([1.013e-4, 5.054e-5, 3.460e-5, 2.567e-5])
    emp_pair = np.array([2.257e-5, 1.178e-5, 7.677e-6, 5.668e-6])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    ax0.loglog(Bs, emp_single, "o", color="#1f78b4", markersize=10, label="single-proj empirical")
    ax0.loglog(Bs, pred_single, "-", color="#1f78b4", lw=2, label=r"single-proj: $8\pi^4/(45B) c^4$")
    ax0.loglog(Bs, emp_pair, "s", color="#0d4a6d", markersize=10, label="pair-proj empirical")
    ax0.loglog(Bs, pred_pair, "--", color="#0d4a6d", lw=2, label=r"pair-proj: $7\pi^4/(180B) c^4$")
    ax0.set_xlabel("# disjoint blocks B")
    ax0.set_ylabel(f"Var[K]  (c = {c})")
    ax0.set_title("Multi-block 1/B scaling — both kernels")
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3, which="both")
    ax0.set_xticks([1, 2, 3, 4])
    ax0.set_xticklabels(["1", "2", "3", "4"])

    ratio_emp = emp_pair / emp_single
    ratio_theory = (A_pair / A_single) * np.ones_like(Bs)
    ax1.plot(Bs, ratio_emp, "o-", color="#e31a1c", lw=2, markersize=10,
             label=f"empirical ratio = {ratio_emp.mean():.3f}")
    ax1.axhline(7 / 32, color="black", ls="--", lw=2, label=r"theory: $\kappa_{pair} = 7/32 = 0.21875$")
    ax1.set_xlabel("# disjoint blocks B")
    ax1.set_ylabel(r"Var[K_pair] / Var[K_proj]")
    ax1.set_title("κ_pair verification (constant in B)")
    ax1.set_ylim(0.18, 0.26)
    ax1.set_xticks([1, 2, 3, 4])
    ax1.legend()
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG / "fig9_pair_proj_kappa.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ---------- fig10: saturation curve Var vs c ----------

def fig10_saturation():
    e12 = json.loads((TAB / "E12_theorem_sanity.json").read_text())
    cs = np.array([r["c"] for r in e12["rows"]])
    var = np.array([r["var"] for r in e12["rows"]])

    A = 8 * np.pi ** 4 / 45
    pred = A * cs ** 4

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(cs, var, "o-", color="#1f78b4", lw=2, markersize=10, label="empirical Var[K_proj]")
    ax.loglog(cs, pred, "--", color="black", lw=2, label=r"theory leading $\frac{8\pi^4}{45} c^4$")
    ax.axhline(0.25, color="red", ls=":", lw=1.5, alpha=0.7, label="upper bound = 1/4 (binary)")
    ax.fill_betweenx([1e-8, 1e1], 0.04, 0.20, alpha=0.15, color="green", label=r"$c^4$ regime ($c \leq 0.20$)")
    ax.fill_betweenx([1e-8, 1e1], 0.20, 0.50, alpha=0.15, color="orange", label="saturation onset")
    ax.fill_betweenx([1e-8, 1e1], 0.50, 2.5, alpha=0.15, color="red", label="saturated")

    ax.set_xlim(0.04, 2.5)
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("bandwidth c")
    ax.set_ylabel("Var[K_proj]")
    ax.set_title("c⁴-regime, saturation onset, and 1/4 upper bound (single block)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = FIG / "fig10_saturation.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    fig0_circuit()
    fig8_headline_5datasets()
    fig9_pair_proj_verify()
    fig10_saturation()


if __name__ == "__main__":
    main()

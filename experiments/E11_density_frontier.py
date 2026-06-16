"""E11: density frontier — MCC vs features/qubit across methods.

Extracts feat/q ratio from existing baselines + CovNCG + CPMap results.
1.0 feat/q = ZZFeatureMap (4 features, 4 qubits)
1.5 feat/q = CovNCG (12 features, 8 qubits, group_size=3)
2.0 feat/q = CPMap (12 features, 6 qubits, 4 features/pair)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(p): return json.loads((TAB / p).read_text())


def main() -> None:
    points = {
        "wdbc": [],
        "parkinsons": [],
        "heart": [],
    }

    zz = load("zz_baseline.json")
    for ds, key in [("wdbc", "wdbc"), ("parkinsons", "parkinsons"), ("heart", "heart")]:
        m = zz[key]["mcc"]
        points[ds].append(("ZZFeatureMap", 1.0, 4, np.mean(m), np.std(m)))

    covncg_files = {
        "wdbc": "covncg_wdbc_c1.0_reps1.json",
        "parkinsons": "covncg_parkinsons_c0.5_reps1.json",
        "heart": "covncg_heart_cleveland_c0.5_reps1.json",
    }
    for ds, f in covncg_files.items():
        m = load(f)["folds"]["mcc"]
        points[ds].append(("CovNCG", 1.5, 8, np.mean(m), np.std(m)))

    cpmap_files = {
        "wdbc": "cpmap_wdbc_c1.0_q6.json",
        "parkinsons": "cpmap_parkinsons_c1.0_q6.json",
        "heart": "cpmap_heart_cleveland_c1.0_q6.json",
    }
    for ds, f in cpmap_files.items():
        m = load(f)["folds"]["mcc"]
        points[ds].append(("CPMap", 2.0, 6, np.mean(m), np.std(m)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {"wdbc": "#1f78b4", "parkinsons": "#33a02c", "heart": "#e31a1c"}
    markers = {"ZZFeatureMap": "o", "CovNCG": "s", "CPMap": "D"}

    for ds, pts in points.items():
        xs = [p[1] for p in pts]
        ys = [p[3] for p in pts]
        errs = [p[4] for p in pts]
        ax.errorbar(xs, ys, yerr=errs, color=palette[ds], lw=2, capsize=4,
                    marker="", linestyle="-", alpha=0.6, label=f"{ds} (line)")
        for method, x, q, y, e in pts:
            ax.errorbar([x], [y], yerr=[e], color=palette[ds],
                        marker=markers[method], markersize=11, capsize=4,
                        markeredgecolor="black", markeredgewidth=0.7)
            ax.annotate(f"{method}\n(q={q})", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7.5, color=palette[ds])

    ax.set_xlabel("features per qubit (density)")
    ax.set_ylabel("MCC (5-fold CV)")
    ax.set_title("Density frontier: MCC vs features/qubit across 3 medical datasets")
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.grid(alpha=0.3)
    ax.legend(title="dataset trajectory", loc="lower right")
    plt.tight_layout()
    out = FIG / "fig5_density_frontier.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    print("\n=== density frontier table ===")
    print(f"{'dataset':12} {'method':14} {'feat/q':>7} {'qubits':>7} {'MCC':>8} {'std':>8}")
    for ds, pts in points.items():
        for method, x, q, y, e in pts:
            print(f"{ds:12} {method:14} {x:>7.2f} {q:>7d} {y:>8.4f} {e:>8.4f}")


if __name__ == "__main__":
    main()

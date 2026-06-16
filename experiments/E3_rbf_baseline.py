"""E3: tuned classical RBF-SVM 5-fold CV across all medical datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.baselines import cv_rbf
from src.data import LOADERS, load

TAB_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(LOADERS),
                   choices=list(LOADERS))
    args = p.parse_args()

    summary = {}
    for name in args.datasets:
        d = load(name)
        print(f"\n===== RBF-SVM on {d.name} =====")
        res = cv_rbf(d.X, d.y, n_splits=5)
        summary[name] = {
            "mcc": [float(v) for v in res["mcc"]],
            "f1":  [float(v) for v in res["f1"]],
            "auc": [float(v) for v in res["auc"]],
            "mcc_mean": float(np.mean(res["mcc"])),
            "mcc_std":  float(np.std(res["mcc"])),
        }
        print(f"  {name}: MCC={summary[name]['mcc_mean']:.4f}±{summary[name]['mcc_std']:.4f}")

    out = TAB_DIR / "rbf_baseline.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

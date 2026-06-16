"""E2: ZZFeatureMap QSVM 5-fold CV across all medical datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.baselines import cv_zz
from src.data import LOADERS, load

TAB_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(LOADERS),
                   choices=list(LOADERS))
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--n-train", type=int, default=100)
    args = p.parse_args()

    summary = {}
    for name in args.datasets:
        d = load(name)
        print(f"\n===== ZZFeatureMap on {d.name} =====")
        res = cv_zz(d.X, d.y, n_qubits=args.n_qubits, reps=args.reps,
                    n_train_sub=args.n_train, n_splits=5)
        summary[name] = {
            "mcc": [float(v) for v in res["mcc"]],
            "f1":  [float(v) for v in res["f1"]],
            "auc": [float(v) for v in res["auc"]],
            "mcc_mean": float(np.mean(res["mcc"])),
            "mcc_std":  float(np.std(res["mcc"])),
        }
        print(f"  {name}: MCC={summary[name]['mcc_mean']:.4f}±{summary[name]['mcc_std']:.4f}")

    out = TAB_DIR / "zz_baseline.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

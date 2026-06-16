"""E6: CovNCG full 5-fold CV per dataset at chosen c*. Primary result.

Heavy. Statevector cost O(N² · 2^n_qubits). Run per dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LOADERS, load
from src.evaluation import cv_covncg, summary_row

TAB_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(LOADERS))
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--n-feat", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kernel", default="projected", choices=["projected", "pair_projected"])
    p.add_argument("--selector", default="mi", choices=["mi", "kta"])
    p.add_argument("--adaptive-c", action="store_true")
    args = p.parse_args()

    d = load(args.dataset)
    print(f"===== CovNCG on {d.name} =====")
    print(d.summary())
    res = cv_covncg(
        d.X, d.y,
        c=args.c, reps=args.reps,
        group_size=args.group_size,
        n_feat=args.n_feat,
        n_train_sub=args.n_train, n_splits=5, seed=args.seed,
        kernel_name=args.kernel,
        selector=args.selector,
        adaptive_c=args.adaptive_c,
    )
    summary = summary_row(f"CovNCG_{d.name}_c{args.c}_reps{args.reps}", res)
    print(f"\n{summary}")

    out = TAB_DIR / f"covncg_{d.name}_c{args.c}_reps{args.reps}.json"
    with open(out, "w") as f:
        json.dump({
            "summary": summary,
            "folds": {k: [float(v) for v in vs] for k, vs in res.items() if k != "diagnostics"},
            "diagnostics": res["diagnostics"],
        }, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

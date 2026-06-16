"""E8: ablation — covariance vs random grouping. Proves rule matters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LOADERS, load
from src.evaluation import cv_covncg, summary_row
from src.grouping import group_greedy, group_random

TAB_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(LOADERS))
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--grouping", required=True, choices=["greedy", "random"])
    p.add_argument("--n-feat", type=int, default=12)
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fn = group_greedy if args.grouping == "greedy" else (lambda X, group_size=3: group_random(X, group_size, seed=args.seed))
    d = load(args.dataset)
    print(f"===== ablation {args.grouping} on {d.name} c={args.c} =====")
    res = cv_covncg(
        d.X, d.y, c=args.c, reps=1,
        group_fn=fn, group_size=3,
        n_feat=args.n_feat, n_train_sub=args.n_train,
        n_splits=5, seed=args.seed,
    )
    summary = summary_row(f"CovNCG_{d.name}_{args.grouping}_c{args.c}", res)
    print(f"\n{summary}")

    out = TAB_DIR / f"ablation_{d.name}_{args.grouping}_c{args.c}.json"
    with open(out, "w") as f:
        json.dump({
            "summary": summary,
            "folds": {k: [float(v) for v in vs] for k, vs in res.items() if k != "diagnostics"},
            "diagnostics": res["diagnostics"],
        }, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

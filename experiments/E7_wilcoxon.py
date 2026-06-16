"""E7: Wilcoxon signed-rank — CovNCG vs ZZ, RBF, random-grouping."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluation import significance_wilcoxon

TAB_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def load_json(name: str) -> dict:
    with open(TAB_DIR / name) as f:
        return json.load(f)


def per_dataset_folds(blob: dict, ds: str) -> list[float]:
    return [float(v) for v in blob[ds]["mcc"]]


def main() -> None:
    zz = load_json("zz_baseline.json")
    rbf = load_json("rbf_baseline.json")

    cov_wdbc = load_json("covncg_wdbc_c1.0_reps1.json")["folds"]["mcc"]
    cov_park = load_json("covncg_parkinsons_c0.5_reps1.json")["folds"]["mcc"]  # best c
    cov_heart = load_json("covncg_heart_cleveland_c0.5_reps1.json")["folds"]["mcc"]
    rand_wdbc = load_json("ablation_wdbc_random_c1.0.json")["folds"]["mcc"]

    cpmap_wdbc = load_json("cpmap_wdbc_c1.0_q6.json")["folds"]["mcc"]
    cpmap_park = load_json("cpmap_parkinsons_c1.0_q6.json")["folds"]["mcc"]
    cpmap_heart = load_json("cpmap_heart_cleveland_c1.0_q6.json")["folds"]["mcc"]

    print("=== Wilcoxon (one-sided, MCC) ===\n")
    print("CovNCG > ZZFeatureMap (floor):")
    significance_wilcoxon(cov_wdbc, per_dataset_folds(zz, "wdbc"), "CovNCG_wdbc", "ZZ_wdbc")
    significance_wilcoxon(cov_park, per_dataset_folds(zz, "parkinsons"), "CovNCG_park", "ZZ_park")
    significance_wilcoxon(cov_heart, per_dataset_folds(zz, "heart"), "CovNCG_heart", "ZZ_heart")

    print("\nCovNCG > tuned RBF-SVM (ceiling — expected to fail):")
    significance_wilcoxon(cov_wdbc, per_dataset_folds(rbf, "wdbc"), "CovNCG_wdbc", "RBF_wdbc")
    significance_wilcoxon(cov_park, per_dataset_folds(rbf, "parkinsons"), "CovNCG_park", "RBF_park")
    significance_wilcoxon(cov_heart, per_dataset_folds(rbf, "heart"), "CovNCG_heart", "RBF_heart")

    print("\nCovNCG vs CPMap (both projected kernel, both training-free):")
    significance_wilcoxon(cov_wdbc, cpmap_wdbc, "CovNCG_wdbc", "CPMap_wdbc")
    significance_wilcoxon(cpmap_wdbc, cov_wdbc, "CPMap_wdbc", "CovNCG_wdbc")
    significance_wilcoxon(cov_park, cpmap_park, "CovNCG_park", "CPMap_park")
    significance_wilcoxon(cpmap_park, cov_park, "CPMap_park", "CovNCG_park")
    significance_wilcoxon(cov_heart, cpmap_heart, "CovNCG_heart", "CPMap_heart")
    significance_wilcoxon(cpmap_heart, cov_heart, "CPMap_heart", "CovNCG_heart")

    print("\nAblation — covariance grouping > random grouping (WDBC, seed=0):")
    significance_wilcoxon(cov_wdbc, rand_wdbc, "Cov_wdbc", "Rand_wdbc")

    print("\n=== summary (mean ± std MCC) ===")
    def row(name, zz_s, cov_s, cpmap_s, rbf_s):
        return (f"  {name:12s} ZZ {np.mean(zz_s):.4f}±{np.std(zz_s):.4f}  "
                f"CovNCG {np.mean(cov_s):.4f}±{np.std(cov_s):.4f}  "
                f"CPMap {np.mean(cpmap_s):.4f}±{np.std(cpmap_s):.4f}  "
                f"RBF {np.mean(rbf_s):.4f}±{np.std(rbf_s):.4f}")
    print(row("wdbc", per_dataset_folds(zz, "wdbc"), cov_wdbc, cpmap_wdbc, per_dataset_folds(rbf, "wdbc")))
    print(row("parkinsons", per_dataset_folds(zz, "parkinsons"), cov_park, cpmap_park, per_dataset_folds(rbf, "parkinsons")))
    print(row("heart", per_dataset_folds(zz, "heart"), cov_heart, cpmap_heart, per_dataset_folds(rbf, "heart")))


if __name__ == "__main__":
    main()

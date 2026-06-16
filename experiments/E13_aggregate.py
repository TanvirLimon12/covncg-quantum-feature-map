"""Aggregate 5 QPU runs + run simulator parity for each. Produce Table for paper."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stratified_subset(X, y, n, seed):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    k0 = n // 2
    sel0 = rng.choice(idx0, k0, replace=False)
    sel1 = rng.choice(idx1, n - k0, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return X[sel], y[sel]


def sim_baseline(n_train, n_test, c, seed, n_feat=12, reps=1):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score

    from src.data import load_wdbc
    from src.feature_select import top_k_mi
    from src.grouping import group_greedy
    from src.kernel import build_gram_matrix, projected_kernel_entry

    ds = load_wdbc()
    X_sub, y_sub = stratified_subset(ds.X, ds.y, n_train + n_test, seed)
    feats = top_k_mi(X_sub, y_sub, n_feat, seed=seed)
    X_sub = X_sub[:, feats]
    X_sub = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(X_sub)

    X_train, X_test = X_sub[:n_train], X_sub[n_train:]
    y_train, y_test = y_sub[:n_train], y_sub[n_train:]
    groups = group_greedy(X_sub, group_size=3)
    K_tr = build_gram_matrix(X_train, groups, c=c, reps=reps,
                             kernel_fn=projected_kernel_entry)
    K_te = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K_te[i, j] = projected_kernel_entry(X_test[i], X_train[j], groups, c=c, reps=reps)
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(K_tr, y_train)
    yp = clf.predict(K_te)
    try:
        auc = roc_auc_score(y_test, clf.decision_function(K_te))
    except Exception:
        auc = float("nan")
    return {
        "MCC": float(matthews_corrcoef(y_test, yp)),
        "Acc": float(accuracy_score(y_test, yp)),
        "AUC": float(auc),
        "K_diag": float(K_tr.diagonal().mean()),
        "K_off": float(K_tr[~np.eye(n_train, dtype=bool)].mean()),
    }


def main():
    runs = [
        ("seed=0",  "E13_qpu_wdbc.json",    {"n_train": 20, "n_test": 10, "c": 1.0, "seed": 0}),
        ("seed=1",  "E13_qpu_seed1.json",   {"n_train": 20, "n_test": 10, "c": 1.0, "seed": 1}),
        ("seed=2",  "E13_qpu_seed2.json",   {"n_train": 20, "n_test": 10, "c": 1.0, "seed": 2}),
        ("c=0.5",   "E13_qpu_c0p5.json",    {"n_train": 20, "n_test": 10, "c": 0.5, "seed": 0}),
        ("scaleup", "E13_qpu_scaleup.json", {"n_train": 40, "n_test": 20, "c": 1.0, "seed": 0}),
    ]
    rows = []
    for label, qpu_file, cfg in runs:
        with open(ROOT / "results" / qpu_file) as f:
            qpu = json.load(f)
        sim = sim_baseline(**cfg)
        rows.append({
            "label": label, "cfg": cfg,
            "QPU_MCC": qpu["MCC"], "QPU_Acc": qpu["Acc"], "QPU_AUC": qpu["AUC"],
            "QPU_K_diag": qpu["K_train_diag_mean"], "QPU_K_off": qpu["K_train_offdiag_mean"],
            "Sim_MCC": sim["MCC"], "Sim_Acc": sim["Acc"], "Sim_AUC": sim["AUC"],
            "Sim_K_diag": sim["K_diag"], "Sim_K_off": sim["K_off"],
            "job_id": qpu.get("job_id"),
        })

    print(f"\n{'Run':<10} {'n_tr':>4} {'n_te':>4} {'c':>5} {'seed':>4}  "
          f"{'QPU MCC':>8} {'Sim MCC':>8} {'ΔMCC':>7}  "
          f"{'QPU Acc':>8} {'Sim Acc':>8}  {'QPU K_d':>8} {'Sim K_d':>8}")
    for r in rows:
        cfg = r["cfg"]
        print(f"{r['label']:<10} {cfg['n_train']:>4} {cfg['n_test']:>4} "
              f"{cfg['c']:>5.2f} {cfg['seed']:>4}  "
              f"{r['QPU_MCC']:>8.4f} {r['Sim_MCC']:>8.4f} "
              f"{r['QPU_MCC'] - r['Sim_MCC']:>+7.4f}  "
              f"{r['QPU_Acc']:>8.4f} {r['Sim_Acc']:>8.4f}  "
              f"{r['QPU_K_diag']:>8.4f} {r['Sim_K_diag']:>8.4f}")

    seed_runs = [r for r in rows if r["label"].startswith("seed=")]
    mccs_qpu = [r["QPU_MCC"] for r in seed_runs]
    mccs_sim = [r["Sim_MCC"] for r in seed_runs]
    print(f"\n3-seed mean ± std (n=20+10, c=1.0):")
    print(f"  QPU: MCC = {np.mean(mccs_qpu):.4f} ± {np.std(mccs_qpu, ddof=1):.4f}")
    print(f"  Sim: MCC = {np.mean(mccs_sim):.4f} ± {np.std(mccs_sim, ddof=1):.4f}")

    out = ROOT / "results" / "E13_aggregate.json"
    with open(out, "w") as f:
        json.dump({
            "rows": rows,
            "seed_mean_QPU": float(np.mean(mccs_qpu)),
            "seed_std_QPU": float(np.std(mccs_qpu, ddof=1)),
            "seed_mean_Sim": float(np.mean(mccs_sim)),
            "seed_std_Sim": float(np.std(mccs_sim, ddof=1)),
        }, f, indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()

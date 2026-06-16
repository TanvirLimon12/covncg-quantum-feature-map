"""E10: CPMap baseline 5-fold CV per dataset (same protocol as E6, fair comparison)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.cpmap import build_cpmap_circuit
from src.data import LOADERS, load
from src.feature_select import top_k_mi

TAB_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def cpmap_projected_kernel_entry(x, xp, n_qubits, c, reps=1):
    qc  = build_cpmap_circuit(x,  n_qubits=n_qubits, n_features=len(x), c=c, reps=reps)
    qcp = build_cpmap_circuit(xp, n_qubits=n_qubits, n_features=len(xp), c=c, reps=reps)
    dm  = DensityMatrix(Statevector(qc))
    dmp = DensityMatrix(Statevector(qcp))
    fids = []
    for q in range(n_qubits):
        trace_out = [i for i in range(n_qubits) if i != q]
        rho  = partial_trace(dm, trace_out)
        rhop = partial_trace(dmp, trace_out)
        fids.append(float(np.real(np.trace(rho.data @ rhop.data))))
    return float(np.mean(fids))


def gram(X, n_qubits, c, reps=1, X_b=None):
    if X_b is None:
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k = cpmap_projected_kernel_entry(X[i], X[j], n_qubits, c, reps)
                K[i, j] = k; K[j, i] = k
        return K
    K = np.zeros((len(X), len(X_b)))
    for i in range(len(X)):
        for j in range(len(X_b)):
            K[i, j] = cpmap_projected_kernel_entry(X[i], X_b[j], n_qubits, c, reps)
    return K


def cv_cpmap(d, c, n_feat, n_train, n_qubits, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    mccs, f1s, aucs = [], [], []
    for fold, (tr, te) in enumerate(skf.split(d.X, d.y)):
        print(f"fold {fold+1}/5...")
        sc = MinMaxScaler(feature_range=(0, np.pi))
        X_tr_full = sc.fit_transform(d.X[tr])
        X_te_full = sc.transform(d.X[te])
        y_tr, y_te = d.y[tr], d.y[te]

        feat_idx = top_k_mi(X_tr_full, y_tr, k=n_feat, seed=seed)
        X_tr, X_te = X_tr_full[:, feat_idx], X_te_full[:, feat_idx]

        n_sub = min(n_train, len(X_tr))
        idx = rng.choice(len(X_tr), n_sub, replace=False)
        X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]

        K_tr = gram(X_tr_sub, n_qubits, c)
        K_te = gram(X_te, n_qubits, c, X_b=X_tr_sub)

        svm = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
        svm.fit(K_tr, y_tr_sub)
        y_pred = svm.predict(K_te)
        y_score = svm.decision_function(K_te)

        mccs.append(matthews_corrcoef(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred))
        aucs.append(roc_auc_score(y_te, y_score))
        print(f"  MCC={mccs[-1]:.4f} F1={f1s[-1]:.4f} AUC={aucs[-1]:.4f}")
    return {"mcc": mccs, "f1": f1s, "auc": aucs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(LOADERS))
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--n-feat", type=int, default=12)
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-qubits", type=int, default=6,
                   help="2 feat/qubit -> n_qubits = ceil(n_feat/2)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = load(args.dataset)
    print(f"===== CPMap on {d.name}  c={args.c} n_qubits={args.n_qubits} =====")
    res = cv_cpmap(d, args.c, args.n_feat, args.n_train, args.n_qubits, args.seed)
    summary = {
        "method": f"CPMap_{d.name}_c{args.c}_q{args.n_qubits}",
        "mcc_mean": float(np.mean(res["mcc"])),
        "mcc_std":  float(np.std(res["mcc"])),
        "f1_mean":  float(np.mean(res["f1"])),
        "f1_std":   float(np.std(res["f1"])),
        "auc_mean": float(np.mean(res["auc"])),
        "auc_std":  float(np.std(res["auc"])),
    }
    print(f"\n{summary}")
    out = TAB_DIR / f"cpmap_{d.name}_c{args.c}_q{args.n_qubits}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "folds": {k: [float(v) for v in vs] for k, vs in res.items()}}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

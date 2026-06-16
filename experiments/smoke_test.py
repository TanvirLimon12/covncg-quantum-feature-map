"""Cheap end-to-end smoke across all 3 medical datasets.

Each: 30 samples → 20 train / 10 test, 6 features (or all if <6), c=1.0.
Verifies grouping + circuit + projected Gram + SVM fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.data import load_all
from src.diagnostics import print_summary, summarise
from src.grouping import group_greedy, qubit_layout
from src.kernel import build_gram_matrix, build_test_gram, projected_kernel_entry


def smoke_one(d) -> bool:
    print(f"\n========== {d.name} ==========")
    print(d.summary())
    rng = np.random.default_rng(0)
    n = min(30, len(d.X))
    idx = rng.choice(len(d.X), n, replace=False)
    n_feat = min(6, d.X.shape[1])
    feat_idx = list(range(n_feat))
    X_sub = d.X[idx][:, feat_idx]
    y_sub = d.y[idx]

    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_sub)
    groups = group_greedy(X, group_size=3)
    n_qubits, _ = qubit_layout(groups)
    print(f"groups={groups}  n_qubits={n_qubits}")

    n_tr = 20
    X_tr, y_tr = X[:n_tr], y_sub[:n_tr]
    X_te, y_te = X[n_tr:], y_sub[n_tr:]

    K_tr = build_gram_matrix(X_tr, groups, c=1.0, kernel_fn=projected_kernel_entry)
    K_te = build_test_gram(X_te, X_tr, groups, c=1.0, kernel_fn=projected_kernel_entry)
    K_rbf = rbf_kernel(X_tr, gamma=0.5)
    diag = summarise(K_tr, y_tr, K_C_rbf=K_rbf)
    print_summary(diag, label=d.name)

    if diag["off_diag_var"] < 1e-12:
        print(f"FAIL: {d.name} Gram concentrated.")
        return False

    svm = SVC(kernel="precomputed", C=1.0)
    svm.fit(K_tr, y_tr)
    mcc = matthews_corrcoef(y_te, svm.predict(K_te))
    print(f"smoke MCC (n_test={len(y_te)}): {mcc:.4f}")
    return True


def main() -> int:
    results = [smoke_one(d) for d in load_all()]
    print("\n========== summary ==========")
    print(f"passed: {sum(results)}/{len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

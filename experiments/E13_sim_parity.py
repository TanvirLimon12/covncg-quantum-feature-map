"""Simulator parity for E13 QPU run. Same seed, same split, exact statevector kernel.

Outputs MCC, Acc, AUC + Bloch-vector diff vs QPU result for paper table.
"""
from __future__ import annotations

import argparse
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
    k1 = n - k0
    sel0 = rng.choice(idx0, k0, replace=False)
    sel1 = rng.choice(idx1, k1, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return X[sel], y[sel]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--n-feat", type=int, default=12)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qpu-result", default=str(ROOT / "results" / "E13_qpu_wdbc.json"))
    p.add_argument("--out", default=str(ROOT / "results" / "E13_sim_parity.json"))
    args = p.parse_args()

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score

    from src.data import load_wdbc
    from src.feature_select import top_k_mi
    from src.grouping import group_greedy
    from src.kernel import build_gram_matrix, projected_kernel_entry
    from src.circuit import build_cov_ncg_circuit
    from qiskit.quantum_info import Statevector, partial_trace

    ds = load_wdbc()
    X_full, y_full = ds.X, ds.y
    X_sub, y_sub = stratified_subset(X_full, y_full, args.n_train + args.n_test, args.seed)
    feats = top_k_mi(X_sub, y_sub, args.n_feat, seed=args.seed)
    X_sub = X_sub[:, feats]
    X_sub = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(X_sub)

    X_train = X_sub[:args.n_train]
    y_train = y_sub[:args.n_train]
    X_test = X_sub[args.n_train:]
    y_test = y_sub[args.n_train:]

    groups = group_greedy(X_sub, group_size=3)

    K_train = build_gram_matrix(X_train, groups, c=args.c, reps=args.reps,
                                kernel_fn=projected_kernel_entry)
    # test gram
    K_test = np.zeros((args.n_test, args.n_train))
    for i in range(args.n_test):
        for j in range(args.n_train):
            K_test[i, j] = projected_kernel_entry(
                X_test[i], X_train[j], groups, c=args.c, reps=args.reps,
            )

    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, clf.decision_function(K_test))
    except Exception:
        auc = float("nan")
    print(f"Simulator: MCC={mcc:.4f}  Acc={acc:.4f}  AUC={auc:.4f}")
    print(f"K_train diag mean={K_train.diagonal().mean():.4f}  "
          f"off-diag mean={K_train[~np.eye(args.n_train, dtype=bool)].mean():.4f}")

    # exact Bloch vectors for comparison
    n_qubits = None
    Bloch = []
    X_all = np.vstack([X_train, X_test])
    for x in X_all:
        qc = build_cov_ncg_circuit(x, groups, c=args.c, reps=args.reps)
        n_qubits = qc.num_qubits
        sv = Statevector(qc)
        m = np.zeros((n_qubits, 3))
        for q in range(n_qubits):
            trace_out = [i for i in range(n_qubits) if i != q]
            rho = partial_trace(sv, trace_out).data
            # m = (Tr[ρX], Tr[ρY], Tr[ρZ])
            X_op = np.array([[0, 1], [1, 0]], dtype=complex)
            Y_op = np.array([[0, -1j], [1j, 0]], dtype=complex)
            Z_op = np.array([[1, 0], [0, -1]], dtype=complex)
            m[q, 0] = np.real(np.trace(rho @ X_op))
            m[q, 1] = np.real(np.trace(rho @ Y_op))
            m[q, 2] = np.real(np.trace(rho @ Z_op))
        Bloch.append(m)
    Bloch = np.array(Bloch)

    # Compare to QPU Bloch
    if Path(args.qpu_result).exists():
        with open(args.qpu_result) as f:
            qpu = json.load(f)
        M_qpu = np.array(qpu["Bloch_train"] + qpu["Bloch_test"])
        diff = Bloch - M_qpu
        rms = float(np.sqrt(np.mean(diff ** 2)))
        max_err = float(np.max(np.abs(diff)))
        print(f"Bloch RMS err sim vs QPU: {rms:.4f}  max: {max_err:.4f}")
        print(f"QPU MCC: {qpu['MCC']:.4f}  Sim MCC: {mcc:.4f}  Δ: {mcc - qpu['MCC']:+.4f}")
    else:
        rms = max_err = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "MCC_sim": float(mcc),
            "Acc_sim": float(acc),
            "AUC_sim": float(auc),
            "K_train_diag_mean": float(K_train.diagonal().mean()),
            "K_train_offdiag_mean": float(K_train[~np.eye(args.n_train, dtype=bool)].mean()),
            "Bloch_RMS_vs_QPU": rms,
            "Bloch_max_err_vs_QPU": max_err,
        }, f, indent=2)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()

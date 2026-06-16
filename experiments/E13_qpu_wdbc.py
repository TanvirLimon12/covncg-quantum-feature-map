"""E13: IBM QPU run on WDBC. Sampling-based projected CovNCG kernel.

Pipeline:
  1. Load WDBC, MI-top-12 features, MinMax→[0, π]
  2. Subset: n_train + n_test points (stratified)
  3. Greedy correlation grouping → 8-qubit CovNCG circuits
  4. For each x: 3 basis circuits (X, Y, Z) → SamplerV2 → counts
  5. Bloch vectors per qubit per data point
  6. Build train/test Gram from Bloch dot products
  7. SVC(precomputed) → MCC, accuracy, AUC
  8. Save results + parity with simulator

Usage:
  python experiments/E13_qpu_wdbc.py --n-train 20 --n-test 10 --shots 4096
  python experiments/E13_qpu_wdbc.py --dry-run   # no submission, prints circuits
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_token() -> str:
    with open(ROOT / "apikey.json") as f:
        return json.load(f)["apikey"]


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
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default=None, help="default = least busy ≥ 8q")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--optimization-level", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "E13_qpu_wdbc.json"))
    args = p.parse_args()

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score

    from src.data import load_wdbc
    from src.feature_select import top_k_mi
    from src.grouping import group_greedy
    from src.projected_kernel_sampler import (
        build_basis_circuits, bloch_vectors_from_counts, gram_from_bloch,
    )

    # ---- Data prep ----
    ds = load_wdbc()
    X_full, y_full = ds.X, ds.y
    n_total = args.n_train + args.n_test
    X_sub, y_sub = stratified_subset(X_full, y_full, n_total, args.seed)

    # MI on subset to pick top-k features
    feats = top_k_mi(X_sub, y_sub, args.n_feat, seed=args.seed)
    X_sub = X_sub[:, feats]

    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_sub = scaler.fit_transform(X_sub)

    X_train = X_sub[:args.n_train]
    y_train = y_sub[:args.n_train]
    X_test = X_sub[args.n_train:]
    y_test = y_sub[args.n_train:]

    groups = group_greedy(X_sub, group_size=3)
    print(f"groups: {groups}")
    print(f"train labels: {np.bincount(y_train).tolist()}  test labels: {np.bincount(y_test).tolist()}")

    # ---- Build circuits ----
    X_all = np.vstack([X_train, X_test])
    circuits, n_qubits = build_basis_circuits(X_all, groups, c=args.c, reps=args.reps)
    print(f"n_qubits={n_qubits}  n_circuits={len(circuits)} (3 bases × {len(X_all)} pts)")
    print(f"circuit depth (pre-transpile): {circuits[0].depth()}  size: {circuits[0].size()}")

    if args.dry_run:
        print("DRY RUN — printing first circuit:")
        print(circuits[0])
        return

    # ---- Connect to IBM ----
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit import transpile

    token = load_token()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
    print(f"backend: {backend.name}  qubits={backend.num_qubits}  "
          f"queue={getattr(backend.status(), 'pending_jobs', 'n/a')}")

    # ---- Transpile ----
    t0 = time.time()
    isa_circuits = transpile(circuits, backend=backend, optimization_level=args.optimization_level)
    print(f"transpile: {time.time() - t0:.1f}s  isa depth: {isa_circuits[0].depth()}")

    # ---- Submit ----
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = args.shots

    print(f"submitting {len(isa_circuits)} circuits, shots={args.shots} ...")
    t0 = time.time()
    job = sampler.run(isa_circuits)
    print(f"job id: {job.job_id()}")
    result = job.result()
    print(f"completed in {time.time() - t0:.1f}s")

    # ---- Extract counts ----
    all_counts = []
    for r in result:
        # SamplerV2: r.data.meas.get_counts()
        data = r.data
        # cregs created by measure_all() are typically called "meas"
        creg_name = list(data.keys())[0] if hasattr(data, "keys") else "meas"
        if hasattr(data, "meas"):
            counts = data.meas.get_counts()
        else:
            counts = getattr(data, creg_name).get_counts()
        all_counts.append(counts)

    # ---- Reshape to per-data Bloch ----
    M = np.zeros((len(X_all), n_qubits, 3))
    for i in range(len(X_all)):
        counts_xyz = all_counts[3 * i:3 * (i + 1)]
        M[i] = bloch_vectors_from_counts(counts_xyz, n_qubits)
    M_train = M[:args.n_train]
    M_test = M[args.n_train:]

    # ---- Build Gram + SVC ----
    K_train = gram_from_bloch(M_train)
    K_test = gram_from_bloch(M_test, M_train)
    print(f"K_train shape={K_train.shape}  diag mean={K_train.diagonal().mean():.4f}  "
          f"off-diag mean={K_train[~np.eye(args.n_train, dtype=bool)].mean():.4f}")

    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    try:
        y_score = clf.decision_function(K_test)
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")
    print(f"QPU result: MCC={mcc:.4f}  Acc={acc:.4f}  AUC={auc:.4f}")

    # ---- Save ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "backend": backend.name,
            "job_id": job.job_id(),
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_qubits": n_qubits,
            "shots": args.shots,
            "c": args.c,
            "reps": args.reps,
            "seed": args.seed,
            "groups": groups,
            "feats_idx": [int(i) for i in feats],
            "MCC": float(mcc),
            "Acc": float(acc),
            "AUC": float(auc),
            "K_train_diag_mean": float(K_train.diagonal().mean()),
            "K_train_offdiag_mean": float(K_train[~np.eye(args.n_train, dtype=bool)].mean()),
            "Bloch_train": M_train.tolist(),
            "Bloch_test": M_test.tolist(),
        }, f, indent=2)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()

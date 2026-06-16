"""Quantum + classical baselines for comparison.

  - cv_zz:  ZZFeatureMap (4 qubits, reps=2) global fidelity kernel + SVC.
            Defines the quantum floor (~0.4-0.6 MCC across datasets).
  - cv_rbf: Grid-searched RBF-SVM (C × γ on training fold).
            Defines the classical ceiling. Beats every quantum kernel evaluated.

The RBF wins consistently — predicted by the Egginger 2025 / Slattery 2023
structural wall. Reported honestly in the paper; not a defect of CovNCG.
"""
from __future__ import annotations

import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def zz_fidelity_kernel(X_a: np.ndarray, X_b: np.ndarray, n_qubits: int, reps: int = 2) -> np.ndarray:
    """Standard fidelity QSVM with ZZFeatureMap. Uses first n_qubits features."""
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    K = np.zeros((len(X_a), len(X_b)))

    def state(x):
        bound = fm.assign_parameters(dict(zip(fm.parameters, x[:n_qubits])))
        return Statevector(bound)

    states_b = [state(x) for x in X_b]
    for i, x in enumerate(X_a):
        sa = state(x)
        for j, sb in enumerate(states_b):
            K[i, j] = float(np.abs(sa.inner(sb)) ** 2)
    return K


def rbf_grid_search(X_train: np.ndarray, y_train: np.ndarray, cv: int = 3) -> SVC:
    grid = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1.0, "scale", "auto"]}
    inner = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    gs = GridSearchCV(SVC(kernel="rbf"), grid, cv=inner, scoring="matthews_corrcoef", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs


def cv_rbf(X_raw: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    mccs, f1s, aucs = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        sc = MinMaxScaler(feature_range=(0, np.pi))
        X_tr = sc.fit_transform(X_raw[tr])
        X_te = sc.transform(X_raw[te])
        gs = rbf_grid_search(X_tr, y[tr])
        y_pred = gs.predict(X_te)
        y_prob = gs.decision_function(X_te)
        mccs.append(matthews_corrcoef(y[te], y_pred))
        f1s.append(f1_score(y[te], y_pred))
        aucs.append(roc_auc_score(y[te], y_prob))
        print(f"  fold {fold+1}: MCC={mccs[-1]:.4f} F1={f1s[-1]:.4f} AUC={aucs[-1]:.4f} best={gs.best_params_}")
    return {"mcc": mccs, "f1": f1s, "auc": aucs}


def cv_zz(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_qubits: int = 4,
    reps: int = 2,
    n_train_sub: int = 100,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    mccs, f1s, aucs = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        sc = MinMaxScaler(feature_range=(0, np.pi))
        X_tr = sc.fit_transform(X_raw[tr])
        X_te = sc.transform(X_raw[te])
        y_tr, y_te = y[tr], y[te]

        n_sub = min(n_train_sub, len(X_tr))
        idx = rng.choice(len(X_tr), n_sub, replace=False)
        X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]

        K_tr = zz_fidelity_kernel(X_tr_sub, X_tr_sub, n_qubits=n_qubits, reps=reps)
        K_te = zz_fidelity_kernel(X_te, X_tr_sub, n_qubits=n_qubits, reps=reps)

        svm = SVC(kernel="precomputed", C=1.0, probability=False)
        svm.fit(K_tr, y_tr_sub)
        y_pred = svm.predict(K_te)
        y_score = svm.decision_function(K_te)

        mccs.append(matthews_corrcoef(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred))
        aucs.append(roc_auc_score(y_te, y_score))
        print(f"  fold {fold+1}: MCC={mccs[-1]:.4f} F1={f1s[-1]:.4f} AUC={aucs[-1]:.4f}")
    return {"mcc": mccs, "f1": f1s, "auc": aucs}

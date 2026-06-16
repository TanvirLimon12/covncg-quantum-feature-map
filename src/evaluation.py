"""Leakage-free 5-fold stratified CV harness for CovNCG.

Each fold:
  1. Fit MinMaxScaler on training fold only (transform both).
  2. Select top-k features via mutual information (or KTA).
  3. Subsample training to n_train_sub points for statevector tractability.
  4. Build training + test Grams via chosen kernel.
  5. Fit SVC(kernel='precomputed', C=1, class_weight='balanced').
  6. Report MCC, F1, AUC-ROC + Wilcoxon significance.

CRITICAL: scaler + feature selector fit on training fold only — no leakage.

Supports both projected and pair-projected kernels (KERNELS dict). Default to
'projected'; pair-projected variant lifts MCC +0.025 to +0.110 on 4 of 5
datasets (see FINAL_REPORT §5.2).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from .diagnostics import summarise
from .feature_select import select as _select_features
from .grouping import group_greedy
from .kernel import (
    build_gram_matrix,
    build_test_gram,
    pair_projected_kernel_entry,
    projected_kernel_entry,
)


KERNELS = {
    "projected": projected_kernel_entry,
    "pair_projected": pair_projected_kernel_entry,
}


def cv_covncg(
    X_raw: np.ndarray,
    y: np.ndarray,
    c: float = 0.5,
    reps: int = 1,
    group_fn=group_greedy,
    group_size: int = 3,
    n_feat: int | None = None,
    n_train_sub: int = 100,
    n_splits: int = 5,
    seed: int = 42,
    kernel_name: str = "projected",
    selector: str = "mi",
    adaptive_c: bool = False,
    collect_diagnostics: bool = True,
    class_weight: str | None = "balanced",
) -> dict:
    kernel_fn = KERNELS[kernel_name]
    """5-fold CV with strict leakage-free protocol:

    For each fold:
      1. fit MinMaxScaler on train only
      2. compute covariance grouping on train only
      3. subsample train (statevector cost), build Gram
      4. fit SVC(precomputed), score test
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    mccs, f1s, aucs, diags = [], [], [], []

    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        print(f"fold {fold+1}/{n_splits}...")
        sc = MinMaxScaler(feature_range=(0, np.pi))
        X_tr_full = sc.fit_transform(X_raw[tr])
        X_te_full = sc.transform(X_raw[te])
        y_tr, y_te = y[tr], y[te]

        if n_feat is not None and n_feat < X_tr_full.shape[1]:
            feat_idx = _select_features(selector, X_tr_full, y_tr, k=n_feat, seed=seed)
            print(f"  top-{n_feat} {selector.upper()} features: {feat_idx}")
            X_tr = X_tr_full[:, feat_idx]
            X_te = X_te_full[:, feat_idx]
        else:
            X_tr, X_te = X_tr_full, X_te_full

        if group_fn is group_greedy:
            groups = group_fn(X_tr, group_size=group_size)
        else:
            groups = group_fn(X_tr)
        print(f"  groups: {groups}")

        n_sub = min(n_train_sub, len(X_tr))
        idx = rng.choice(len(X_tr), n_sub, replace=False)
        X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]

        print(f"  train Gram {n_sub}x{n_sub}...")
        K_tr = build_gram_matrix(X_tr_sub, groups, c=c, reps=reps, kernel_fn=kernel_fn,
                                 adaptive_c=adaptive_c)
        print(f"  test Gram {len(X_te)}x{n_sub}...")
        K_te = build_test_gram(X_te, X_tr_sub, groups, c=c, reps=reps, kernel_fn=kernel_fn,
                               adaptive_c=adaptive_c)

        svm = SVC(kernel="precomputed", C=1.0, class_weight=class_weight)
        svm.fit(K_tr, y_tr_sub)
        y_pred = svm.predict(K_te)
        y_score = svm.decision_function(K_te)

        mccs.append(matthews_corrcoef(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred))
        aucs.append(roc_auc_score(y_te, y_score))
        print(f"  MCC={mccs[-1]:.4f} F1={f1s[-1]:.4f} AUC={aucs[-1]:.4f}")

        if collect_diagnostics:
            gamma = 1.0 / (2 * c * c)
            K_rbf = rbf_kernel(X_tr_sub, gamma=gamma)
            diags.append(summarise(K_tr, y_tr_sub, K_C_rbf=K_rbf))

    return {"mcc": mccs, "f1": f1s, "auc": aucs, "diagnostics": diags}


def significance_wilcoxon(a: list[float], b: list[float], label_a: str, label_b: str) -> float:
    """One-sided Wilcoxon: H1 a > b. Paired fold scores."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    stat, p = wilcoxon(a_arr, b_arr, alternative="greater")
    verdict = "significant" if p < 0.05 else "n.s."
    print(f"  {label_a} > {label_b}: W={stat:.2f} p={p:.4f} ({verdict})")
    return float(p)


def summary_row(name: str, res: dict) -> dict:
    return {
        "method": name,
        "mcc_mean": float(np.mean(res["mcc"])),
        "mcc_std": float(np.std(res["mcc"])),
        "f1_mean": float(np.mean(res["f1"])),
        "f1_std": float(np.std(res["f1"])),
        "auc_mean": float(np.mean(res["auc"])),
        "auc_std": float(np.std(res["auc"])),
    }

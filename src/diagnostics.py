"""Kernel diagnostics for CovNCG.

Five quantities per Gram:
  - off_diagonal_gram_variance:  Thanasilp 2024 concentration indicator
  - effective_dimension:         entropy of eigenvalue distribution
  - condition_number:            λ_max / λ_min (guard against ill-conditioned g)
  - kernel_target_alignment:     Cristianini 2001 alignment with yy^T
  - geometric_difference:        Huang 2021 g(K_C, K_Q); large g ↛ advantage if cond is high
  - frobenius_distance_to_rbf:   structural distance to RBF kernel

cond ≈ 1e13 across all medical datasets in our evaluation — note in any
quantum-advantage claim that draws on g (Gap 5).
"""
from __future__ import annotations

import numpy as np


def off_diagonal_gram_variance(K: np.ndarray) -> float:
    """Thanasilp 2024 concentration diagnostic. Want >> 0."""
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.var(K[mask]))


def off_diagonal_mean(K: np.ndarray) -> float:
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(K[mask]))


def effective_dimension(K: np.ndarray) -> float:
    """Entropy-based participation ratio of eigenvalues."""
    eig = np.linalg.eigvalsh(K)
    eig = np.maximum(eig, 0.0)
    s = eig.sum()
    if s == 0:
        return 0.0
    p = eig / s
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def condition_number(K: np.ndarray, reg: float = 1e-12) -> float:
    eig = np.linalg.eigvalsh(K)
    eig = np.maximum(eig, reg)
    return float(eig[-1] / eig[0])


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """KTA = <K, yy^T>_F / (||K||_F ||yy^T||_F). y ∈ {-1, +1}."""
    y_pm = np.where(y > 0, 1.0, -1.0)
    yyt = np.outer(y_pm, y_pm)
    num = float(np.sum(K * yyt))
    den = float(np.linalg.norm(K, "fro") * np.linalg.norm(yyt, "fro"))
    return num / den if den > 0 else 0.0


def geometric_difference(K_Q: np.ndarray, K_C: np.ndarray, reg: float = 1e-3) -> float:
    """g(K_C, K_Q) = sqrt(||K_Q^{1/2} K_C^{-1} K_Q^{1/2}||_F). Huang 2021.

    Large g = necessary condition for quantum advantage. Check condition # to rule
    out spurious g from ill-conditioning.
    """
    n = K_Q.shape[0]
    K_Q_reg = K_Q + reg * np.eye(n)
    K_C_reg = K_C + reg * np.eye(n)

    eig, vec = np.linalg.eigh(K_Q_reg)
    eig = np.maximum(eig, 0.0)
    K_Q_sqrt = vec @ np.diag(np.sqrt(eig)) @ vec.T

    K_C_inv = np.linalg.inv(K_C_reg)
    M = K_Q_sqrt @ K_C_inv @ K_Q_sqrt
    return float(np.sqrt(np.linalg.norm(M, "fro")))


def frobenius_distance_to_rbf(K_Q: np.ndarray, K_C_rbf: np.ndarray) -> float:
    """Trace-normalised Frobenius distance — proxy for geometric difference."""
    def norm(K):
        tr = np.trace(K)
        return K / tr if tr > 0 else K
    return float(np.linalg.norm(norm(K_Q) - norm(K_C_rbf), "fro"))


def summarise(K_Q: np.ndarray, y: np.ndarray, K_C_rbf: np.ndarray | None = None) -> dict:
    out = {
        "off_diag_var": off_diagonal_gram_variance(K_Q),
        "off_diag_mean": off_diagonal_mean(K_Q),
        "eff_dim": effective_dimension(K_Q),
        "cond": condition_number(K_Q),
        "kta": kernel_target_alignment(K_Q, y),
    }
    if K_C_rbf is not None:
        out["g"] = geometric_difference(K_Q, K_C_rbf)
        out["rbf_frob"] = frobenius_distance_to_rbf(K_Q, K_C_rbf)
    return out


def print_summary(d: dict, label: str = "") -> None:
    print(f"--- diagnostics: {label} ---")
    print(f"  off_diag_var:  {d['off_diag_var']:.6e}  (want >> 0)")
    print(f"  off_diag_mean: {d['off_diag_mean']:.6f}")
    print(f"  eff_dim:       {d['eff_dim']:.3f}")
    print(f"  cond:          {d['cond']:.3e}  (>> 1e6 → suspect)")
    print(f"  kta:           {d['kta']:.4f}")
    if "g" in d:
        print(f"  g:             {d['g']:.4f}  (> 1 necessary for advantage)")
        print(f"  rbf_frob:      {d['rbf_frob']:.4f}")

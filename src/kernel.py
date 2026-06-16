"""Projected quantum kernels for CovNCG.

Two variants:
  - projected_kernel_entry:      single-qubit reduced density inner products
  - pair_projected_kernel_entry: 2-qubit reduced density inner products

Closed-form variance (Theorems 10, 11 in docs/):
  Var[K_proj^B(x, x')] = (8π⁴ / (45 B)) c⁴ + O(c⁶)        single-projected
  Var[K_pair^B(x, x')] = (7π⁴ / (180 B)) c⁴ + O(c⁶)       pair-projected, κ = 7/32

Pair-projected wins 4/5 medical datasets despite lower variance: it captures
intra-pair entanglement that single-projection averages away. See FINAL_REPORT
§5.2 + docs/theorem_pair_projected.md.

Also implements global_fidelity_entry as a sanity baseline; not used in production.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from .circuit import build_cov_ncg_circuit


def _density_matrix(x: np.ndarray, groups: list[list[int]], c: float, reps: int,
                    adaptive_c: bool = False) -> DensityMatrix:
    qc = build_cov_ncg_circuit(x, groups, c=c, reps=reps, adaptive_c=adaptive_c)
    return DensityMatrix(Statevector(qc))


def projected_kernel_entry(
    x: np.ndarray,
    xp: np.ndarray,
    groups: list[list[int]],
    c: float = 1.0,
    reps: int = 1,
    adaptive_c: bool = False,
) -> float:
    """K_proj(x,x') = (1/n) Σ_q Tr[ρ_q(x) ρ_q(x')]."""
    dm = _density_matrix(x, groups, c, reps, adaptive_c=adaptive_c)
    dmp = _density_matrix(xp, groups, c, reps, adaptive_c=adaptive_c)
    n_qubits = int(np.log2(dm.dim))

    fids = []
    for q in range(n_qubits):
        trace_out = [i for i in range(n_qubits) if i != q]
        rho = partial_trace(dm, trace_out)
        rhop = partial_trace(dmp, trace_out)
        fids.append(float(np.real(np.trace(rho.data @ rhop.data))))
    return float(np.mean(fids))


def pair_projected_kernel_entry(
    x: np.ndarray,
    xp: np.ndarray,
    groups: list[list[int]],
    c: float = 1.0,
    reps: int = 1,
    adaptive_c: bool = False,
) -> float:
    """Pair-projected kernel: K = (1/B) Σ_b Tr[ρ_{2b,2b+1}(x) ρ_{2b,2b+1}(x')].

    Captures intra-pair entanglement that single-qubit projection discards.
    Trades anti-concentration robustness for more expressive subsystem.
    """
    dm = _density_matrix(x, groups, c, reps, adaptive_c=adaptive_c)
    dmp = _density_matrix(xp, groups, c, reps, adaptive_c=adaptive_c)
    n_qubits = int(np.log2(dm.dim))

    fids = []
    # iterate qubit pairs (0,1), (2,3), ...
    for p in range(0, n_qubits - 1, 2):
        keep = [p, p + 1]
        trace_out = [i for i in range(n_qubits) if i not in keep]
        if trace_out:
            rho = partial_trace(dm, trace_out)
            rhop = partial_trace(dmp, trace_out)
        else:
            rho, rhop = dm, dmp
        fids.append(float(np.real(np.trace(rho.data @ rhop.data))))
    # handle odd-qubit-count tail with single-qubit projection
    if n_qubits % 2 == 1:
        q = n_qubits - 1
        trace_out = list(range(n_qubits - 1))
        rho = partial_trace(dm, trace_out)
        rhop = partial_trace(dmp, trace_out)
        fids.append(float(np.real(np.trace(rho.data @ rhop.data))))
    return float(np.mean(fids))


def global_fidelity_entry(
    x: np.ndarray,
    xp: np.ndarray,
    groups: list[list[int]],
    c: float = 1.0,
    reps: int = 1,
) -> float:
    """Standard fidelity kernel |<φ(x)|φ(x')>|² — for comparison vs projected."""
    qc = build_cov_ncg_circuit(x, groups, c=c, reps=reps)
    qcp = build_cov_ncg_circuit(xp, groups, c=c, reps=reps)
    sv = Statevector(qc)
    svp = Statevector(qcp)
    return float(np.abs(sv.inner(svp)) ** 2)


def build_gram_matrix(
    X: np.ndarray,
    groups: list[list[int]],
    c: float = 1.0,
    reps: int = 1,
    kernel_fn=projected_kernel_entry,
    adaptive_c: bool = False,
) -> np.ndarray:
    """Symmetric N×N Gram. O(N² · 2^n_qubits) — caller subsamples."""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k = kernel_fn(X[i], X[j], groups, c=c, reps=reps, adaptive_c=adaptive_c)
            K[i, j] = k
            K[j, i] = k
    return K


def build_test_gram(
    X_test: np.ndarray,
    X_train: np.ndarray,
    groups: list[list[int]],
    c: float = 1.0,
    reps: int = 1,
    kernel_fn=projected_kernel_entry,
    adaptive_c: bool = False,
) -> np.ndarray:
    n_test, n_train = len(X_test), len(X_train)
    K = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K[i, j] = kernel_fn(X_test[i], X_train[j], groups, c=c, reps=reps, adaptive_c=adaptive_c)
    return K

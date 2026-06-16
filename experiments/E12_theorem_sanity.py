"""E12: numerical verification of CONJECTURED concentration lower bound.

Theorem (see THEORY_NOTE §6):
  For projected kernel K_proj on n qubits with 3 pairwise non-commuting generators
  G_1, G_2, G_3 inside one exponent at bandwidth c:
    Var[K_proj(x, x')] ≥ A · c^4 · ε^2 − O(c^6)
  where ε^2 = min_q Σ_{i<j} ||π_q([G_i, G_j])||_F^2.

This script:
  (1) Computes ε^2 exactly for (IX, XI, ZZ) on 2 qubits via partial-trace.
  (2) Samples N=5000 random (x, x') pairs uniform in [0, π]^3.
  (3) Computes empirical Var[K_proj] for c in {0.05, ..., 1.0}.
  (4) Fits A from empirical Var = A c^4 ε^2 + B c^6.
  (5) Reports + plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from src.circuit import non_commuting_block
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def pauli(name: str) -> np.ndarray:
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    table = {"I": I, "X": X, "Y": Y, "Z": Z}
    mats = [table[c] for c in name]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def epsilon_squared() -> float:
    """ε² REVISED — sum of squared Frobenius norms of NON-ZERO commutators.

    Discovered (2026-06-01): [IX, XI] = 0 (disjoint-qubit Paulis commute). And
    partial_trace of pure single-Pauli tensor products vanishes (Pauli traces = 0).
    So the partial-trace formulation in THEORY_NOTE §6 draft is wrong.

    The quantity that survives partial trace + drives cross-frequency variance is
    Σ_{i<j} ||[G_i, G_j]||²_F  (full commutator Frobenius norm, no projection).

    For (IX, XI, ZZ):
      [IX, ZZ] = -2i Z⊗Y → ||[IX,ZZ]||²_F = 16
      [XI, ZZ] = -2i Y⊗Z → ||[XI,ZZ]||²_F = 16
      [IX, XI] = 0
      Σ = 32
    """
    IX = pauli("IX"); XI = pauli("XI"); ZZ = pauli("ZZ")
    pairs = [(IX, XI), (IX, ZZ), (XI, ZZ)]
    total = 0.0
    for A, B in pairs:
        C = A @ B - B @ A
        total += float(np.linalg.norm(C, "fro") ** 2)
    return total


def build_qc(x: np.ndarray, c: float) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    non_commuting_block(qc, [0, 1], [0, 1, 2], x, c=c)
    return qc


def projected_kernel(x: np.ndarray, xp: np.ndarray, c: float) -> float:
    dm = DensityMatrix(Statevector(build_qc(x, c)))
    dmp = DensityMatrix(Statevector(build_qc(xp, c)))
    s = 0.0
    for q in range(2):
        trace_out = [i for i in range(2) if i != q]
        rho = partial_trace(dm, trace_out)
        rhop = partial_trace(dmp, trace_out)
        s += float(np.real(np.trace(rho.data @ rhop.data)))
    return s / 2.0


def main() -> None:
    eps2 = epsilon_squared()
    print(f"ε² (min over qubits) = {eps2:.6f}")

    rng = np.random.default_rng(0)
    N = 5000
    cs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    rows = []
    for c in cs:
        ks = np.empty(N)
        for i in range(N):
            x = rng.uniform(0, np.pi, 3)
            xp = rng.uniform(0, np.pi, 3)
            ks[i] = projected_kernel(x, xp, c)
        var = float(np.var(ks))
        mean = float(np.mean(ks))
        rows.append({"c": c, "var": var, "mean": mean})
        print(f"  c={c:.3f}  Var[K]={var:.6e}  mean={mean:.4f}  c^4 ε²={c**4 * eps2:.6e}")

    cs_arr = np.array([r["c"] for r in rows])
    var_arr = np.array([r["var"] for r in rows])

    # Fit Var = A c^4 eps^2 + B c^6
    # Linear in A, B
    M = np.stack([cs_arr ** 4 * eps2, cs_arr ** 6], axis=1)
    coef, *_ = np.linalg.lstsq(M, var_arr, rcond=None)
    A_fit, B_fit = coef
    print(f"\nFit: Var ≈ ({A_fit:.4f}) c^4 ε² + ({B_fit:.4f}) c^6")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(cs_arr, var_arr, "o-", label="empirical Var[K_proj]")
    ax.loglog(cs_arr, A_fit * cs_arr ** 4 * eps2, "--", label=f"{A_fit:.3f}·c⁴·ε²")
    ax.loglog(cs_arr, A_fit * cs_arr ** 4 * eps2 + B_fit * cs_arr ** 6, ":", label=f"fit (c⁴+c⁶)")
    ax.set_xlabel("bandwidth c")
    ax.set_ylabel("Var[K_proj]")
    ax.set_title("Theorem sanity: empirical vs c⁴ε² lower bound (2-qubit (IX, XI, ZZ))")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = FIG / "fig6_theorem_sanity.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")

    import json
    with open(TAB / "E12_theorem_sanity.json", "w") as f:
        json.dump({
            "eps_squared": eps2,
            "A_fit": float(A_fit),
            "B_fit": float(B_fit),
            "rows": rows,
        }, f, indent=2)


if __name__ == "__main__":
    main()

"""Sampling-based projected kernel for QPU.

Per-qubit Bloch-vector estimation: for each data point x, run CovNCG circuit
3 times with measurement basis rotated to X, Y, Z. From counts, estimate
  m_q(x) = (<X>_q, <Y>_q, <Z>_q)
Then single-qubit reduced density:
  ρ_q(x) = (I + m_q(x) · σ) / 2
And projected-kernel entry:
  K(x, x') = (1/n_q) Σ_q Tr[ρ_q(x) ρ_q(x')]
           = (1/n_q) Σ_q (1 + m_q(x) · m_q(x')) / 2

Cost: 3 circuits per data point (constant in N). Gram built from N Bloch sets.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .circuit import build_cov_ncg_circuit


def make_basis_circuit(qc_state: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Append basis-rotation + measurement to a state-prep circuit."""
    qc = qc_state.copy()
    n = qc.num_qubits
    if basis == "X":
        for q in range(n):
            qc.h(q)
    elif basis == "Y":
        for q in range(n):
            qc.sdg(q)
            qc.h(q)
    elif basis == "Z":
        pass
    else:
        raise ValueError(f"basis must be X, Y, Z (got {basis!r})")
    qc.measure_all()
    return qc


def build_basis_circuits(
    X: np.ndarray,
    groups: list[list[int]],
    c: float = 1.0,
    reps: int = 1,
) -> tuple[list[QuantumCircuit], int]:
    """Return [3N circuits in order x0-X, x0-Y, x0-Z, x1-X, ...] and n_qubits."""
    circuits: list[QuantumCircuit] = []
    n_qubits = 0
    for x in X:
        qc_state = build_cov_ncg_circuit(x, groups, c=c, reps=reps)
        n_qubits = qc_state.num_qubits
        for basis in ("X", "Y", "Z"):
            circuits.append(make_basis_circuit(qc_state, basis))
    return circuits, n_qubits


def _expectation_from_counts(counts: dict, n_qubits: int) -> np.ndarray:
    """<Z>_q per qubit from bitstring counts (basis already rotated to Z).

    Qiskit bitstring order: bit[0] is qubit 0 = rightmost char.
    """
    shots = sum(counts.values())
    z = np.zeros(n_qubits)
    for bs, c in counts.items():
        bs_clean = bs.replace(" ", "")
        for q in range(n_qubits):
            bit = bs_clean[-(q + 1)]
            z[q] += (1 if bit == "0" else -1) * c
    return z / shots


def bloch_vectors_from_counts(
    counts_per_basis: list[dict],
    n_qubits: int,
) -> np.ndarray:
    """3 counts dicts (X, Y, Z basis order) → (n_qubits, 3) Bloch vector array."""
    m = np.zeros((n_qubits, 3))
    for k, basis in enumerate(("X", "Y", "Z")):
        m[:, k] = _expectation_from_counts(counts_per_basis[k], n_qubits)
    return m


def gram_from_bloch(M_a: np.ndarray, M_b: np.ndarray | None = None) -> np.ndarray:
    """Single-qubit projected Gram from Bloch vector arrays.

    M_a: (N_a, n_qubits, 3). Returns K[i, j] = (1/n_qubits) Σ_q (1 + m_q^i · m_q^j) / 2.
    """
    if M_b is None:
        M_b = M_a
    Na, nq, _ = M_a.shape
    Nb = M_b.shape[0]
    dots = np.einsum("aqd,bqd->abq", M_a, M_b)
    K = (1.0 + dots).mean(axis=2) / 2.0
    return K

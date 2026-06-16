"""CPMap baseline re-implementation (Singh, Laprade et al. arXiv:2507.03689).

Dense packing 2 features/qubit via separate sandwich blocks:
  U_pair(x_a, x_b, x_c, x_d) = (A1(x_a) ⊗ A2(x_b)) · N(α,β,γ) · (A3(x_c) ⊗ A4(x_d))
  A_i = single-qubit rotation, one feature each → 4 features per pair (2 feat/qubit)
  N   = exp(-i [α XX + β YY + γ ZZ]) — commuting fixed entangler (not data-dependent)

Defaults α = β = γ = π/4 (no training).

Key structural difference vs CovNCG:
  CPMap:  4 features in separate single-qubit rotations + fixed entangler N
          → features do NOT co-inhabit one generator exponent
  CovNCG: 3 features INSIDE one non-commuting generator exp(-i [x_a IX + x_b XI + x_c ZZ])
          → BCH cross-frequencies

Our re-implementation reproduces MCC 0.866 ± 0.04 on WDBC vs the published
0.944; gap attributable to leakage-free CV protocol + n_train=120 vs the
~455 used in the original. CPMap wins on WDBC + Parkinsons; CovNCG wins on
Heart (weak-correlation regime — see FINAL_REPORT §5.7 + paper §heart-detail).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def _single_rot(qc: QuantumCircuit, qubit: int, theta: float, axis: str = "Y") -> None:
    if axis == "X":
        qc.rx(theta, qubit)
    elif axis == "Y":
        qc.ry(theta, qubit)
    else:
        qc.rz(theta, qubit)


def n_gate(qc: QuantumCircuit, qubits: tuple[int, int],
           alpha: float = np.pi / 4, beta: float = np.pi / 4, gamma: float = np.pi / 4) -> None:
    """Fixed XX+YY+ZZ entangler N(α,β,γ). Three Pauli generators all commute pairwise."""
    q0, q1 = qubits
    # exp(-i α XX) = H⊗H · CX · RZ(2α) on q1 · CX · H⊗H ... but standard impl: rxx/ryy/rzz
    qc.rxx(2.0 * alpha, q0, q1)
    qc.ryy(2.0 * beta, q0, q1)
    qc.rzz(2.0 * gamma, q0, q1)


def cpmap_pair_block(
    qc: QuantumCircuit,
    qubits: tuple[int, int],
    features: list[int],
    x: np.ndarray,
    c: float = 1.0,
    alpha: float = np.pi / 4, beta: float = np.pi / 4, gamma: float = np.pi / 4,
    axis: str = "Y",
) -> None:
    """One CPMap pair-block: 4 features → 2 qubits via (A⊗A) · N · (A⊗A)."""
    q0, q1 = qubits
    xa = x[features[0]]
    xb = x[features[1]] if len(features) > 1 else 0.0
    xc = x[features[2]] if len(features) > 2 else 0.0
    xd = x[features[3]] if len(features) > 3 else 0.0

    qc.h(q0); qc.h(q1)

    _single_rot(qc, q0, 2.0 * c * xa, axis=axis)
    _single_rot(qc, q1, 2.0 * c * xb, axis=axis)

    n_gate(qc, (q0, q1), alpha=alpha, beta=beta, gamma=gamma)

    _single_rot(qc, q0, 2.0 * c * xc, axis=axis)
    _single_rot(qc, q1, 2.0 * c * xd, axis=axis)


def build_cpmap_circuit(
    x: np.ndarray,
    n_qubits: int,
    n_features: int | None = None,
    c: float = 1.0,
    reps: int = 1,
) -> QuantumCircuit:
    """CPMap circuit. Packs 4 features per qubit-pair (2 feat/qubit).

    Pairs qubits sequentially: (0,1), (2,3), ...
    Encodes features in order. Pads with zeros if features < 2 * n_qubits.
    """
    if n_features is None:
        n_features = len(x)
    qc = QuantumCircuit(n_qubits)
    feat_per_pair = 4
    feat_per_qubit = 2

    if n_qubits % 2 == 1:
        n_qubits -= 1  # last qubit unused for pair encoding

    feat_idx = 0
    for rep in range(reps):
        for p in range(0, n_qubits, 2):
            feats = list(range(feat_idx, min(feat_idx + feat_per_pair, n_features)))
            while len(feats) < feat_per_pair:
                feats.append(feats[-1] if feats else 0)
            cpmap_pair_block(qc, (p, p + 1), feats, x, c=c)
            feat_idx = (feat_idx + feat_per_pair) % max(n_features, 1)
        if rep < reps - 1:
            qc.barrier()
    return qc

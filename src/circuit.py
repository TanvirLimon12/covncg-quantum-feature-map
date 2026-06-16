"""CovNCG circuit: non-commuting multi-feature generator blocks.

Each non-commuting block packs 3 features into 2 qubits via the unitary
  U(x_a, x_b, x_c) = exp(-i c [x_a IX + x_b XI + x_c ZZ])
implemented via the Trotter-factorized circuit
  (H ⊗ H) → R_X(2cx_a)_{q1} → R_X(2cx_b)_{q0} → CX → R_Z(2cx_c)_{q1} → CX

ANTI-FINDING: reps=2 with c=1.0 crashes accuracy (over-concentration). Use
reps=1 at c ∈ {0.5, 1.0}, OR reps=2 with c=0.25 (saturation prediction
confirmed empirically — see THEORY_NOTE §6 and FINAL_REPORT §5.3).

ANTI-FINDING: adaptive_c (c_b = c / σ_b per group) HURTS accuracy on WDBC
(0.833 → 0.514). Default to fixed c.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .grouping import qubit_layout


def non_commuting_block(
    qc: QuantumCircuit,
    qubits: list[int],
    features: list[int],
    x: np.ndarray,
    c: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> None:
    """3 features → 2 qubits via (IX, XI, ZZ) non-commuting generators.

    Trotter step approximating exp(-i [α x_a IX + β x_b XI + γ x_c ZZ]):
        RX on q1   (G1 = IX)
        RX on q0   (G2 = XI)
        CX-RZ-CX   (G3 = ZZ)
    """
    q0, q1 = qubits
    xa = x[features[0]]
    xb = x[features[1]] if len(features) > 1 else 0.0
    xc = x[features[2]] if len(features) > 2 else 0.0

    qc.h(q0)
    qc.h(q1)

    qc.rx(2.0 * c * alpha * xa, q1)
    qc.rx(2.0 * c * beta * xb, q0)

    qc.cx(q0, q1)
    qc.rz(2.0 * c * gamma * xc, q1)
    qc.cx(q0, q1)


def pair_block(qc: QuantumCircuit, qubits: list[int], features: list[int], x: np.ndarray, c: float) -> None:
    q0, q1 = qubits
    qc.h(q0)
    qc.h(q1)
    qc.rx(2.0 * c * x[features[0]], q1)
    qc.rx(2.0 * c * x[features[1]], q0)


def single_feature_block(qc: QuantumCircuit, qubit: int, feature_idx: int, x: np.ndarray, c: float) -> None:
    qc.h(qubit)
    qc.rz(2.0 * c * x[feature_idx], qubit)


def build_cov_ncg_circuit(
    x: np.ndarray,
    groups: list[list[int]],
    c: float = 1.0,
    reps: int = 1,
    entangle_between: bool = False,
    adaptive_c: bool = False,
    c_per_group: list[float] | None = None,
) -> QuantumCircuit:
    """Build full CovNCG circuit from feature groups.

    Args:
      adaptive_c: if True, scale c_b per group b by inverse feature std → balanced
                  rotation magnitudes across groups regardless of input range.
                  Requires x to be scaled (e.g. MinMaxScaler to [0, π]).
      c_per_group: if given, use these per-group bandwidths directly (overrides adaptive_c).
    """
    n_qubits, layout = qubit_layout(groups)
    qc = QuantumCircuit(n_qubits)

    if c_per_group is None:
        if adaptive_c:
            c_per_group = []
            for grp in groups:
                vals = np.asarray([x[i] for i in grp])
                sigma = float(np.std(vals)) if len(vals) > 1 else 1.0
                c_per_group.append(c / max(sigma, 0.1))
        else:
            c_per_group = [c] * len(groups)

    for rep in range(reps):
        for (qubits, feats), c_b in zip(layout, c_per_group):
            if len(qubits) == 2 and len(feats) >= 2:
                if len(feats) >= 3:
                    non_commuting_block(qc, qubits, feats, x, c=c_b)
                else:
                    pair_block(qc, qubits, feats, x, c=c_b)
            else:
                single_feature_block(qc, qubits[0], feats[0], x, c=c_b)

        if entangle_between and n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

        if rep < reps - 1:
            qc.barrier()

    return qc

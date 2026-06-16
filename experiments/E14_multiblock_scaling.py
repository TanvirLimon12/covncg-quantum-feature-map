"""E14: empirical verification of multi-block theorem (§15 of proof).

Theorem prediction: Var[K_proj] = (8π⁴/(45B)) · c⁴ at small c.

Test B ∈ {1, 2, 3, 4} (i.e. 2, 4, 6, 8 qubits) at fixed c=0.05.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from src.circuit import non_commuting_block

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


def build_circuit(x: np.ndarray, B: int, c: float) -> QuantumCircuit:
    """B disjoint non-commuting blocks. Total 2B qubits, 3B features."""
    qc = QuantumCircuit(2 * B)
    for b in range(B):
        feats = [3 * b, 3 * b + 1, 3 * b + 2]
        non_commuting_block(qc, [2 * b, 2 * b + 1], feats, x, c=c)
    return qc


def kernel(x: np.ndarray, xp: np.ndarray, B: int, c: float) -> float:
    dm = DensityMatrix(Statevector(build_circuit(x, B, c)))
    dmp = DensityMatrix(Statevector(build_circuit(xp, B, c)))
    n_qubits = 2 * B
    s = 0.0
    for q in range(n_qubits):
        trace_out = [i for i in range(n_qubits) if i != q]
        rho = partial_trace(dm, trace_out)
        rhop = partial_trace(dmp, trace_out)
        s += float(np.real(np.trace(rho.data @ rhop.data)))
    return s / n_qubits


def main() -> None:
    c = 0.05
    N = 4000
    A_thm = 8 * np.pi ** 4 / 45
    rng = np.random.default_rng(0)
    rows = []
    for B in [1, 2, 3, 4]:
        n_feat = 3 * B
        ks = np.empty(N)
        for i in range(N):
            x = rng.uniform(0, np.pi, n_feat)
            xp = rng.uniform(0, np.pi, n_feat)
            ks[i] = kernel(x, xp, B, c)
        var = float(np.var(ks))
        pred = A_thm * c ** 4 / B
        rows.append({"B": B, "var": var, "pred": pred, "ratio": var / pred})
        print(f"B={B}  n_qubits={2*B}  Var={var:.4e}  pred=A·c⁴/B={pred:.4e}  ratio={var/pred:.3f}")

    with open(TAB / "E14_multiblock.json", "w") as f:
        json.dump({"c": c, "N": N, "A_thm": A_thm, "rows": rows}, f, indent=2)

    Bs = [r["B"] for r in rows]
    vars_ = [r["var"] for r in rows]
    preds = [r["pred"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(Bs, vars_, "o-", label="empirical Var[K_proj]", lw=2, markersize=8)
    ax.loglog(Bs, preds, "--", label="theory: 8π⁴/(45B) · c⁴", lw=2)
    ax.set_xlabel("# disjoint blocks B")
    ax.set_ylabel(f"Var[K_proj]  (c={c})")
    ax.set_title("Multi-block 1/B scaling (§15 theorem)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = FIG / "fig7_multiblock_scaling.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

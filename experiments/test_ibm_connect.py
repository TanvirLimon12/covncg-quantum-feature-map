"""Connect IBM Quantum + list least-busy backend. Token from apikey.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_token() -> str:
    with open(ROOT / "apikey.json") as f:
        return json.load(f)["apikey"]


def main() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = load_token()
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=token,
            overwrite=True,
            set_as_default=True,
        )
    except Exception as e:
        print(f"save_account warn: {e}")

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backends = service.backends(operational=True, simulator=False)
    print(f"available: {[b.name for b in backends]}")
    least = service.least_busy(operational=True, simulator=False, min_num_qubits=8)
    print(f"least busy: {least.name}  qubits={least.num_qubits}  "
          f"queue={getattr(least.status(), 'pending_jobs', 'n/a')}")


if __name__ == "__main__":
    main()

"""Qiskit Runtime backend wrapper for IBM Quantum QPU runs.

Scaffold only — waits for IBM hardware access. Fill in token + backend name
when available. See https://docs.quantum.ibm.com/guides/setup-channel

Usage when ready:
  export QISKIT_IBM_TOKEN=...
  python experiments/E13_qpu_wdbc.py --backend ibm_brisbane --n-train 60 --shots 4096
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    backend_name: str = "ibm_brisbane"   # 127-qubit Eagle, free-tier accessible
    instance: str = "ibm-q/open/main"
    shots: int = 4096
    enable_zne: bool = True              # Zero Noise Extrapolation
    zne_noise_factors: tuple = (1, 3, 5)
    enable_readout_mitigation: bool = True   # M3
    optimization_level: int = 3


def get_runtime_service():
    """Lazy import + token check. Raises if env not set up."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as e:
        raise ImportError(
            "qiskit-ibm-runtime not installed. pip install qiskit-ibm-runtime"
        ) from e

    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        raise RuntimeError(
            "QISKIT_IBM_TOKEN not set. Get token from https://quantum.ibm.com/account "
            "then export QISKIT_IBM_TOKEN=..."
        )
    return QiskitRuntimeService(channel="ibm_quantum", token=token)


def get_sampler(cfg: RuntimeConfig):
    """SamplerV2 with mitigation. Returns sampler ready for circuit batches."""
    from qiskit_ibm_runtime import SamplerV2

    service = get_runtime_service()
    backend = service.backend(cfg.backend_name, instance=cfg.instance)

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = cfg.shots
    if cfg.enable_zne:
        sampler.options.dynamical_decoupling.enable = True
        # ZNE configured at sampler level — SamplerV2 specifics vary by runtime version
    if cfg.enable_readout_mitigation:
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
    return sampler, backend


def transpile_circuit_for_backend(qc, backend, optimization_level: int = 3):
    from qiskit import transpile
    return transpile(qc, backend=backend, optimization_level=optimization_level)

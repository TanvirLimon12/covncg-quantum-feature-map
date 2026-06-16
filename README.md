# CovNCG — training-free dense quantum feature maps

Code for a training-free quantum kernel built from non-commuting multi-feature
generator blocks. Each block packs three features into two qubits via
`exp(-i c [x_a IX + x_b XI + x_c ZZ])`, which gives a kernel whose variance has a
closed form, `Var[K] = (kappa * 8*pi^4) / (45 * B) * c^4 + O(c^6)`, so you can tune
the bandwidth `c` to avoid exponential concentration instead of guessing.

Evaluated on five small medical datasets (WDBC, Parkinson's voice, Heart Cleveland,
PneumoniaMNIST, BreastMNIST) with a projected kernel + SVM. No training, no
architecture search.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Qiskit + scikit-learn. The QPU experiments (E13, `test_ibm_connect.py`) need an IBM
Quantum account; set `IBM_QUANTUM_TOKEN` in your environment or drop an `apikey.json`
in the project root (gitignored). Everything else runs on the statevector simulator.

## Running

```bash
python experiments/smoke_test.py        # quick end-to-end sanity check
python experiments/E6_full_cv.py        # full 5-dataset CV
python experiments/E12_theorem_sanity.py  # variance closed-form check
python experiments/E15_paper_figures.py   # regenerate figures/
```

Experiments are numbered in the order they were run; each writes JSON to `results/`
and PNGs to `figures/`.

## Layout

```
src/          circuit construction, kernels, grouping, baselines (ZZ, RBF, CPMap)
experiments/  E1..E16 — one script per experiment
results/      per-experiment JSON summaries
figures/      generated figures
```

## Notes

- `reps=2` with `c=1.0` over-concentrates the kernel and tanks accuracy — use `reps=1`
  or drop `c` to ~0.25. See the `ANTI-FINDING` notes in `src/circuit.py`.
- Covariance-based feature grouping turned out to be statistically indistinguishable
  from random grouping; the accuracy comes from the non-commuting block + projected
  kernel, not the grouping heuristic.
- No quantum advantage over a tuned classical RBF is claimed.

# Quantum layer selection (QAOA)

Q-GEOCompress uses a **real quantum algorithm** — QAOA (Quantum Approximate
Optimization Algorithm) — to decide **which layers to compress**. This document
explains the formulation, the implementation, and the honest scope of the claim.

> **Terminology.** This project separates two things that are often conflated:
>
> - **Quantum-inspired** — classical linear algebra borrowed from quantum
>   many-body physics (SVD low-rank / structural factorization). No qubits.
> - **Quantum** — an actual variational quantum circuit (QAOA) executed on a
>   state-vector **simulator** (`PennyLane default.qubit`). This is what this
>   document is about.

---

## 1. The decision as an optimization problem

After the structural probe (`compress_model.py --method structural-probe`), each
candidate layer `i` has:

- `param_gain_abs` (`g_i`) — parameters saved if this layer is factorized.
- `map50_drop` (`d_i`) — single-layer accuracy cost measured on the **select** split.

We want a subset `S` of layers that **maximizes total savings** while keeping the
**accuracy cost low** and the **number of replaced layers** near a budget `k`.

Introduce one binary variable `x_i ∈ {0, 1}` (compress layer `i` or not) and
minimize the energy:

```
E(x) = − α · Σ ĝ_i x_i        (reward parameter savings)
       + β · Σ d̂_i x_i        (penalize accuracy drop)
       + γ · (Σ x_i − k)²      (push the count toward k)
```

`ĝ_i` and `d̂_i` are min–max normalized so the three terms are comparable.
Weights default to `α = 1.0`, `β = 2.0`, `γ = 1.5`.

Because `x_i² = x_i` for binaries, expanding the cardinality term gives a
standard **QUBO** (Quadratic Unconstrained Binary Optimization):

```
E(x) = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j + const
```

This is a knapsack-flavored combinatorial problem — exactly the class QAOA
targets.

See [`src/qgeocompress/quantum/qubo.py`](../src/qgeocompress/quantum/qubo.py).

## 2. QUBO → Ising → quantum circuit

QAOA works on **Ising** Hamiltonians, so we map `x_i = (1 − Z_i) / 2`
(`Z_i ∈ {−1, +1}` are Pauli-Z eigenvalues):

```
H_cost = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + offset
```

The ground state (lowest energy) of `H_cost` encodes the optimal layer subset.

## 3. QAOA

QAOA prepares a `p`-layer variational state:

```
|ψ(γ, β)⟩ = Π_{l=1..p} e^{−i β_l H_mixer} e^{−i γ_l H_cost} · H^⊗n |0⟩
```

- `H_cost` — the problem Hamiltonian above.
- `H_mixer` — transverse-field mixer `Σ_i X_i`.
- `(γ, β)` — `2p` classical angles optimized by gradient descent to minimize
  `⟨ψ| H_cost |ψ⟩`.

We then read the output distribution, keep the most probable bitstrings, and
return the one with the **lowest true QUBO energy** ("quantum proposal +
classical post-selection"). This keeps the search genuinely QAOA-driven while
guaranteeing a valid, high-quality assignment.

See [`src/qgeocompress/quantum/qaoa_selection.py`](../src/qgeocompress/quantum/qaoa_selection.py).

## 4. Usage

```bash
# Requires the quantum extra:  pip install -e ".[quantum]"
python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --selection-strategy quantum-qaoa \
  --quantum-backend qaoa \
  --max-replaced-layers 5 \
  --sensitivity-file results/summaries/structural_layer_sensitivity_holdout_select.json \
  --bn-data-yaml "$HOLDOUT/dota128_holdout_train.yaml" \
  --eval-data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --run-label holdout_top5_qaoa
```

`--quantum-backend`:

| Value | Behavior |
| ----- | -------- |
| `auto` (default) | QAOA if PennyLane is installed, else exact/greedy classical |
| `qaoa` | Require PennyLane; raises if missing |
| `classical` | Exact brute-force (`n ≤ 22`) or greedy — used as reference |

Programmatic:

```python
from qgeocompress.quantum import select_layers_by_qaoa

layers, meta = select_layers_by_qaoa(probe_rows, max_layers=5,
                                     backend="qaoa", return_meta=True)
print(meta["solver"], meta["qubo_energy"])
```

## 5. Why this is a legitimate quantum use case

- The layer-selection problem is a genuine **NP-hard combinatorial** problem
  (weighted knapsack / cardinality-constrained max-gain).
- QUBO/Ising is the **native input format** for QAOA and quantum annealers
  (D-Wave), so the pipeline is hardware-portable in principle.
- The classical brute-force solver provides an **exact ground-truth** to measure
  QAOA's approximation quality on real probe data.

## 6. Honest limitations

- **Simulator, not hardware.** Runs on `default.qubit` (CPU state vector). No
  real QPU, no quantum speedup is claimed.
- **Small scale.** State-vector simulation is exponential in qubit count; keep
  candidate layers ≲ 20 for QAOA. Larger instances fall back to classical.
- **Heuristic.** QAOA is approximate. On the hold-out probe (8 layers, `k=5`) it
  reached ≈ 97% of the classical optimum energy — competitive but not exact.
- **No advantage claim.** For these sizes, classical brute-force is instant and
  optimal. The value here is **methodological**: a clean, portable QUBO encoding
  of the compression decision, ready for quantum-annealing hardware at scale.

## 7. Future work

- **Quantum annealing** — submit the same QUBO to D-Wave (`dimod` / Ocean SDK).
- **Warm-start QAOA** — seed angles from the greedy/pareto solution.
- **Hard budget constraint** — slack-variable encoding of `Σ d_i x_i ≤ B` instead
  of the soft linear penalty.
- **Tensor-network compression** — Matrix Product State (MPS/TT) factorization of
  conv weights as a second, genuinely quantum-inspired method.

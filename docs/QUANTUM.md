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

## 5. What the formulation can and cannot buy

**The selection QUBO is solvable by a sort.** Expanding `γ(Σx − k)²` puts the
*same* coupling `2γ` on every pair, so the quadratic part depends on `x` only
through `m = Σx`. For a fixed `m` the minimum takes the `m` smallest linear
coefficients, hence

```
min_x E(x) = min over m in [0, n] of  ( sum of m smallest c_i ) + γ(m − k)²
```

A sort and a prefix sum solve this exactly in `O(n log n)`
(`qubo.solve_uniform_coupling_exact`, verified against exhaustive search on 20
random instances in `tests/test_quantum_selection.py`). **No quantum method can
beat it** — not because QAOA is weak, but because the instance class is trivial.
Any gap a solver shows here is that solver failing to reach a free optimum.

**Heterogeneous couplings are what would make it non-trivial — and they were
measured.** Writing `δy_i` for the change in the network's output when only
layer `i` is factorized, the first-order composition is

```
ε²(S) = ‖Σ_{i∈S} δy_i‖² = Σ_i ‖δy_i‖² + 2 Σ_{i<j} ⟨δy_i, δy_j⟩
```

so the true objective is an Ising model whose couplings are the Gram matrix of
those perturbations. Measured directly as `J_ij = ε²({i,j}) − ε²({i}) − ε²({j})`
over the 16 highest-saving layers of YOLO11n-OBB (120 pairs, 3 probe images):

| quantity | value |
| -------- | ----- |
| linear terms, sum | 0.00250 |
| couplings, sum | −0.00064 (26% of the linear part) |
| couplings negative | 63% |
| max \|J\| / max linear | 0.22 |

The couplings are real and mostly negative — pairs of layers whose errors
partially cancel. But they do not move the answer: at k = 4, 6 and 8 the subset
minimizing the *coupled* objective is exactly the one the separable sort picks,
gain 0.0%. The linear part dominates the decision.

**Conclusion for this project.** Layer selection offers no quantum lever, at
either formulation. The honest quantum-inspired direction is not a solver for
this decision but a different compression *mechanism* — tensor-train / MPS
factorization, where SVD is the two-site case and the bond dimension is set by
an entanglement entropy rather than a hand-picked rank.

## 6. Why the QUBO encoding is still worth keeping

- The layer-selection problem is a genuine **NP-hard combinatorial** problem
  (weighted knapsack / cardinality-constrained max-gain).
- QUBO/Ising is the **native input format** for QAOA and quantum annealers
  (D-Wave), so the pipeline is hardware-portable in principle.
- The classical brute-force solver provides an **exact ground-truth** to measure
  QAOA's approximation quality on real probe data.

## 7. Honest limitations

- **Simulator, not hardware.** Runs on `default.qubit` (CPU state vector). No
  real QPU, no quantum speedup is claimed.
- **Small scale.** Cost grows with both the state vector (`2**n`) and the
  `~n²/2` `ZZ` terms the optimizer differentiates through. Measured on 4 CPU
  cores (2 QAOA layers, 40 gradient steps):

  | qubits | peak RSS | wall time |
  | -----: | -------: | --------: |
  | 12 | 187 MB | 16 s |
  | 14 | 276 MB | 23 s |
  | 16 | 692 MB | 59 s |
  | 18 | 2.7 GB | 210 s |
  | 20 | ~11 GB | OOM-killed |

  `select_layers_by_qaoa` therefore caps QAOA at **16 variables**: past that it
  keeps the 16 highest parameter-gain candidates and records
  `qaoa_prefiltered_from` in the run metadata. The full backbone+neck probe
  yields 28 candidates, so this prefilter is on by default.
- **Heuristic.** QAOA is approximate. On the hold-out probe (8 layers, `k=5`) it
  reached ≈ 97% of the classical optimum energy — competitive but not exact.
- **No advantage claim.** For these sizes, classical brute-force is instant and
  optimal. The value here is **methodological**: a clean, portable QUBO encoding
  of the compression decision, ready for quantum-annealing hardware at scale.

## 8. Future work

- **Quantum annealing** — submit the same QUBO to D-Wave (`dimod` / Ocean SDK).
- **Warm-start QAOA** — seed angles from the greedy/pareto solution.
- **Hard budget constraint** — slack-variable encoding of `Σ d_i x_i ≤ B` instead
  of the soft linear penalty.
- **Tensor-network compression** — Matrix Product State (MPS/TT) factorization of
  conv weights as a second, genuinely quantum-inspired method.

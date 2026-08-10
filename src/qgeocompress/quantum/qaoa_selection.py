"""Solve the layer-selection QUBO with QAOA (or a classical fallback).

QAOA runs on a CPU state-vector simulator (PennyLane ``default.qubit``).
Because the instances here are small (tens of candidate layers, and we
usually probe far fewer), we read the exact output distribution, keep the
most probable bitstrings, and return the one with the lowest true QUBO
energy. This "quantum proposal + classical post-selection" pattern makes
the result robust while keeping the search genuinely QAOA-driven.

If PennyLane is unavailable, ``select_layers_by_qaoa`` transparently falls
back to an exact brute-force solve (small ``n``) or a greedy solve.
"""

from __future__ import annotations

import itertools
from typing import Any

from qgeocompress.quantum.qubo import (
    QUBO,
    _param_gain_abs,
    build_selection_qubo,
    qubo_energy,
    qubo_to_ising,
)

# Above this many variables, skip exhaustive enumeration.
_BRUTEFORCE_LIMIT = 22

# Measured on 4 CPU cores, 2 QAOA layers, 40 gradient steps (peak RSS / wall time):
#   12 qubits 187 MB / 16 s | 14 → 276 MB / 23 s | 16 → 692 MB / 59 s | 18 → 2.7 GB / 210 s
# Cost is driven by autodiff through the ~n**2/2 ZZ terms, not by the state
# vector alone. 20 qubits needed ~11 GB here and was OOM-killed, so cap at 16.
_QAOA_QUBIT_LIMIT = 16


def is_quantum_available() -> bool:
    """Return True if PennyLane can be imported."""
    try:
        import pennylane  # noqa: F401

        return True
    except Exception:
        return False


def _bits_from_index(index: int, n: int) -> list[int]:
    return [(index >> (n - 1 - i)) & 1 for i in range(n)]


def solve_bruteforce(q: QUBO, n: int) -> list[int]:
    """Exact QUBO minimizer by enumeration (only for small ``n``)."""
    if n == 0:
        return []
    best_bits = [0] * n
    best_energy = float("inf")
    for combo in itertools.product((0, 1), repeat=n):
        energy = qubo_energy(q, combo)
        if energy < best_energy:
            best_energy = energy
            best_bits = list(combo)
    return best_bits


def _solve_greedy(q: QUBO, n: int) -> list[int]:
    """Greedy descent: flip the bit that most reduces energy until stable."""
    bits = [0] * n
    current = qubo_energy(q, bits)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            trial = list(bits)
            trial[i] ^= 1
            energy = qubo_energy(q, trial)
            if energy < current - 1e-12:
                bits, current = trial, energy
                improved = True
    return bits


def solve_qaoa(
    q: QUBO,
    n: int,
    *,
    layers: int = 2,
    steps: int = 40,
    stepsize: float = 0.3,
    seed: int = 42,
    top_candidates: int = 16,
) -> list[int]:
    """Approximate QUBO minimizer via QAOA on a CPU simulator.

    Raises ImportError if PennyLane is not installed.
    """
    import numpy as np
    import pennylane as qml
    from pennylane import numpy as pnp

    if n == 0:
        return []
    if n == 1:
        return [0] if q.get((0, 0), 0.0) >= 0 else [1]

    h, j, _offset = qubo_to_ising(q)

    coeffs: list[float] = []
    ops: list[Any] = []
    for i, coeff in h.items():
        if abs(coeff) > 1e-12:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(i))
    for (a, b), coeff in j.items():
        if abs(coeff) > 1e-12:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(a) @ qml.PauliZ(b))

    if not ops:
        # Degenerate cost (all zero) — nothing to optimize.
        return [0] * n

    cost_h = qml.Hamiltonian(coeffs, ops)
    mixer_h = qml.qaoa.x_mixer(range(n))
    dev = qml.device("default.qubit", wires=n)

    def qaoa_layer(gamma: float, beta: float) -> None:
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(beta, mixer_h)

    @qml.qnode(dev)
    def expectation(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, layers, params[0], params[1])
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def distribution(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, layers, params[0], params[1])
        return qml.probs(wires=range(n))

    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0.0, np.pi, size=(2, layers)), requires_grad=True)
    optimizer = qml.GradientDescentOptimizer(stepsize=stepsize)
    for _ in range(steps):
        params = optimizer.step(expectation, params)

    probs = np.asarray(distribution(params))
    ranked_indices = np.argsort(probs)[::-1][:top_candidates]

    best_bits = _bits_from_index(int(ranked_indices[0]), n)
    best_energy = qubo_energy(q, best_bits)
    for idx in ranked_indices[1:]:
        bits = _bits_from_index(int(idx), n)
        energy = qubo_energy(q, bits)
        if energy < best_energy:
            best_energy, best_bits = energy, bits
    return best_bits


def _solve(
    q: QUBO,
    n: int,
    *,
    backend: str,
    layers: int,
    steps: int,
    seed: int,
) -> tuple[list[int], str]:
    """Dispatch to the requested backend with graceful fallback."""
    want_quantum = backend in ("auto", "qaoa")
    if want_quantum and is_quantum_available():
        try:
            return solve_qaoa(q, n, layers=layers, steps=steps, seed=seed), "qaoa"
        except Exception:
            if backend == "qaoa":
                raise
    elif backend == "qaoa":
        raise ImportError("PennyLane is required for backend='qaoa' (pip install '.[quantum]')")

    if n <= _BRUTEFORCE_LIMIT:
        return solve_bruteforce(q, n), "bruteforce"
    return _solve_greedy(q, n), "greedy"


def select_layers_by_qaoa(
    probe_results: list[dict[str, Any]],
    max_layers: int,
    *,
    gain_weight: float = 1.0,
    drop_penalty: float = 2.0,
    cardinality_penalty: float = 1.5,
    backend: str = "auto",
    qaoa_layers: int = 2,
    qaoa_steps: int = 40,
    seed: int = 42,
    return_meta: bool = False,
) -> list[str] | tuple[list[str], dict[str, Any]]:
    """Select structural layers by minimizing the selection QUBO with QAOA.

    ``backend`` is one of ``"auto"`` (QAOA if available, else classical),
    ``"qaoa"`` (require PennyLane), or ``"classical"`` (brute-force/greedy).
    """
    rows = list(probe_results)
    prefiltered_from = None
    if backend != "classical" and len(rows) > _QAOA_QUBIT_LIMIT and is_quantum_available():
        # The simulator cannot hold 2**n amplitudes past ~20 qubits. Keep the
        # highest parameter-gain candidates so QAOA still decides the subset.
        prefiltered_from = len(rows)
        rows = sorted(rows, key=_param_gain_abs, reverse=True)[:_QAOA_QUBIT_LIMIT]

    q, layer_names, qubo_meta = build_selection_qubo(
        rows,
        max_layers,
        gain_weight=gain_weight,
        drop_penalty=drop_penalty,
        cardinality_penalty=cardinality_penalty,
    )
    n = len(layer_names)

    if n == 0:
        return ([], {"solver": "none", **qubo_meta}) if return_meta else []

    effective_backend = "classical" if backend == "classical" else backend
    if effective_backend == "classical":
        bits = solve_bruteforce(q, n) if n <= _BRUTEFORCE_LIMIT else _solve_greedy(q, n)
        solver = "bruteforce" if n <= _BRUTEFORCE_LIMIT else "greedy"
    else:
        bits, solver = _solve(
            q, n, backend=effective_backend, layers=qaoa_layers, steps=qaoa_steps, seed=seed
        )

    selected_idx = [i for i, b in enumerate(bits) if b == 1]

    # The cardinality term is soft; enforce the hard cap by keeping the
    # highest parameter-gain layers if QAOA over-selects.
    if len(selected_idx) > max_layers:
        selected_idx.sort(
            key=lambda i: qubo_meta["normalized_gains"][i], reverse=True
        )
        selected_idx = selected_idx[:max_layers]

    selected = [layer_names[i] for i in selected_idx]

    if return_meta:
        meta = {
            "solver": solver,
            "backend_requested": backend,
            "qaoa_prefiltered_from": prefiltered_from,
            "selected_bits": bits,
            "num_selected": len(selected),
            "qubo_energy": qubo_energy(q, bits),
            **qubo_meta,
        }
        return selected, meta
    return selected

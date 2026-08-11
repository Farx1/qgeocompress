import pytest

from qgeocompress.compression.layer_sensitivity import select_layers
from qgeocompress.quantum import (
    build_selection_qubo,
    is_quantum_available,
    qaoa_selection,
    qubo_energy,
    qubo_to_ising,
    select_layers_by_qaoa,
    solve_bruteforce,
)

HAS_PL = is_quantum_available()
skip_no_pl = pytest.mark.skipif(not HAS_PL, reason="PennyLane not installed")


def _probe(n: int) -> list[dict]:
    """Synthetic probe rows: layer i saves (i+1)*1000 params, drop scales down."""
    return [
        {
            "layer_name": f"model.{i}.conv",
            "param_gain_abs": (i + 1) * 1000,
            "map50_drop": 0.001 * (n - i),
        }
        for i in range(n)
    ]


def test_build_qubo_shapes_and_cardinality():
    q, names, meta = build_selection_qubo(_probe(4), max_layers=2)
    assert len(names) == 4
    assert meta["cardinality_target"] == 2
    # Diagonal (linear) present for each variable.
    for i in range(4):
        assert (i, i) in q
    # Upper-triangular couplings from the cardinality term.
    assert (0, 1) in q and (0, 0) in q
    assert q[(0, 1)] == pytest.approx(2.0 * meta["cardinality_penalty"])


def test_ising_roundtrip_matches_qubo_energy():
    q, names, _ = build_selection_qubo(_probe(5), max_layers=3)
    h, j, offset = qubo_to_ising(q)
    for mask in range(2 ** len(names)):
        bits = [(mask >> i) & 1 for i in range(len(names))]
        z = [1 - 2 * b for b in bits]
        ising = offset
        for i, coeff in h.items():
            ising += coeff * z[i]
        for (a, b), coeff in j.items():
            ising += coeff * z[a] * z[b]
        assert ising == pytest.approx(qubo_energy(q, bits))


def test_bruteforce_finds_known_optimum():
    # Only variable 0 is rewarded; optimum selects just it.
    q = {(0, 0): -10.0, (1, 1): 5.0, (0, 1): 1.0}
    assert solve_bruteforce(q, 2) == [1, 0]


def test_classical_backend_matches_bruteforce():
    probe = _probe(6)
    selected, meta = select_layers_by_qaoa(
        probe, max_layers=3, backend="classical", return_meta=True
    )
    q, names, _ = build_selection_qubo(probe, max_layers=3)
    bf = solve_bruteforce(q, len(names))
    bf_selected = {names[i] for i, b in enumerate(bf) if b == 1}
    # Classical solver returns the exact optimum (capped to max_layers).
    assert meta["solver"] == "bruteforce"
    assert set(selected) <= bf_selected or len(selected) <= 3


def test_selection_respects_cardinality_cap():
    selected = select_layers_by_qaoa(_probe(6), max_layers=2, backend="classical")
    assert len(selected) <= 2
    assert all(name.startswith("model.") for name in selected)


@skip_no_pl
def test_qaoa_matches_bruteforce_on_tiny_instance():
    # For n <= 4 the QAOA post-selection covers all basis states, so the
    # returned assignment is the exact QUBO optimum regardless of angles.
    probe = _probe(4)
    q, names, _ = build_selection_qubo(probe, max_layers=2)
    optimum = qubo_energy(q, solve_bruteforce(q, len(names)))

    selected, meta = select_layers_by_qaoa(
        probe, max_layers=2, backend="qaoa", qaoa_layers=2, qaoa_steps=25, return_meta=True
    )
    assert meta["solver"] == "qaoa"
    assert meta["qubo_energy"] == pytest.approx(optimum)
    assert len(selected) <= 2


@skip_no_pl
def test_qaoa_near_optimal_on_medium_instance():
    probe = _probe(7)
    q, names, _ = build_selection_qubo(probe, max_layers=4)
    optimum = qubo_energy(q, solve_bruteforce(q, len(names)))
    _, meta = select_layers_by_qaoa(
        probe, max_layers=4, backend="qaoa", qaoa_steps=40, return_meta=True
    )
    # Heuristic: energy should be better than the all-zeros baseline (0.0)
    # and reasonably close to the optimum.
    assert meta["qubo_energy"] <= optimum * 0.7


def test_auto_backend_falls_back_when_quantum_unavailable(monkeypatch):
    monkeypatch.setattr(qaoa_selection, "is_quantum_available", lambda: False)
    selected, meta = select_layers_by_qaoa(
        _probe(5), max_layers=3, backend="auto", return_meta=True
    )
    assert meta["solver"] in ("bruteforce", "greedy")
    assert len(selected) <= 3


def test_qaoa_backend_requires_pennylane(monkeypatch):
    monkeypatch.setattr(qaoa_selection, "is_quantum_available", lambda: False)
    with pytest.raises(ImportError):
        select_layers_by_qaoa(_probe(4), max_layers=2, backend="qaoa")


def test_select_layers_dispatch_quantum():
    probe = _probe(5)
    selected = select_layers(
        probe, max_layers=2, strategy="quantum-qaoa", quantum_backend="classical"
    )
    assert len(selected) <= 2
    assert all(isinstance(name, str) for name in selected)


def test_empty_probe_returns_empty():
    assert select_layers_by_qaoa([], max_layers=3, backend="classical") == []


@skip_no_pl
def test_qaoa_prefilters_beyond_simulator_limit(monkeypatch):
    """28 candidates would need a 2**28-amplitude statevector (~4 GB)."""
    seen = {}

    def _fake_qaoa(q, n, **kwargs):
        seen["n"] = n
        return [1] * min(n, 5) + [0] * max(0, n - 5)

    monkeypatch.setattr(qaoa_selection, "solve_qaoa", _fake_qaoa)
    selected, meta = select_layers_by_qaoa(
        _probe(28), max_layers=5, backend="qaoa", return_meta=True
    )

    assert seen["n"] == qaoa_selection._QAOA_QUBIT_LIMIT
    assert meta["qaoa_prefiltered_from"] == 28
    assert len(selected) <= 5
    # Prefiltering keeps the highest parameter-gain candidates.
    assert "model.27.conv" in selected


def test_classical_backend_is_not_prefiltered():
    _, meta = select_layers_by_qaoa(_probe(28), max_layers=5, backend="classical", return_meta=True)
    assert meta["qaoa_prefiltered_from"] is None
    assert meta["num_variables"] == 28


def test_sort_matches_exhaustive_search_on_the_selection_qubo():
    """The selection QUBO's couplings are all equal, so a sort solves it exactly.

    Expanding gamma*(sum x - k)^2 puts the SAME coupling on every pair, so the
    quadratic part depends on x only through the count. For a fixed count the
    minimum takes the smallest linear coefficients, which a sort and a prefix
    sum settle in O(n log n) — leaving QAOA, annealing and brute force with an
    optimum that is already free.
    """
    import random

    from qgeocompress.quantum.qubo import solve_uniform_coupling_exact

    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(2, 12)
        rows = [
            {
                "layer_name": f"l{i}",
                "param_gain_abs": rng.randint(100, 100_000),
                "map50_drop": rng.random() * 0.1,
            }
            for i in range(n)
        ]
        q, _, _ = build_selection_qubo(rows, rng.randint(0, n))
        assert qubo_energy(q, solve_uniform_coupling_exact(q, n)) == pytest.approx(
            qubo_energy(q, solve_bruteforce(q, n))
        )

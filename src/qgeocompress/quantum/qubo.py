"""Build a QUBO for structural-layer selection.

The compression decision is: pick a subset S of candidate layers to
replace with low-rank blocks, maximizing total parameter savings while
keeping the accuracy cost (single-layer mAP drop) low and the number of
replaced layers close to a target budget ``k``.

Encode one binary variable ``x_i in {0, 1}`` per candidate layer and
minimize the energy

    E(x) = - alpha * sum_i g_i * x_i          # reward parameter savings
           + beta  * sum_i d_i * x_i          # penalize accuracy drop
           + gamma * (sum_i x_i - k) ** 2      # push cardinality toward k

where ``g_i`` and ``d_i`` are min-max normalized so the three terms are
comparable regardless of raw units. Expanding the cardinality term
(with ``x_i ** 2 == x_i`` for binaries) yields a standard QUBO whose
diagonal holds the linear coefficients and whose upper triangle holds the
pairwise couplings.
"""

from __future__ import annotations

from typing import Any

QUBO = dict[tuple[int, int], float]


def _param_gain_abs(row: dict[str, Any]) -> float:
    if row.get("param_gain_abs") is not None:
        return float(row["param_gain_abs"])
    before = row.get("params_before")
    after = row.get("params_after")
    if before is not None and after is not None:
        return float(before) - float(after)
    return 0.0


def _map50_drop(row: dict[str, Any]) -> float:
    drop = row.get("map50_drop")
    return 0.0 if drop is None else max(0.0, float(drop))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 1e-12:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / span for v in values]


def build_selection_qubo(
    probe_results: list[dict[str, Any]],
    max_layers: int,
    *,
    gain_weight: float = 1.0,
    drop_penalty: float = 2.0,
    cardinality_penalty: float = 1.5,
) -> tuple[QUBO, list[str], dict[str, Any]]:
    """Return ``(Q, layer_names, meta)`` for the layer-selection QUBO.

    ``Q`` maps ``(i, i)`` to a linear coefficient and ``(i, j)`` (i < j) to
    a quadratic coupling. Minimizing ``sum_ij Q[i, j] x_i x_j`` (with the
    diagonal acting as the linear term) selects the layers.
    """
    rows = list(probe_results)
    n = len(rows)
    layer_names = [str(r.get("layer_name", f"layer_{i}")) for i, r in enumerate(rows)]

    gains = _minmax([_param_gain_abs(r) for r in rows])
    drops = _minmax([_map50_drop(r) for r in rows])

    k = max(0, min(max_layers, n))
    alpha, beta, gamma = gain_weight, drop_penalty, cardinality_penalty

    q: QUBO = {}
    # Linear terms: objective + expansion of gamma*(sum x - k)^2.
    # (sum x - k)^2 = sum_i (1 - 2k) x_i + 2 sum_{i<j} x_i x_j + k^2
    for i in range(n):
        linear = -alpha * gains[i] + beta * drops[i] + gamma * (1.0 - 2.0 * k)
        q[(i, i)] = linear

    # Pairwise couplings from the cardinality term.
    for i in range(n):
        for j in range(i + 1, n):
            q[(i, j)] = q.get((i, j), 0.0) + 2.0 * gamma

    meta = {
        "num_variables": n,
        "cardinality_target": k,
        "gain_weight": alpha,
        "drop_penalty": beta,
        "cardinality_penalty": gamma,
        "constant": gamma * k * k,
        "normalized_gains": gains,
        "normalized_drops": drops,
    }
    return q, layer_names, meta


def qubo_energy(q: QUBO, bits: list[int] | tuple[int, ...]) -> float:
    """Evaluate ``sum_ij Q[i, j] x_i x_j`` for a binary assignment."""
    energy = 0.0
    for (i, j), coeff in q.items():
        if i == j:
            energy += coeff * bits[i]
        else:
            energy += coeff * bits[i] * bits[j]
    return energy


def qubo_to_ising(q: QUBO) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
    """Convert a QUBO to Ising coefficients via ``x_i = (1 - z_i) / 2``.

    Returns ``(h, j, offset)`` where ``h[i]`` multiplies ``Z_i``, ``j[(i, k)]``
    multiplies ``Z_i Z_k`` and ``offset`` is the constant shift. Minimizing
    ``sum_i h_i Z_i + sum_ik j_ik Z_i Z_k + offset`` over ``z in {-1, +1}``
    is equivalent to minimizing the QUBO over ``x in {0, 1}``.
    """
    h: dict[int, float] = {}
    j: dict[tuple[int, int], float] = {}
    offset = 0.0

    for (a, b), coeff in q.items():
        if a == b:
            # coeff * x = coeff * (1 - z)/2
            offset += coeff / 2.0
            h[a] = h.get(a, 0.0) - coeff / 2.0
        else:
            # coeff * x_a x_b = coeff * (1 - z_a)(1 - z_b)/4
            offset += coeff / 4.0
            h[a] = h.get(a, 0.0) - coeff / 4.0
            h[b] = h.get(b, 0.0) - coeff / 4.0
            j[(a, b)] = j.get((a, b), 0.0) + coeff / 4.0

    return h, j, offset

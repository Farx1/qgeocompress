"""Quantum optimization for compression decisions.

This subpackage frames structural-layer selection as a QUBO (quadratic
unconstrained binary optimization) problem and solves it with QAOA
(Quantum Approximate Optimization Algorithm) on a CPU state-vector
simulator. A classical brute-force / greedy solver provides an exact
reference and a graceful fallback when PennyLane is not installed.
"""

from qgeocompress.quantum.qaoa_selection import (
    is_quantum_available,
    select_layers_by_qaoa,
    solve_bruteforce,
    solve_qaoa,
)
from qgeocompress.quantum.qubo import (
    build_selection_qubo,
    qubo_energy,
    qubo_to_ising,
)

__all__ = [
    "build_selection_qubo",
    "qubo_energy",
    "qubo_to_ising",
    "select_layers_by_qaoa",
    "solve_bruteforce",
    "solve_qaoa",
    "is_quantum_available",
]

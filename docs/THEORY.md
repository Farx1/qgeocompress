# Theory notes

What the code optimises, under which conditions the claims hold, and where they stop.
Every statement is tied to the file that implements it. References were checked against the
papers themselves; a section or equation number is given only where it was read in the source.

1. [Layer selection: QUBO penalty equivalence](#1-layer-selection-qubo-penalty-equivalence)
2. [QAOA: what is and is not guaranteed](#2-qaoa-what-is-and-is-not-guaranteed)
3. [Data-aware low-rank factorization](#3-data-aware-low-rank-factorization)
4. [Calibration metrics used by the gate](#4-calibration-metrics-used-by-the-gate)

---

## 1. Layer selection: QUBO penalty equivalence

**Code:** `src/qgeocompress/quantum/qubo.py` (`build_selection_qubo`).

Each probed layer $i \in \{1,\dots,n\}$ gets a parameter saving $g_i$ and a single-layer
mAP50 drop $d_i \ge 0$, both min-max normalised to $\hat g_i, \hat d_i \in [0,1]$. With weights
$\alpha$ (`gain_weight`) and $\beta$ (`drop_penalty`), the per-layer cost is

$$
c_i = -\alpha\,\hat g_i + \beta\,\hat d_i .
$$

The problem we actually want is the **constrained** one, with $k$ = `max_layers` (clipped to $[0,n]$):

$$
\text{(P)}\qquad \min_{x \in \{0,1\}^n} \; c^\top x \quad \text{s.t.} \quad \mathbf 1^\top x = k .
$$

The code builds the **unconstrained** QUBO with penalty $\gamma$ (`cardinality_penalty`):

$$
E_\gamma(x) = c^\top x + \gamma\,(\mathbf 1^\top x - k)^2 ,
$$

stored without the constant $\gamma k^2$ (kept in `meta["constant"]`), using $x_i^2 = x_i$:
diagonal $c_i + \gamma(1-2k)$, couplings $2\gamma$ for $i<j$. The constraint is an equality, so no
slack variables are needed (an inequality $\mathbf 1^\top x \le k$ would need extra slack bits;
the code does not use that form).

**Proposition (sufficient penalty).** If $\gamma > \max_i |c_i|$, then
$\arg\min_x E_\gamma(x) = \arg\min \text{(P)}$ (as sets of minimisers).

*Proof.* Take $x$ with $\mathbf 1^\top x = k + m$, $m \ge 1$. Removing any selected $i$ changes the
energy by $-c_i + \gamma\big((m-1)^2 - m^2\big) = -c_i - \gamma(2m-1) \le |c_i| - \gamma < 0$.
If $m \le -1$, some $i$ is unselected (since $k \le n$) and adding it changes the energy by
$c_i - \gamma(2|m|-1) \le |c_i| - \gamma < 0$. So every infeasible $x$ has a strictly better
neighbour, hence every minimiser of $E_\gamma$ is feasible. On the feasible set the penalty is $0$
and $E_\gamma = c^\top x$, so the minimisers coincide. $\square$

Because $\hat g_i, \hat d_i \in [0,1]$, $|c_i| \le \max(\alpha, \beta)$, so
$\gamma > \max(\alpha,\beta)$ is sufficient **for every input**. This input-free bound cannot be
lowered: a layer with $\hat g_i = 0,\ \hat d_i = 1$ has $c_i = \beta$, and if $\gamma < \beta$,
dropping it from a feasible $k$-set lowers the energy by $\beta - \gamma$.

**Consequence for the defaults.** The defaults are $\alpha = 1,\ \beta = 2,\ \gamma = 1.5$, so
$\gamma < \max(\alpha,\beta)$: the budget is **soft**. The QUBO can return fewer than $k$ layers
(more than $k$ is truncated after solving, in `select_layers_by_qaoa`; fewer is not corrected). This
matches the module docstring ("push cardinality toward $k$"), but it means the QUBO solves (P) only
when $\gamma > \max_i |c_i|$ for the instance at hand.

**References.**
- F. Glover, G. Kochenberger, Y. Du, *A Tutorial on Formulating and Using QUBO Models*,
  [arXiv:1811.11538](https://arxiv.org/abs/1811.11538). Section 5 (Transformation #1): the penalty
  scalar "can always be chosen so that the optimal solution to QUBO is the optimal solution to the
  original constrained problem". Section 4 separates hard constraints (penalty large enough to
  preclude a violation) from soft ones.
- A. Lucas, *Ising formulations of many NP problems*, Front. Phys. 2, 5 (2014),
  [arXiv:1302.5843](https://arxiv.org/abs/1302.5843). §2.2 lower-bounds the constraint weight with
  the same single-violation argument used above; §3 treats linear equality constraints.

---

## 2. QAOA: what is and is not guaranteed

**Code:** `src/qgeocompress/quantum/qaoa_selection.py` (`solve_qaoa`).

QAOA with depth $p$ prepares
$|\boldsymbol\gamma,\boldsymbol\beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l B} e^{-i\gamma_l C}\,|+\rangle^{\otimes n}$
and optimises $F_p(\boldsymbol\gamma,\boldsymbol\beta) = \langle \boldsymbol\gamma,\boldsymbol\beta | C | \boldsymbol\gamma,\boldsymbol\beta \rangle$.
Farhi, Goldstone and Gutmann show $M_p \ge M_{p-1}$ (eq. 9) and
$\lim_{p\to\infty} M_p = \max_z C(z)$ (eq. 10), via the adiabatic argument of their Section VI,
where $M_p$ is the **optimum** of $F_p$ over all angles (stated for maximisation; here we minimise
$E_\gamma$).

That is the only guarantee, and the code is outside it on three counts:
1. $p$ is fixed and small (`layers=2`);
2. the angles come from 40 gradient steps from a random start, so $M_p$ itself is not reached;
3. the returned bitstring is the lowest-energy one among the 16 most probable, not the optimum.

So a QAOA answer carries **no optimality guarantee** at the sizes used. Because the simulator is
capped at 16 variables and `solve_bruteforce` is exact up to 22, an exact reference is always
available: compare the QAOA energy with the brute-force optimum on the same QUBO
(`--quantum-backend classical`), as described in [`QUANTUM.md`](QUANTUM.md).

**Reference.** E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization Algorithm*,
[arXiv:1411.4028](https://arxiv.org/abs/1411.4028), eqs. (9)–(10), Section VI.

---

## 3. Data-aware low-rank factorization

**Code:** `src/qgeocompress/compression/data_aware.py` (`collect_patch_gram`, `data_aware_factors`).

A conv layer with kernel $W \in \mathbb R^{c_\text{out} \times d}$, $d = c_\text{in} k_h k_w$, acts on
im2col patches $X \in \mathbb R^{d \times N}$. What matters downstream is the output error:

$$
\min_{\operatorname{rank}(\hat W) \le r} \; \|(W - \hat W) X\|_F .
$$

Let $XX^\top = LL^\top$ with $L$ invertible (Cholesky). Since
$\|AX\|_F^2 = \operatorname{tr}(A XX^\top A^\top) = \|AL\|_F^2$,
the objective equals $\|WL - \hat W L\|_F$. The map $\hat W \mapsto \hat W L$ is a rank-preserving
bijection, so by Eckart–Young the optimum is the truncated SVD of $WL$:

$$
WL = U \Sigma V^\top, \qquad \hat W^\star = U_r \Sigma_r V_r^\top L^{-1}, \qquad
\|(W - \hat W^\star)X\|_F = \Big(\sum_{i>r} \sigma_i(WL)^2\Big)^{1/2}.
$$

**Does the code do this?** Yes, up to a ridge and a fallback:
- `collect_patch_gram` accumulates $XX^\top$ over the real inputs (up to 200 000 random patches per
  batch), and `data_aware_factors` computes `svd(W @ L)`, then `down` $= \Sigma_r V_r^\top L^{-1}$
  (triangular solve) and `up` $= U_r$: exactly $\hat W^\star$, split into a $k\times k$ conv and a
  $1\times 1$ conv.
- **Ridge:** $L$ is the Cholesky factor of $XX^\top + \varepsilon I$ with
  $\varepsilon = 10^{-6}\cdot\operatorname{mean}\operatorname{diag}(XX^\top)$ (multiplied by 10, up to
  six times, if the factorization fails). The result is optimal for $X$ augmented with
  $\sqrt\varepsilon\, I$, i.e. near-optimal for $X$.
- **Silent fallback:** if all six Cholesky attempts fail, $L = I$ and the result is plain SVD.

The same theorem explains the frontier study's H3 result: plain SVD ($L=I$) minimises
$\|W - \hat W\|_F$, so it cannot beat the data-aware factor on $\|(W-\hat W)X\|_F$ (up to the ridge).
The drift-corrected refit and the Lagrangian rank allocation in the same file are **not** covered
by this argument.

**References.**
- C. Eckart, G. Young, *The approximation of one matrix by another of lower rank*,
  Psychometrika 1(3):211–218, 1936.
- X. Wang, Y. Zheng, Z. Wan, M. Zhang, *SVD-LLM: Truncation-aware Singular Value Decomposition for
  Large Language Model Compression*, ICLR 2025, [arXiv:2403.07378](https://arxiv.org/abs/2403.07378).
  Theorem 3.2 and Corollary 3.3: with $S$ the Cholesky factor of $XX^\top$, truncating the $i$-th
  singular value of $WS$ costs exactly $\sigma_i$, so truncating the smallest is best. Optimality
  over *all* rank-$r$ matrices is the Eckart–Young step above.

---

## 4. Calibration metrics used by the gate

**Code:** `src/qgeocompress/evaluation/gt_matching.py`, `src/qgeocompress/calibration/ece.py`,
`src/qgeocompress/evaluation/calibration_real.py`, `src/qgeocompress/evaluation/reliability_gate.py`.

**Sample.** Every detection with confidence $s_j \ge 0.001$ is matched greedily, by descending
confidence, to an unmatched ground-truth box of the same class; it is correct ($y_j = 1$) when the
oriented IoU is $\ge 0.5$. Missed ground truth does not enter the calibration sample.

**ECE** (15 equal-width bins $B_b$ on $[0,1]$, $N$ detections):

$$
\mathrm{ECE} = \sum_{b} \frac{|B_b|}{N}\,\Big|\,\overline{y}_{B_b} - \overline{s}_{B_b}\Big| .
$$

Binning makes this a **lower bound** of the true calibration error, not an unbiased estimate: for
any binning scheme the binned calibration error is at most the true one (Kumar et al.,
Proposition 3.3, by Jensen's inequality), and it can be $0$ while the true error is $\ge 0.49$
(Example 3.2). The per-bin plug-in estimate is also biased on finite samples (Section 5). This is
why ECE is **reported**, not gated.

**CER@0.8** (confident error rate):

$$
\mathrm{CER}@\tau = \frac{\big|\{j : s_j \ge \tau,\ y_j = 0\}\big|}{\big|\{j : s_j \ge \tau\}\big|},
\qquad \tau = 0.8, \qquad \mathrm{CER}@\tau := 0 \text{ if no } s_j \ge \tau .
$$

**Gate** (`evaluate_reliability_gate`, default `GateThresholds`): a candidate passes when
$\mathrm{mAP50}_\text{base} - \mathrm{mAP50}_\text{cand} \le 0.10$ and
$\mathrm{CER}@0.8_\text{cand} - \mathrm{CER}@0.8_\text{base} \le 0.02$; it is a
*deployable compression candidate* if, in addition, parameters fall by $\ge 2\%$.

Limits of CER@0.8 as a gate metric:
- it is a ratio over the few detections above $0.8$, reported without a confidence interval;
- a model that becomes under-confident (no score above $0.8$) gets $\mathrm{CER} = 0$ and passes
  this check trivially; only the mAP50 check catches that case.

**Reference.** A. Kumar, P. Liang, T. Ma, *Verified Uncertainty Calibration*, NeurIPS 2019,
[arXiv:1909.10155](https://arxiv.org/abs/1909.10155), Example 3.2, Proposition 3.3, Section 5.

# Decision-margin weighting: theory and a falsifiable pilot

Status: exploratory research, not a production compressor or a claim of scientific priority.

Follow-up: [detection-level distillation with second/fourth moments](detection_moments.md)
tests localized geometry and per-anchor margins, with proofs and another complete pilot.

## Question and related work

Can weighting calibration images by proximity to a confidence threshold improve a
low-rank OBB detector at the **same ranks and parameter count** as uniform activation weighting?

The ingredients have substantial prior art:

- [FWSVD, Hsu et al., ICLR 2022](https://arxiv.org/abs/2207.00112):
  Fisher importance addresses the mismatch between weight reconstruction and task accuracy.
- [BALF, González Martínez, v3 2026](https://arxiv.org/abs/2509.25136v3):
  activation-aware low-rank compression, including convolutions and budget allocation.
  This is especially close to the project's existing activation Gram factorization.
- [HQOD, Huang et al., ICME 2024](https://arxiv.org/abs/2408.02561):
  compression can worsen classification/localization inconsistency and the decisions made by NMS.
  Its method is quantization-aware training, not the image weighting tested here.
- [Mind the Approximation: Fisher-Weighted SVD Compression for ViTs, 2026](https://arxiv.org/abs/2609.07155):
  a better approximation of a theoretical importance matrix need not yield better compressed accuracy.

These papers motivate testing a task-related surrogate; they do not establish this pilot's
effectiveness. The weighted SVD solution below is standard linear algebra, not a new theorem
claimed by this project. This targeted search is not an exhaustive novelty review.

## Exact statement in an affine setting

Fix inputs $x_i\in\mathbb R^d$, downstream vectors $a_i\in\mathbb R^p$, biases $b_i$,
and $W\in\mathbb R^{p\times d}$. Let $m_i=a_i^T W x_i+b_i\ne0$ be scalar decision
margins. Only $W$ changes: $\widehat W=W+\Delta$. Classify by $m_i>0$.

**Proposition.** With $\alpha_i=\|a_i\|_2^2/m_i^2$ and
$C=n^{-1}\sum_i\alpha_i x_i x_i^T$,

$$
\frac1n\sum_i \mathbf1\{\mathbf1[m_i>0]\ne\mathbf1[m_i+a_i^T\Delta x_i>0]\}
\le \frac1n\sum_i\frac{(a_i^T\Delta x_i)^2}{m_i^2}
\le \operatorname{tr}(\Delta C\Delta^T).
$$

**Proof.** A changed sign requires $|a_i^T\Delta x_i|\ge|m_i|$, including ties
under the stated convention. Otherwise the first indicator is zero. Cauchy–Schwarz gives
$(a_i^T\Delta x_i)^2\le\|a_i\|^2\|\Delta x_i\|^2$. Sum and use
$\|\Delta x_i\|^2=\operatorname{tr}(\Delta x_i x_i^T\Delta^T)$. QED.

**Rank-constrained minimizer of the regularized upper bound.** For $\gamma>0$, let
$LL^T=C+\gamma I$. Among matrices $\widehat W$ of rank at most $r$,

$$
\widehat W=(WL)_r L^{-1}
$$

minimizes $\operatorname{tr}((\widehat W-W)(C+\gamma I)(\widehat W-W)^T)$.
Indeed this objective equals $\|\widehat W L-WL\|_F^2$. Since $L$ is invertible,
$\widehat W\mapsto\widehat W L$ preserves rank, and truncated SVD solves the
resulting Frobenius problem by the Eckart–Young theorem. Its minimum is
$\sum_{j>r}\sigma_j(WL)^2$. Adding the ridge is conservative for the preceding
bound because its extra term is $\gamma\|\Delta\|_F^2\ge0$. QED.

This is an empirical finite-set bound, not a population generalization theorem.
If the bound exceeds one it may be vacuous. Zero original margins are excluded.

## Counterexamples and the gap to YOLO

1. **A smaller bound need not give fewer flips.** Take margins $(1,1)$.
   Perturbations $(-1.1,0)$ have squared normalized error $1.21$ and one flip;
   perturbations $(0.9,0.9)$ have error $1.62$ and zero flips. Optimizing the surrogate
   does not order the actual discrete losses.
2. **First derivatives do not certify finite nonlinear changes.** For
   $m(w)=1-w^2$, at $w=0$ the derivative vanishes, but a perturbation of $2$ flips the sign.
   A nonlinear extension would need a bounded Taylor remainder or another verified bound.
3. **Clipping loses the exact certificate.** Substituting
   $1/\max(|m_i|,\epsilon)^2$ for $1/m_i^2$ decreases some weights. The exact
   inequality need not survive. The same problem applies to $1/(m_i^2+\epsilon^2)$.
4. YOLO's downstream vectors are neither fixed nor measured here. Each image contains
   many anchors, classes and spatial patches; a single image weight discards their
   correspondence. NMS also depends on box overlap and score ordering. No claim of
   certified NMS stability or detection accuracy follows from the affine proposition.

The tests exercise the Gram identity, the optimal weighted tail energy, the affine
inequality, invalid inputs, and the two explicit counterexamples. Numerical tests are
implementation checks, not substitutes for the proofs above.

## Fixed pilot protocol

Run from the repository root, with project dependencies installed:

```bash
PYTHONPATH=src python scripts/margin_weighting_study.py
pytest tests/test_margin_weighting.py tests/test_data_aware.py -q
```

The runner's `PROTOCOL` fixes the following choices before evaluating any arm in a run;
this is not an externally registered preregistration.

- Pretrained `yolo11n-obb.pt`, CPU, float32 inference, 640-square letterbox, two Torch threads.
- Bundled DOTA128 images. Group by the original scene identifier before `__`; take the
  lexicographically first tile from each chosen scene. For seeds 0 and 1, independently
  shuffle scene IDs and use 16 for calibration and the next 32 for evaluation.
  No scene crosses calibration/evaluation **within a seed**. Seeds can overlap each other.
- Select the two largest eligible 3x3 convolutions by weight count, before inspecting
  metrics; standard groups, input channels at most 128, excluding the final detector head.
- Ratios 0.5 and 0.7 of maximum matrix rank, floored; reject non-saving ranks.
  All three arms use exactly the same layers and ranks: plain SVD, uniform activation
  Gram SVD, and margin-weighted activation Gram SVD. No fine-tuning or BN recalibration.
- On baseline calibration outputs, take the maximum class probability at each anchor.
  For each image average $1/\max(|s-0.25|,0.05)^2$ over its 128 highest-scoring anchors.
  Normalize these image priorities to mean one. Report effective sample size
  $(\sum_i w_i)^2/\sum_i w_i^2$ to expose weight concentration.
- Compute per-image patch Grams with at most 2048 uniformly sampled patches per layer.
  The uniform and weighted arms share the **identical patch samples**. Reuse the
  repository's `data_aware_factors` ridge and factorization rather than a new compressor.
- Evaluate all arms without convolution/BN fusion. Predictor settings: confidence 0.001,
  NMS IoU 0.7, maximum 300 detections per image, batch 1, rectangular batching disabled.
- Primary contrast: margin minus uniform repository AP50, separately for each seed/rank.
  Paired percentile bootstrap with 500 resamples of scene IDs. One tile per scene makes
  image resampling equivalent to scene resampling here. Intervals are exploratory,
  unadjusted for multiple comparisons; classes present can change between resamples.
- Secondary: precision/recall at 0.25 and raw anchor threshold flips, including the
  fraction among anchors active in either baseline or compressed model, and score MSE.
  Raw anchor metrics use identical preprocessing and anchor alignment; they are not NMS metrics.
- Store image and label SHA256s, checkpoint SHA256, dependency versions, script SHA256,
  source base commit, ranks, image priorities, per-image AP records and metrics.
  The experiment code is identified by the PR plus script hash; `source_commit` alone
  identifies the starting checkout, not uncommitted research additions.

**Limits.** DOTA128 is a small convenience dataset and may overlap detector pretraining.
Scene separation here only addresses leakage from compression calibration. This is not
a clean unseen-domain benchmark. Repository AP uses its probabilistic OBB overlap and
all-point interpolation; it is not the official DOTA polygon-IoU evaluation. Parameter
savings are not measured latency savings. Two compressed layers are not a full-network
budget allocation study. The two seeds must not be pooled as independent replications.

## Results

Executed on YOLO11n-OBB with Torch 2.14.0+cpu and Ultralytics 8.4.153.
[Full machine-readable results](../../results/research/margin_weighting_pilot.json)
retain all arms, manifests and per-image AP records. AP entries below are percentages;
deltas and intervals are **percentage points**, not relative percentage changes.

| Seed | Rank ratio | Plain AP50 | Uniform AP50 | Margin AP50 | Margin − uniform | Paired 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.5 | 94.8298 | 94.7639 | 94.6982 | −0.0657 | [−0.1296, +0.0089] |
| 0 | 0.7 | 94.6971 | 94.7951 | 94.8606 | +0.0655 | [−0.0085, +0.1763] |
| 1 | 0.5 | 93.3667 | 93.3865 | 93.4403 | +0.0539 | [−0.0006, +0.1297] |
| 1 | 0.7 | 93.3971 | 93.3814 | 93.3914 | +0.0101 | [−0.0226, +0.0500] |

Baseline AP50 is 94.8618% for seed 0 and 93.3535% for seed 1. Compressing
`model.7.conv` and `model.20.conv` gives 2,484,208 parameters at ratio 0.5
and 2,588,016 at ratio 0.7, versus 2,664,432 baseline parameters. Those are
approximately 6.76% and 2.87% whole-model parameter reductions, identical across methods.

**Finding: this pilot does not establish an AP benefit from image margin weighting.**
Every paired interval contains zero. There are three positive point estimates and one
negative; none warrants selecting a winner. The surrogate's intended mechanism also
fails a useful diagnostic: raw threshold flips per image are **higher for margin
weighting than uniform weighting in all four comparisons**:

| Seed | Ratio | Uniform flips/image | Margin flips/image |
|---|---:|---:|---:|
| 0 | 0.5 | 1.5000 | 1.6250 |
| 0 | 0.7 | 0.7500 | 0.7813 |
| 1 | 0.5 | 1.3438 | 1.4375 |
| 1 | 0.7 | 0.5313 | 0.6563 |

This is descriptive evidence on a small pilot, not proof that every margin-based method
fails. Effective calibration sizes fall from 16 to 10.58 and 7.04. Weight concentration
and loss of anchor/patch correspondence are possible explanations, not established causes.
The practical decision is to retain this as an experimental negative/inconclusive result,
without changing the default compression method. Further research should test localized
sensitivities and a clean independent dataset rather than optimize these pilot scores.

Validation: 18 targeted tests pass, including existing data-aware factorization tests;
Ruff passes on all added Python files. Both seeds and all 12 compressed arms executed.

## Research leads beyond this pilot

1. Replace image priorities with **anchor-specific downstream sensitivities**, including
   classification and localization, and measure whether their cost is justified against
   uniform Grams and Fisher-based baselines. The current proposition explains which
   correspondence the image proxy discards.
2. Derive a **nonlinear remainder bound** for one compressed block before composing
   multiple blocks. First check whether the resulting certificate is non-vacuous on
   real samples; a valid but huge bound is not a useful discovery.
3. Study **NMS comparison margins**: score ordering and rotated-box overlap thresholds.
   Preserving confidence threshold crossings alone cannot preserve NMS. A useful result
   would couple geometric perturbation bounds to actual suppression decisions.

Each lead needs separate prior-art review, a falsifiable claim and independent data.
None is asserted here to be unprecedented.

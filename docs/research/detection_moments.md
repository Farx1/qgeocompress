# Detection-level distillation with second and fourth rectangle moments

Exploratory follow-up to [the image-weighting pilot](margin_weighting.md).
This is not a claim of a new state of the art or an end-to-end stability certificate.

## Questions and prior art

Does restoring individual teacher detections improve a fixed low-rank approximation?
Does an orientation-sensitive fourth moment help beyond covariance matching? Does
per-anchor inverse-margin weighting then help beyond the same unweighted loss?

Relevant established work includes:

- [Localization Distillation, CVPR 2022](https://arxiv.org/abs/2102.12252): localization
  information and the selection of distillation regions already matter in detector KD.
- [Location-Aware Discriminant Analysis, WACV 2026](https://arxiv.org/abs/2509.17968):
  explicitly uses location information during detector compression and importance tracing.
- [Gaussian Wasserstein rotated detection, ICML 2021](https://arxiv.org/abs/2101.11952):
  distribution-based box losses address angle/parameterization discontinuities.
- [COBB, CVPR 2024](https://arxiv.org/abs/2402.18975): continuous oriented-box representations
  and decoding ambiguities have already received theoretical treatment.
- [Anisotropic Gaussian boxes, Neurocomputing 2025](https://arxiv.org/abs/2510.16445):
  specifically addresses isotropic variance in square-like boxes.
- [A simple method for fitting of bounding rectangle to closed regions](https://www.sciencedirect.com/science/article/pii/S0031320306003530):
  the publisher search excerpt mentions fourth-order moments for rotation normalization.
  Full text was inaccessible in this review; this is an unresolved prior-art lead,
  not a paper whose method has been reproduced here.

Therefore neither localization distillation, square ambiguity, nor use of higher moments
is claimed as an invention. This pilot tests one inexpensive combination in QGeoCompress.
A novelty claim would require a deeper comparison to these methods, not just an absent keyword match.

## Derivation

Let $(X,Y)$ be uniform on the centered axis-aligned rectangle of positive side lengths
$w,h$. For its rotation by $\theta$, write $Z=e^{i\theta}(X+iY)$.
Independence and symmetry give

$$E[X^2]=w^2/12,\quad E[X^4]=w^4/80,\quad E[X^2Y^2]=w^2h^2/144.$$

Consequently the covariance of the rotated rectangle is

$$\Sigma=R_\theta\operatorname{diag}(w^2/12,h^2/12)R_\theta^T,$$

and expanding $(X+iY)^4$ gives the complex fourth moment

$$q_4=E[Z^4]=\frac{3(w^4+h^4)-10w^2h^2}{240}e^{4i\theta}.$$

The odd cross terms vanish. The coefficient follows from
$E[X^4]+E[Y^4]-6E[X^2Y^2]$. These are direct moment identities.

**Parameterization invariance.** Replacing $(w,h,\theta)$ by
$(h,w,\theta+\pi/2)$ leaves both $\Sigma$ and $q_4$ unchanged: the covariance
eigenaxes swap while the symmetric fourth coefficient and $e^{4i\theta}$ remain
unchanged. Adding $\pi$ to $\theta$ also leaves both unchanged.

**Identifiability on positive-size rectangles.** The center, covariance and $q_4$
together determine the geometric rectangle, modulo its equivalent parameterizations.
For $w\ne h$, the covariance's distinct eigenvalues and eigenvectors already determine
its two lengths and axes. For $w=h=a>0$, covariance gives $a$ but not orientation;
$q_4=-a^4 e^{4i\theta}/60$ determines $\theta$ modulo $\pi/2$, exactly the square's
rotational symmetry. This proves the claim. No assertion of optimal conditioning or
global optimization convergence is implied.

**A covariance counterexample.** A square and its 45-degree rotation have identical
covariances. Their exact polygon IoU is $1/\sqrt 2$, not one. A covariance-only
distance can thus miss a real geometric difference. This does not contradict the
usefulness of Gaussian losses; it identifies information lost by the representation.

## Implemented surrogate and its limits

Freeze teacher boxes $t$ and define $s=\max(\operatorname{tr}\Sigma_t,1)$.
For a student box $b$, the geometry error is

$$L_2=\|c_b-c_t\|^2/s+\|\Sigma_b-\Sigma_t\|_F^2/s^2,$$
$$L_{2,4}=L_2+|q_{4,b}-q_{4,t}|^2/s^4.$$

Covariance is stored as $(\Sigma_{xx},\Sigma_{yy},\sqrt 2\Sigma_{xy})$ so its
vector squared distance equals the full symmetric matrix Frobenius distance.
Away from the one-pixel scale floor these losses are invariant to a shared positive
spatial rescaling. The fourth moment is not a Wasserstein distance or an IoU bound.

For equal-size squares above the scale floor, the fourth term reduces to
$36\sin^2(2\Delta\theta)/25$. It detects the 45-degree discrepancy, but its angular
gradient is zero at that maximum. Therefore even a representation that separates the
boxes does not guarantee that gradient descent can correct every initialization.
Real-valued math identities also do not constitute interval-certified floating-point code.

Each selected anchor contributes ten times mean squared class-probability error plus
the geometry error. An optional weight $1/\max(|s_t-0.25|,0.05)^2$ is normalized to
mean one within the image and multiplies both components. All selections, targets and
scales use the frozen teacher. As in the preceding pilot, clipping removes the exact
inverse-margin decision bound; there is no nonlinear or NMS certificate.

## Protocol fixed before running the arms

```bash
PYTHONPATH=src python scripts/detection_distillation_study.py
pytest tests/test_detection_moments.py tests/test_margin_weighting.py tests/test_data_aware.py -q
```

- YOLO11n-OBB, CPU float32, 640-square input, two Torch threads, deterministic algorithms.
- Seeds 2 and 3; 16 calibration and 32 evaluation scenes per seed, with one tile per scene.
  Uses the preceding pilot's scene splitting function. New seeds do **not** create an
  independent dataset: scenes may overlap the preceding experiment or the other seed.
- Uniform activation SVD initialization of `model.7.conv` and `model.20.conv`, ratio 0.5.
- Four arms with identical compressed architecture: no refinement; score+covariance;
  score+covariance+fourth; the latter with per-anchor margin weights.
- Only the two low-rank blocks are trainable. All other parameters and BN statistics
  remain frozen; the network stays in evaluation mode throughout refinement.
- Adam, learning rate $10^{-5}$, 64 steps, gradient norm clipped at 1, no weight decay.
  One seeded calibration ordering repeats for four passes. Same schedule for all trained arms.
  No early stopping, selection on evaluation scores, or hyperparameter sweep.
- At most 128 highest-scoring teacher anchors per image, retaining scores at least 0.05;
  fall back to the single highest anchor if none qualify. This selection is heuristic.
- Primary representation contrast: fourth versus covariance. Secondary contrasts:
  margin versus fourth, and fourth versus no refinement.
- Repository AP50, operating precision/recall and threshold flips follow the previous
  pilot. Paired 500-draw scene bootstrap intervals are exploratory and unadjusted.
- Additionally record **exact convex-polygon overlap with the teacher** on aligned top
  teacher anchors scoring at least 0.25, including a near-square subset (aspect ratio
  at most 1.2). This measures teacher fidelity, not ground-truth accuracy; anchors from
  the same image are correlated. It is intentionally separate from probabilistic OBB AP.
- Store manifests, hashes, complete AP records, losses and training times. Training time
  is a local wall-clock observation, not an inference speed benchmark.

This pilot still uses DOTA128 with possible pretraining overlap. It has neither official
DOTA polygon AP nor comparisons against full implementations of LD/COBB/anisotropic
Gaussian methods. Those are required before publication-grade superiority claims.

## Results

Full [results and per-image records](../../results/research/detection_distillation_pilot.json)
are committed. All eight arms executed, including six refinements of 64 steps each.
Each compressed model has 2,484,208 parameters versus 2,664,432 in the teacher.

| Seed | Teacher AP50 (%) | Uniform SVD | Covariance refinement | Fourth refinement | Fourth + margin |
|---|---:|---:|---:|---:|---:|
| 2 | 98.62223 | 98.59360 | 98.58930 | 98.58955 | 98.59065 |
| 3 | 95.30000 | 95.20088 | 95.32141 | 95.19961 | 95.20005 |

Primary contrast, fourth minus covariance, in **AP percentage points**:

| Seed | Difference | Paired 95% percentile interval |
|---|---:|---:|
| 2 | +0.00024 | [−0.00141, +0.00114] |
| 3 | −0.12179 | [−0.92374, +0.00186] |

All six reported primary/secondary intervals include zero. The apparent covariance
gain on seed 3 is not a reason to pick that seed or declare a winner.

Mean teacher polygon IoU on the **near-square subset**, descriptive only:

| Seed | Anchors | Uniform | Covariance | Fourth | Fourth + margin |
|---|---:|---:|---:|---:|---:|
| 2 | 448 | 0.993240 | 0.993349 | 0.993258 | 0.993449 |
| 3 | 530 | 0.994230 | 0.993947 | 0.994240 | 0.994178 |

The all-anchor teacher IoU is already about 0.99385 without refinement. The differences
are tiny and mixed. These are correlated anchors, not 978 independent examples.
Mean calibration loss decreased from the first to the last pass in all six refinements;
this confirms optimization progress, but it did not translate into a reliable evaluation
advantage. Loss magnitudes across different objectives must not be compared directly.
Refinement took approximately 7.9–8.7 seconds per arm on this runtime; this does not
measure deployment latency or the cost of the complete study.

**Decision:** do not change the default compressor. The proposed fourth moment repairs
a representational ambiguity, but this pilot does not demonstrate an AP advantage or
a stable fidelity benefit. It neither proves all fourth-moment methods ineffective nor
establishes this combination as a publishable advance.

Validation: 24 targeted tests passed across the two research additions and existing
data-aware factors. Tests include independent numerical integration of the fourth
moment, equivalent parameterizations, the exact square polygon counterexample,
autograd finite-difference checking, and the stationary-maximum limitation. All new
Python files pass Ruff.

## Next decision gate

Before expanding this into a method, fix a more challenging compression regime and an
independent dataset, then compare with stronger localization distillation and rotated-box
loss baselines. Use exact polygon ground-truth AP, equal training budgets and a locked
evaluation set. A targeted square-rotation stress test can establish mechanism, but cannot
replace real detection evaluation. Stop pursuing this formulation if the additional
complexity yields no reproducible practical benefit under that protocol.

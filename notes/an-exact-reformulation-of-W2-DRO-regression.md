# An Exact Reformulation of W2-DRO Regression

This note describes the regression problem solved by
`fit_W2_DRO_weighted_AR1_model`. The goal is to fit a weighted AR(1) model

```math
y_{t+1} = \mu + A y_t
```

from weighted transition samples. Write each transition as

```math
\zeta_t = (x_t, y_t) = (y_t, y_{t+1}),
\qquad
\hat P = \sum_{t=1}^n p_t \delta_{\zeta_t},
```

where the weights satisfy `p_t >= 0` and `sum_t p_t = 1`.

## The DRO Problem

The joint W2-DRO regression problem is

```math
\min_{\mu,A}
\sup_{Q: W_2(Q,\hat P) \le \rho}
E_Q\left[\|y-\mu-Ax\|_2^2\right].
```

Here `W_2` is the 2-Wasserstein distance on the stacked transition space
`R^{2m}` with ground cost `\|\cdot\|_2^2`. The ambiguity set is placed on the
full transition pair `(x,y)`: the adversary may redistribute the weighted
empirical transition points, perturbing both the current observation `x` and
the next observation `y`, subject to a total root-mean-square transport budget
`\rho`. One adversarial law `Q` must explain all output coordinates together,
using one shared transport budget.

## Step 1: Dualize the Inner Supremum

By Wasserstein strong duality (Blanchet–Murthy, Gao–Kleywegt; the quadratic
loss satisfies the required quadratic-growth condition), for fixed `(\mu, A)`
and `\rho > 0`,

```math
\sup_{Q: W_2(Q,\hat P) \le \rho}
E_Q\left[\|y-\mu-Ax\|_2^2\right]
=
\min_{\lambda \ge 0}\;
\lambda\rho^2
+ \sum_{t=1}^n p_t
\sup_{\zeta \in R^{2m}}
\Big( \|B\zeta - \mu\|_2^2 - \lambda\|\zeta-\zeta_t\|_2^2 \Big),
```

where the residual map is written through

```math
r_{\mu,A}(x,y) = y-\mu-Ax = B\begin{bmatrix}x\\y\end{bmatrix}-\mu,
\qquad
B = [-A \quad I_m].
```

Substituting `\delta = \zeta - \zeta_t` and writing
`r_t = y_t - \mu - Ax_t`, the inner supremum is a quadratic in `\delta`:

```math
\sup_{\delta}\;
\|r_t + B\delta\|_2^2 - \lambda\|\delta\|_2^2.
```

This is finite if and only if `\lambda I_{2m} \succeq B^\top B`, equivalently

```math
\lambda \ge \|B\|_2^2 = 1 + \|A\|_2^2,
```

since `BB^\top = I_m + AA^\top`. When it is finite, the maximizer is
`\delta^\star = (\lambda I - B^\top B)^{-1} B^\top r_t`, and by the
push-through identity

```math
I + B(\lambda I - B^\top B)^{-1}B^\top
= \lambda(\lambda I - BB^\top)^{-1}
```

the supremum evaluates to

```math
\lambda\, r_t^\top \big(\lambda I_m - BB^\top\big)^{-1} r_t
=
\lambda\, r_t^\top \big((\lambda-1) I_m - AA^\top\big)^{-1} r_t.
```

The exact problem is therefore

```math
\min_{\mu,\,A,\;\lambda \ge 1+\|A\|_2^2}\;
\lambda\rho^2
+ \lambda \sum_{t=1}^n p_t\,
(y_t-\mu-Ax_t)^\top
\big((\lambda-1)I_m - AA^\top\big)^{-1}
(y_t-\mu-Ax_t).
```

This is jointly convex in `(\mu, A, \lambda)`: each summand is a supremum of
functions that are affine in `\lambda` and convex quadratic in `(\mu, A)`.

### Sanity Check: the Scalar Case

For `m = 1`, write `c = 1+\beta^2` and `E = \sum_t p_t r_t^2`. Minimizing
`\lambda\rho^2 + \lambda E/(\lambda - c)` over `\lambda > c` gives
`\lambda^\star = c + \sqrt{cE}/\rho` and optimal value

```math
\left(\sqrt{\textstyle\sum_t p_t r_t^2} + \rho\sqrt{1+\beta^2}\right)^2,
```

the familiar closed form for scalar W2-DRO least-squares regression.

## Step 2: Linearize into an SDP

The nonlinearity is confined to one matrix that factors affinely. The
epigraph condition

```math
s_t \ge \sup_{\delta}\;\|r_t + B\delta\|_2^2 - \lambda\|\delta\|_2^2
```

states that a quadratic form in `(\delta, 1)` is nonnegative, which is the
matrix inequality

```math
\begin{bmatrix}
\lambda I_{2m} - B^\top B & -B^\top r_t \\
-r_t^\top B & s_t - r_t^\top r_t
\end{bmatrix}
\succeq 0
\quad\Longleftrightarrow\quad
\begin{bmatrix}
\lambda I_{2m} & 0 \\
0 & s_t
\end{bmatrix}
- C_t^\top C_t \succeq 0,
\qquad
C_t := [\,B \quad r_t\,] \in R^{m \times (2m+1)}.
```

The key observation is that `C_t` is affine in the decision variables:
`B = [-A \; I_m]` is affine in `A`, and `r_t = y_t - \mu - Ax_t` is affine in
`(\mu, A)`. A Schur complement then turns the condition into a linear matrix
inequality:

```math
\begin{bmatrix}
I_m & C_t \\
C_t^\top & \operatorname{diag}(\lambda I_{2m},\, s_t)
\end{bmatrix}
\succeq 0.
```

The generalized Schur complement handles the boundary case
`\lambda = 1 + \|A\|_2^2` correctly: it permits it exactly when the residuals
are compatible with the degenerate directions, which is the right closure of
the epigraph.

## The SDP

Putting the pieces together, the exact W2-DRO regression problem is the
semidefinite program

```math
\begin{aligned}
\min_{\mu,\, A,\, \lambda \ge 0,\, s \in R^n}
\quad & \lambda\rho^2 + \sum_{t=1}^n p_t\, s_t \\
\text{s.t.}
\quad &
\begin{bmatrix}
I_m & -A & I_m & y_t - \mu - Ax_t \\
-A^\top & \lambda I_m & 0 & 0 \\
I_m & 0 & \lambda I_m & 0 \\
(y_t - \mu - Ax_t)^\top & 0 & 0 & s_t
\end{bmatrix}
\succeq 0,
\qquad t = 1,\dots,n.
\end{aligned}
```

Every block is affine in `(\mu, A, \lambda, s)`, so this is a genuine SDP
whose optimal value equals the original min–sup exactly — no relaxation is
made anywhere. Each linear matrix inequality has size `3m+1`, and there is
one per transition sample with positive weight.

## Compression via the Second-Moment Matrix

The per-sample program grows linearly in the number of samples, but the
objective touches the data only through a weighted second moment, and this
collapses the `n` linear matrix inequalities into a single one of constant
size. Each residual is affine in the augmented data vector
`w_t = (1, x_t, y_t)`:

```math
r_t = y_t - \mu - Ax_t = \Theta w_t,
\qquad
\Theta = [\,-\mu \quad -A \quad I_m\,] \in R^{m \times (2m+1)},
```

with `\Theta` affine in `(\mu, A)`. Since each dual term is a trace,
`\lambda\, r_t^\top M^{-1} r_t = \lambda \operatorname{tr}(M^{-1} r_t r_t^\top)`
with `M = \lambda I - BB^\top`, the weighted sum becomes

```math
\lambda \sum_{t=1}^n p_t\, r_t^\top M^{-1} r_t
= \lambda \operatorname{tr}\big(M^{-1}\, \Theta \Sigma \Theta^\top\big),
\qquad
\Sigma = \sum_{t=1}^n p_t\, w_t w_t^\top.
```

The matrix `\Sigma` is a constant `(2m+1) \times (2m+1)` sufficient statistic
of the weighted data. Factor it once as `\Sigma = LL^\top` (a rank-revealing
factorization, so `L` has `k \le 2m+1` columns even when `\Sigma` is rank
deficient) and set `\tilde R = \Theta L`, which is affine in `(\mu, A)` and of
size `m \times k`. Replacing the scalar epigraph variables by one symmetric
`k \times k` matrix variable `S` with

```math
S \succeq \lambda\, \tilde R^\top \big(\lambda I - BB^\top\big)^{-1} \tilde R,
```

which holds with equality at the optimum since only `\operatorname{tr}(S)`
enters the objective, the same Schur-complement argument gives the equivalent
constant-size SDP

```math
\begin{aligned}
\min_{\mu,\, A,\, \lambda \ge 0,\, S}
\quad & \lambda\rho^2 + \operatorname{tr}(S) \\
\text{s.t.}
\quad &
\begin{bmatrix}
I_m & -A & I_m & \tilde R \\
-A^\top & \lambda I_m & 0 & 0 \\
I_m & 0 & \lambda I_m & 0 \\
\tilde R^\top & 0 & 0 & S
\end{bmatrix}
\succeq 0.
\end{aligned}
```

The single linear matrix inequality has size `3m + k \le 5m + 1` regardless of
the number of samples; the sample size enters only through the one-off
assembly of `\Sigma`. This is the formulation the implementation solves.

## Implementation Notes

- Samples with `p_t = 0` contribute nothing to `\Sigma` and are removed
  before it is assembled.
- For `\rho = 0` the ambiguity set collapses to `\{\hat P\}` and the problem
  reduces to weighted least squares, which is solved directly in closed form.
- `\Sigma` is factored by eigendecomposition, discarding eigenvalues below a
  relative tolerance, so rank deficiency (for example, very few positively
  weighted samples) shrinks `k` rather than causing failure.
- The Julia implementation builds the LMI with JuMP's `PSDCone()` and hands
  the problem to a conic optimizer with semidefinite support.

## Interpretation

The multiplier `\lambda` prices a unit of squared transport. At the optimum
it adapts to the largest singular value of `B = [-A \; I_m]`, that is, to
`\sqrt{1+\|A\|_2^2}`: the adversary spends its budget only in the directions
where the residual map amplifies perturbations the most, and the fitted model
is hedged against exactly that. The robustness penalty therefore acts on the
operator norm of the model, leaving small singular directions of `A`
untaxed, while the coupling through the single shared budget makes all `m`
output equations compete for the same adversarial transport.

## References

- J. Blanchet and K. Murthy (2019). Quantifying distributional model risk via
  optimal transport. *Mathematics of Operations Research* 44(2), 565–600.
  (Strong duality for Wasserstein DRO.)
- R. Gao and A. Kleywegt (2023). Distributionally robust stochastic
  optimization with Wasserstein distance. *Mathematics of Operations Research*
  48(2), 603–655. (Strong duality and the growth condition for the quadratic
  loss.)
- J. Blanchet, Y. Kang, and K. Murthy (2019). Robust Wasserstein profile
  inference and applications to machine learning. *Journal of Applied
  Probability* 56(3), 830–857. (Exact W2-DRO square-loss regression via the
  Wasserstein profile function; the scalar closed form.)
- V. A. Nguyen, S. Shafieezadeh-Abadeh, D. Kuhn, and P. Mohajerin Esfahani
  (2023). Bridging Bayesian and minimax mean square error estimation via
  Wasserstein distributionally robust optimization. *Mathematics of Operations
  Research* 48(1), 1–37. (Semidefinite representations of Wasserstein-robust
  quadratic objectives.)

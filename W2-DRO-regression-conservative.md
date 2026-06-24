# Conservative W2-DRO Regression

This note describes the regression problem solved by `W2-DRO-regression.jl`.
The goal is to fit a weighted AR(1) model

```math
y_{t+1} = \mu + A y_t
```

from weighted transition samples. Write each transition as

```math
\zeta_t = (x_t, y_t) = (y_t, y_{t+1}),
\qquad
\hat P = \sum_t p_t \delta_{\zeta_t},
```

where the weights satisfy `p_t >= 0` and `sum_t p_t = 1`.

## The DRO Problem

The ideal joint W2-DRO regression problem is

```math
\min_{\mu,A}
\sup_{Q: W_2(Q,\hat P) \le \rho}
E_Q\left[\|y-\mu-Ax\|_2^2\right].
```

The ambiguity set is placed on the full stacked transition pair `(x,y)`.
This means one adversarial law `Q` must explain all output coordinates
together, using one shared Wasserstein transport budget.

For scalar output this has the familiar closed form

```math
\left(
\sqrt{\sum_t p_t (y_t-\mu-\beta^\top x_t)^2}
+ \rho \sqrt{1+\|\beta\|_2^2}
\right)^2.
```

Because the outer square is monotone, fitting the model is equivalent to
minimizing the square-root expression inside.

Here is why that scalar supremum evaluates this way. Stack the covariate and
response into `zeta = (x,y)` and write the scalar residual as

```math
e_{\mu,\beta}(\zeta)
= y-\mu-\beta^\top x
= b^\top \zeta-\mu,
\qquad
b = \begin{bmatrix}-\beta\\1\end{bmatrix}.
```

For any distribution `Q` in the W2 ball, couple a random point `Z ~ Q` with an
empirical point `\hat Z ~ \hat P`. Then

```math
e_{\mu,\beta}(Z)
= e_{\mu,\beta}(\hat Z) + b^\top(Z-\hat Z).
```

Taking the `L_2` norm under the coupling and using the triangle inequality,

```math
\sqrt{E_Q[e_{\mu,\beta}(Z)^2]}
\le
\sqrt{E_{\hat P}[e_{\mu,\beta}(\hat Z)^2]}
+ \|b\|_2 \sqrt{E[\|Z-\hat Z\|_2^2]}.
```

Since `W_2(Q,\hat P) <= rho`, the coupling cost can be made no larger than
`rho`, so

```math
\sqrt{E_Q[e_{\mu,\beta}(Z)^2]}
\le
\sqrt{\sum_t p_t (y_t-\mu-\beta^\top x_t)^2}
+ \rho\sqrt{1+\|\beta\|_2^2}.
```

In the scalar case this bound is tight: the adversary can move each empirical
point in the direction of `b`, with the sign and size aligned to the current
residual. That makes all residuals grow in the same `L_2` direction and spends
exactly the W2 budget. Squaring both sides gives the closed form above.

For vector output, the exact joint W2-DRO objective is more tightly coupled.
It involves the matrix

```math
(\lambda - 1)I - AA^\top
```

and the domain condition

```math
\lambda > 1 + \|A\|_2^2.
```

That exact formulation is valid, but it is more involved than a single conic
least-squares problem.

## Conservative Upper Bound

The code uses a conservative, solver-friendly upper bound. Define the residual
map on the stacked transition pair by

```math
r_{\mu,A}(x,y) = y-\mu-Ax
              = [-A \quad I]\begin{bmatrix}x\\y\end{bmatrix}-\mu.
```

Let

```math
B = [-A \quad I].
```

The robust risk is the supremum over all distributions `Q` inside the W2 ball.
Fix one such `Q`. By the definition of W2, there is a coupling between
`Z = (X,Y) ~ Q` and an empirical transition `\hat Z = (\hat X,\hat Y) ~ \hat P`
whose root mean squared transport cost is at most `rho`, up to an arbitrarily
small error:

```math
\sqrt{E[\|Z-\hat Z\|_2^2]} \le \rho.
```

Under this coupling, the transported residual decomposes as

```math
r_{\mu,A}(Z)
= r_{\mu,A}(\hat Z) + B(Z-\hat Z).
```

Now take the `L_2` norm of this vector-valued random variable. The first term
is exactly the weighted empirical residual norm:

```math
\|r_{\mu,A}(\hat Z)\|_{L_2}
=
\sqrt{\sum_t p_t \|y_t-\mu-Ax_t\|_2^2}.
```

For the transport term, if `Delta = Z-\hat Z`, then

```math
\|B\Delta\|_2 \le \|B\|_2 \|\Delta\|_2
              \le \|B\|_F \|\Delta\|_2.
```

Therefore

```math
\|B(Z-\hat Z)\|_{L_2}
\le
\|B\|_F\sqrt{E[\|Z-\hat Z\|_2^2]}
\le
\rho\|B\|_F.
```

Combining the two pieces with Minkowski's inequality gives, for every feasible
`Q`,

```math
\sqrt{E_Q[\|y-\mu-Ax\|_2^2]}
\le
\sqrt{\sum_t p_t \|y_t-\mu-Ax_t\|_2^2}
+ \rho \|[-A \quad I]\|_F.
```

Because this holds for every `Q` in the ambiguity set, it also holds after
taking the supremum:

```math
\sqrt{
\sup_{Q: W_2(Q,\hat P) \le \rho}
E_Q[\|y-\mu-Ax\|_2^2]
}
\le
\sqrt{\sum_t p_t \|y_t-\mu-Ax_t\|_2^2}
+ \rho \|[-A \quad I]\|_F.
```

Squaring gives the conservative robust-risk bound used by the code:

```math
\sup_{Q: W_2(Q,\hat P) \le \rho}
E_Q[\|y-\mu-Ax\|_2^2]
\le
\left(
\sqrt{\sum_t p_t \|y_t-\mu-Ax_t\|_2^2}
+ \rho \|[-A \quad I]\|_F
\right)^2.
```

The implementation minimizes the square-root upper bound

```math
\min_{\mu,A}
\sqrt{\sum_t p_t \|y_t-\mu-Ax_t\|_2^2}
+ \rho \sqrt{\|A\|_F^2 + m}.
```

The constant `m` appears because `||I||_F^2 = m`. The implementation always
uses this full-transition version, so the W2 perturbation acts on both the
current observation `x` and the next observation `y`.

The conservative step has two sources. First, the vector case need not make the
triangle inequality tight; the residuals and adversarial shifts may not all
align in one direction. Second, the code uses `||B||_F` rather than the smaller
operator norm `||B||_2`, which keeps the model in a very simple cone form.

## SOCP Formulation

The conservative problem is a second-order cone program. Introduce epigraph
variables `s` and `q`:

```math
s \ge
\left\|
\begin{bmatrix}
\sqrt{p_1}(y_1-\mu-Ax_1) \\
\vdots \\
\sqrt{p_n}(y_n-\mu-Ax_n)
\end{bmatrix}
\right\|_2,
```

and

```math
q \ge
\left\|
\begin{bmatrix}
\operatorname{vec}(A) \\
1 \\
\vdots \\
1
\end{bmatrix}
\right\|_2.
```

Then solve

```math
\min_{\mu,A,s,q} \quad s + \rho q.
```

Both constraints are ordinary second-order cone constraints, and the objective
is linear. This is why the Julia implementation can hand the problem directly
to JuMP and a conic optimizer.

## Kernel Regularization Flavour

The Frobenius penalty has the same flavour as ridge or kernel regularization.
For a linear multi-output model, each row of `A` is one linear predictor. The
penalty

```math
\|A\|_F^2 = \sum_{j=1}^m \|A_{j,:}\|_2^2
```

is the sum of squared coefficient norms across output equations. In the
full-transition formulation, the regularizer is

```math
\|[-A \quad I]\|_F = \sqrt{\|A\|_F^2 + m},
```

so the only trainable part being shrunk is still the Frobenius norm of `A`.

In a kernelized version, the linear map `x -> Ax` would be replaced by a
function in a vector-valued RKHS. The corresponding conservative DRO bound
would penalize the Hilbert-space norm of that function, just as this finite
dimensional version penalizes the Frobenius, or Hilbert-Schmidt, norm of `A`.

So the robustification can be read in two equivalent ways:

1. A conservative W2-DRO upper bound based on the Lipschitz size of the
   residual map.
2. A square-root least-squares objective with a ridge/kernel-style norm
   penalty controlled by the Wasserstein radius `rho`.

For scalar output these two views coincide with the exact W2-DRO regression
formula. For vector output, the formulation remains convex and conservative,
but it can over-shrink relative to the exact joint W2-DRO model because
`||B||_F >= ||B||_2`.

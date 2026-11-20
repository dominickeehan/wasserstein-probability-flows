# Wasserstein probability flow weights of WPF_weights in weights.jl found by a primal-dual
# interior-point method on its reduced network, without a conic solver or licence. Needs
# WPF_distance_matrix from weights.jl.
#
# Variables are the flows s[t] from the source into observation t, e[t] from observation t to the
# sink and G[i, j] from observation i to a later observation j, and the throughputs x[t] (p_diag in
# weights.jl). The problem is
#       min - Σ_t log x[t] + λ Σ_{i<j} D[i, j] * G[i, j]
# subject to, for each observation t,
#       s[t] + Σ_{i<t} G[i, t] - x[t] = 0
#       x[t] - e[t] - Σ_{j>t} G[t, j] = 0
# and Σ_t s[t] = 1, with s, e and G nonnegative. Rows 1:T of the constraint matrix A are flow in,
# rows T + 1:2T are flow out and row 2T + 1 is the total. Keeping x as a variable makes the Hessian
# H diagonal, so each Newton step solves normal equations with M = A H⁻¹ Aᵀ, a dense
# (2T + 1) × (2T + 1) matrix built from the network structure. Mehrotra's predictor-corrector method
# (doi:10.1137/0802028) reuses one Cholesky factorization of M for both of its steps. A primal
# regularization ρ caps each flow's entry of H⁻¹ at 1/ρ, which keeps M factorizable near the optimum.

using LinearAlgebra

# Small dense factorizations are fastest on one BLAS thread, and the scripts already solve in
# parallel threads.
BLAS.set_num_threads(1)



# Optimal flows for one λ. Stops once the flow and optimality residuals and the duality gap are all
# below tolerance.
function interior_point_WPF(D, λ; tolerance = 1e-10, ρ = 1e-10, maximum_iterations = 200)

        T = size(D, 1)
        m = 2T + 1
        n = 2T + div(T * (T - 1), 2)
        pairs = [(i, j) for j in 2:T for i in 1:(j - 1)]
        cost_scale = 1 + λ * maximum(D)

        # Interior starting point.
        s, e, G = fill(1 / T, T), fill(1 / T, T), zeros(T, T)
        @inbounds for (i, j) in pairs; G[i, j] = 1 / T^2; end
        x = s .+ vec(sum(G, dims = 1))
        zs, ze, zG = ones(T), ones(T), zeros(T, T)
        @inbounds for (i, j) in pairs; zG[i, j] = 1 + λ * D[i, j]; end
        y = zeros(m)

        rx, rs, re, rG, rp = zeros(T), zeros(T), zeros(T), zeros(T, T), zeros(m)
        M = zeros(m, m)
        ws, we, wG, wx, rhs = zeros(T), zeros(T), zeros(T, T), zeros(T), zeros(m)
        Δs, Δe, ΔG, Δx, Δy = zeros(T), zeros(T), zeros(T, T), zeros(T), zeros(m)
        Δzs, Δze, ΔzG = zeros(T), zeros(T), zeros(T, T)
        Δs_affine, Δe_affine, ΔG_affine = zeros(T), zeros(T), zeros(T, T)
        Δzs_affine, Δze_affine, ΔzG_affine = zeros(T), zeros(T), zeros(T, T)
        cs, ce, cG = zeros(T), zeros(T), zeros(T, T)
        converged = false

        # Newton direction for complementarity targets v∘z - c with c = cs, ce and cG, using the
        # factorization F of M.
        function direction!(F)

                @inbounds for t in 1:T
                        ws[t] = (rs[t] + zs[t] - cs[t] / s[t]) * s[t] / (zs[t] + ρ * s[t])
                        we[t] = (re[t] + ze[t] - ce[t] / e[t]) * e[t] / (ze[t] + ρ * e[t])
                        wx[t] = x[t]^2 * rx[t]
                end
                @inbounds for (i, j) in pairs; wG[i, j] = (rG[i, j] + zG[i, j] - cG[i, j] / G[i, j]) * G[i, j] / (zG[i, j] + ρ * G[i, j]); end

                @inbounds for t in 1:T
                        rhs[t] = -rp[t] + ws[t] - wx[t]
                        rhs[T + t] = -rp[T + t] - we[t] + wx[t]
                end
                rhs[m] = -rp[m] + sum(ws)
                @inbounds for (i, j) in pairs
                        rhs[j] += wG[i, j]
                        rhs[T + i] -= wG[i, j]
                end

                Δy .= rhs
                ldiv!(F, Δy)

                @inbounds for t in 1:T
                        Δs[t] = s[t] / (zs[t] + ρ * s[t]) * (Δy[t] + Δy[m]) - ws[t]
                        Δe[t] = e[t] / (ze[t] + ρ * e[t]) * (-Δy[T + t]) - we[t]
                        Δx[t] = x[t]^2 * (-Δy[t] + Δy[T + t] - rx[t])
                        Δzs[t] = (cs[t] - s[t] * zs[t] - zs[t] * Δs[t]) / s[t]
                        Δze[t] = (ce[t] - e[t] * ze[t] - ze[t] * Δe[t]) / e[t]
                end
                @inbounds for (i, j) in pairs
                        ΔG[i, j] = G[i, j] / (zG[i, j] + ρ * G[i, j]) * (Δy[j] - Δy[T + i]) - wG[i, j]
                        ΔzG[i, j] = (cG[i, j] - G[i, j] * zG[i, j] - zG[i, j] * ΔG[i, j]) / G[i, j]
                end
        end

        # Largest step up to 1 that keeps the flows, their duals and the throughputs nonnegative.
        function largest_step()

                α = 1.0
                @inbounds for t in 1:T
                        if Δs[t] < 0; α = min(α, -s[t] / Δs[t]); end
                        if Δe[t] < 0; α = min(α, -e[t] / Δe[t]); end
                        if Δx[t] < 0; α = min(α, -x[t] / Δx[t]); end
                        if Δzs[t] < 0; α = min(α, -zs[t] / Δzs[t]); end
                        if Δze[t] < 0; α = min(α, -ze[t] / Δze[t]); end
                end
                @inbounds for (i, j) in pairs
                        if ΔG[i, j] < 0; α = min(α, -G[i, j] / ΔG[i, j]); end
                        if ΔzG[i, j] < 0; α = min(α, -zG[i, j] / ΔzG[i, j]); end
                end

                return α
        end

        # Average product of each flow and its dual after a step α along the current direction.
        complementarity(α) = (sum((s[t] + α * Δs[t]) * (zs[t] + α * Δzs[t]) + (e[t] + α * Δe[t]) * (ze[t] + α * Δze[t]) for t in 1:T) +
                              sum((G[i, j] + α * ΔG[i, j]) * (zG[i, j] + α * ΔzG[i, j]) for (i, j) in pairs)) / n


        for iteration in 1:maximum_iterations

                # Residuals of the optimality conditions.
                @inbounds for t in 1:T
                        rx[t] = -1 / x[t] + y[t] - y[T + t]
                        rs[t] = -y[t] - y[m] - zs[t]
                        re[t] = y[T + t] - ze[t]
                        rp[t] = s[t] - x[t]
                        rp[T + t] = x[t] - e[t]
                end
                rp[m] = sum(s) - 1
                @inbounds for (i, j) in pairs
                        rG[i, j] = λ * D[i, j] - y[j] + y[T + i] - zG[i, j]
                        rp[j] += G[i, j]
                        rp[T + i] -= G[i, j]
                end
                μ = complementarity(0.0)
                dual_residual = max(maximum(abs, rx), maximum(abs, rs), maximum(abs, re), maximum(abs(rG[i, j]) for (i, j) in pairs)) / cost_scale
                if maximum(abs, rp) <= tolerance && dual_residual <= tolerance && n * μ <= tolerance; converged = true; break; end

                # Normal equations matrix M = A H⁻¹ Aᵀ.
                fill!(M, 0.0)
                @inbounds for t in 1:T
                        h = s[t] / (zs[t] + ρ * s[t])
                        M[t, t] += h; M[t, m] += h; M[m, t] += h; M[m, m] += h
                        M[T + t, T + t] += e[t] / (ze[t] + ρ * e[t])
                        q = x[t]^2
                        M[t, t] += q; M[T + t, T + t] += q; M[t, T + t] -= q; M[T + t, t] -= q
                end
                @inbounds for (i, j) in pairs
                        h = G[i, j] / (zG[i, j] + ρ * G[i, j])
                        M[j, j] += h; M[T + i, T + i] += h; M[j, T + i] -= h; M[T + i, j] -= h
                end

                # Stop if M cannot be factorized, which the regularization ρ prevents in practice.
                F = cholesky!(Symmetric(M); check = false)
                if !issuccess(F); break; end

                # Predictor step towards complementarity zero.
                fill!(cs, 0.0); fill!(ce, 0.0); fill!(cG, 0.0)
                direction!(F)
                α_affine = largest_step()
                σ = (complementarity(α_affine) / μ)^3
                Δs_affine .= Δs; Δe_affine .= Δe; ΔG_affine .= ΔG
                Δzs_affine .= Δzs; Δze_affine .= Δze; ΔzG_affine .= ΔzG

                # Corrector step towards complementarity σμ with Mehrotra's second-order term.
                cs .= σ * μ .- Δs_affine .* Δzs_affine
                ce .= σ * μ .- Δe_affine .* Δze_affine
                @inbounds for (i, j) in pairs; cG[i, j] = σ * μ - ΔG_affine[i, j] * ΔzG_affine[i, j]; end
                direction!(F)
                α = min(1.0, 0.99 * largest_step())

                s .+= α .* Δs; e .+= α .* Δe; x .+= α .* Δx
                zs .+= α .* Δzs; ze .+= α .* Δze
                @inbounds for (i, j) in pairs; G[i, j] += α * ΔG[i, j]; zG[i, j] += α * ΔzG[i, j]; end
                y .+= α .* Δy

        end

        if !converged; error("WPF interior-point method stopped before convergence"); end

        return s, e, G, x
end

# Terminal weights for one λ, with the same interface as WPF_weights.
function interior_point_WPF_weights(observations, λ, WPF_norm; distances = nothing)

        T = length(observations)
        if λ == Inf; return fill(1 / T, T); end
        if λ == 0; weights = zeros(T); weights[end] = 1; return weights; end

        D = isnothing(distances) ? WPF_distance_matrix(observations, WPF_norm) : distances
        e = interior_point_WPF(D, λ)[2]

        return e / sum(e)
end

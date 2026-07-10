using JuMP, LinearAlgebra, Statistics
using COPT

forecast_optimizer = optimizer_with_attributes(COPT.ConeOptimizer, "Logging" => 0, "LogToConsole" => 0, "BarIterLimit" => 50)

function fit_weighted_AR1_model(time_series, weights)
    m = length(time_series[1])

    lagged_time_series = time_series[2:end]
    time_series = time_series[1:end-1]


    
    X = hcat(ones(length(time_series)), stack(time_series)') # Leading column of ones for the additive term.
    weights = Diagonal(weights)
    least_squares_solution = (X'*weights*X) \ (X'*weights*stack(lagged_time_series)')

    μ = least_squares_solution[1, :] # First row is the intercept vector.
    A = least_squares_solution[2:end, :]' # Due to transposing these are ordered differently.
    return μ, A
end

function fit_W2_DRO_weighted_AR1_model_conservative(time_series, weights, radius)
    m = length(time_series[1])

    if radius == 0; return fit_weighted_AR1_model(time_series, weights); end

    inputs = time_series[1:end-1]
    outputs = time_series[2:end]

    n = length(weights)

    model = Model(forecast_optimizer)

    @variables(model, begin
                            μ[1:m]
                            A[1:m, 1:m]
                            empirical_error >= 0
                            coefficient_norm >= 0
                      end)

    @expression(model, residual_terms[i = 1:n, k = 1:m],
        sqrt(weights[i])*(outputs[i][k] - μ[k] - sum(A[k,j]*inputs[i][j] for j in 1:m)))

    @constraint(model, vcat([empirical_error], vec(residual_terms)) in SecondOrderCone())

    @expression(model, coefficient_terms[k = 1:m, j = 1:m], A[k,j])
    @expression(model, response_terms[k = 1:m], 1.0)
    @constraint(model, vcat([coefficient_norm], vec(coefficient_terms), response_terms) in SecondOrderCone())

    @objective(model, Min, empirical_error + radius*coefficient_norm)

    optimize!(model)

    return value.(μ), value.(A)
end

"""
    fit_W2_DRO_weighted_AR1_model

Solve the exact W2-DRO AR(1) regression problem

    min_{μ,A} sup_{Q : W_2(Q,P̂) <= radius} E_Q[||y - μ - Ax||_2^2],

where P̂ is the weighted empirical distribution of transition pairs, as a
semidefinite program. Dualizing the inner supremum via Wasserstein strong duality
(Blanchet-Murthy; Gao-Kleywegt) and evaluating the resulting quadratic-loss
supremum in closed form, in the vein of the Wasserstein profile function of
Blanchet-Kang-Murthy for square-loss regression, gives

    min_{μ,A,λ} λ radius^2 + λ Σ_t p_t r_t'((λ-1)I - AA')⁻¹ r_t,   r_t = y_t - μ - Ax_t,

and with B = [-A I] the sum is λ tr((λI - BB')⁻¹ Θ Σ Θ'), where
Θ = [-μ -A I] is affine in (μ,A) and Σ = Σ_t p_t w_t w_t' with w_t = (1, x_t, y_t)
is a constant second-moment matrix of the data. Factoring Σ = LL' and setting
R = ΘL, the whole objective is the Schur complement of the single linear matrix
inequality [I [B R]; [B R]' diag(λI, S)] ⪰ 0 with tr(S) in the objective,
so the program size is independent of the number of samples. The semidefinite
representation of such W2 quadratic objectives follows Nguyen-Shafieezadeh-Abadeh-
Kuhn-Mohajerin-Esfahani.

References:
- Blanchet & Murthy (2019), Quantifying distributional model risk via optimal
  transport, Mathematics of Operations Research 44(2).
- Gao & Kleywegt (2023), Distributionally robust stochastic optimization with
  Wasserstein distance, Mathematics of Operations Research 48(2).
- Blanchet, Kang & Murthy (2019), Robust Wasserstein profile inference and
  applications to machine learning, Journal of Applied Probability 56(3).
- Nguyen, Shafieezadeh-Abadeh, Kuhn & Mohajerin Esfahani (2023), Bridging Bayesian
  and minimax mean square error estimation via Wasserstein distributionally robust
  optimization, Mathematics of Operations Research 48(1).
"""
function fit_W2_DRO_weighted_AR1_model(time_series, weights, radius)
    m = length(time_series[1])

    if radius == 0; return fit_weighted_AR1_model(time_series, weights); end

    inputs = time_series[1:end-1]
    outputs = time_series[2:end]

    old_weights = weights

    nonzero_weight_indices = weights .> 0.0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    inputs = inputs[nonzero_weight_indices]
    outputs = outputs[nonzero_weight_indices]

    n = length(weights)

    second_moment_matrix = zeros(2*m+1, 2*m+1)
    for i in 1:n
        w = [1.0; inputs[i]; outputs[i]]
        second_moment_matrix .+= weights[i] .* (w*w')
    end
    eigendecomposition = eigen(Symmetric(second_moment_matrix)) # Rank-revealing factorisation ΣΣ' = LL' tolerating rank deficiency.
    retained = eigendecomposition.values .> 1e-12*maximum(eigendecomposition.values)
    L = eigendecomposition.vectors[:, retained]*Diagonal(sqrt.(eigendecomposition.values[retained]))
    k = size(L, 2)

    model = Model(forecast_optimizer)

    @variables(model, begin
                            μ[1:m]
                            A[1:m, 1:m]
                            λ >= 0
                            S[1:k, 1:k], Symmetric
                      end)

    Θ = hcat(-μ, -A, Matrix(1.0*I, m, m)) # Residual factor: y - μ - Ax = Θ(1, x, y).
    R = Θ*L

    Id = Matrix(1.0*I, m, m)
    Zero = zeros(m, m)

    @constraint(model, Symmetric([Id           -A            Id            R;
                                  -A'          λ*Id          Zero          zeros(m,k);
                                  Id           Zero          λ*Id          zeros(m,k);
                                  R'           zeros(k,m)    zeros(k,m)    S]) in PSDCone())

    @objective(model, Min, λ*radius^2 + sum(S[i,i] for i in 1:k))

    optimize!(model)

    #assert_is_solved_and_feasible(model)
    try
        return value.(μ), value.(A)

    catch
        return fit_W2_DRO_weighted_AR1_model_conservative(time_series, old_weights, radius)
        
    end
end

function median_pairwise_distance(samples)
    distances = [
        norm(samples[i] - samples[j], 2)
        for i in 1:length(samples)-1 for j in i+1:length(samples)
    ]

    scale = median(distances)
    return scale > 0 ? scale : 1.0
end

function kernel_analog_forecast(samples, h_multiplier)
    inputs = samples[1:end-1]
    outputs = [samples[i+1] - samples[i] for i in 1:length(samples)-1]
    bandwidth = h_multiplier * median_pairwise_distance(inputs)

    scaled_distances = [-norm(samples[end] - input, 2)^2 / (2*bandwidth^2) for input in inputs]
    kernel_weights = exp.(scaled_distances .- maximum(scaled_distances))
    kernel_weights ./= sum(kernel_weights)

    predicted_change = zeros(length(samples[1]))
    for i in eachindex(outputs)
        predicted_change .+= kernel_weights[i] .* outputs[i]
    end

    return samples[end] + predicted_change
end

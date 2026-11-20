# Dominic Keehan : 2025

using JuMP, MathOptInterface
using COPT, Ipopt


const WPF_optimizer = let
    attributes = Pair{String, Any}[
        "Logging" => 0,
        "LogToConsole" => 0,
        "BarIterLimit" => 50,
    ]

    # Avoid nested oversubscription when Julia solves independent models in
    # outer threaded loops. Keep COPT's default for single-threaded Julia.
    if haskey(ENV, "COPT_THREADS")
        copt_threads = parse(Int, ENV["COPT_THREADS"])
        @assert copt_threads > 0 "COPT_THREADS must be positive."
        push!(attributes, "Threads" => copt_threads)
        push!(attributes, "BarThreads" => copt_threads)
    elseif Threads.nthreads() > 1
        push!(attributes, "Threads" => 1)
        push!(attributes, "BarThreads" => 1)
    end

    optimizer_with_attributes(COPT.ConeOptimizer, attributes...)
end

const WPF_ipopt_lock = ReentrantLock()

"""
    WPF_distance_matrix(observations, WPF_norm)

Precompute pairwise distances for use by [`WPF_weights`](@ref) or
[`WPF_weights_batch`](@ref). A distance matrix for the full data set can be
reused for any leading observation prefix.
"""
function WPF_distance_matrix(observations, WPF_norm)
    T = length(observations)
    distances = zeros(Float64, T, T)

    for j in 2:T, i in 1:(j - 1)
        distance = Float64(WPF_norm(observations[i], observations[j]))
        distances[i, j] = distance
        distances[j, i] = distance
    end

    return distances
end


function _WPF_distance_data(observations, WPF_norm, distances)
    T = length(observations)
    if isnothing(distances)
        return WPF_distance_matrix(observations, WPF_norm)
    end
    if !(distances isa AbstractMatrix) || size(distances, 1) < T || size(distances, 2) < T
        throw(DimensionMismatch("the WPF distance matrix must be at least $T x $T"))
    end
    return distances
end


_uniform_WPF_weights(T) = fill(1 / T, T)

function _terminal_WPF_weights(p, T)
    weights = [max(value(p[i, 2]), 0.0) for i in 1:T]
    total_weight = sum(weights)
    if !(isfinite(total_weight) && total_weight > 0)
        error("the WPF solver did not return a finite probability distribution")
    end
    weights ./= total_weight
    return weights
end


function _build_WPF_conic_model(T, distances)
    Problem = Model(WPF_optimizer)

    @variables(Problem, begin
        1 >= p[i = 1:T, t = 1:2] >= 0
        1 >= p_diag[1:T] >= 0
        1 >= γ[i = 1:T, j = 1:T; i < j] >= 0
        z[1:T] <= 0
    end)

    for t in 1:2
        @constraint(Problem, sum(p[i, t] for i in 1:T) == 1)
    end

    for t in 1:T
        @constraint(Problem, p[t, 1] + sum(γ[i, t] for i in 1:(t - 1)) == p_diag[t])
        @constraint(Problem, p_diag[t] == p[t, 2] + sum(γ[t, j] for j in (t + 1):T))
        @constraint(Problem, [z[t]; 1; p_diag[t]] in MathOptInterface.ExponentialCone())
    end

    entropy = @expression(Problem, sum(z[t] for t in 1:T))
    transport_cost = @expression(
        Problem,
        sum(distances[i, j] * γ[i, j] for j in 2:T for i in 1:(j - 1)),
    )
    return Problem, p, entropy, transport_cost
end


function _set_WPF_objective!(Problem, entropy, transport_cost, λ)
    # Bulk objective replacement is faster in COPT than changing each
    # transport variable's objective coefficient individually.
    @objective(Problem, Max, entropy - λ * transport_cost)
    return
end


function _solve_WPF_conic_model(Problem, p, T)
    try
        optimize!(Problem)
        result_count(Problem) > 0 || return nothing

        termination = termination_status(Problem)
        acceptable_termination = termination in (
            MathOptInterface.OPTIMAL,
            MathOptInterface.ALMOST_OPTIMAL,
            MathOptInterface.LOCALLY_SOLVED,
            MathOptInterface.ALMOST_LOCALLY_SOLVED,
            MathOptInterface.ITERATION_LIMIT,
            MathOptInterface.TIME_LIMIT,
            MathOptInterface.SLOW_PROGRESS,
            MathOptInterface.OTHER_LIMIT,
        )
        acceptable_termination || return nothing

        status = primal_status(Problem)
        status in (MathOptInterface.FEASIBLE_POINT, MathOptInterface.NEARLY_FEASIBLE_POINT) || return nothing
        return _terminal_WPF_weights(p, T)
    catch
        return nothing
    end
end

"""
    WPF_weights(observations, λ, WPF_norm; distances=nothing)

Solve for the terminal probability distribution under the Wasserstein
probability flow using an exponential-cone hypographical formulation. Pass a
matrix returned by [`WPF_distance_matrix`](@ref) as `distances` to avoid
recomputing distances. Occasionally COPT gets stuck, in which case
[`Ipopt_WPF_weights`](@ref) is used as a fallback.
"""
function WPF_weights(observations, λ, WPF_norm; distances = nothing)
    return first(WPF_weights_batch(observations, (λ,), WPF_norm; distances = distances))
end


"""
    WPF_weights_batch(observations, λs, WPF_norm; distances=nothing)

Solve WPF for every regularisation parameter in `λs`, returning one weight
vector per parameter in input order. All finite nonzero parameters reuse a
single COPT model; only its affine objective is replaced between solves.
"""
function WPF_weights_batch(observations, λs, WPF_norm; distances = nothing)
    parameters = collect(λs)
    T = length(observations)
    weights = Vector{Vector{Float64}}(undef, length(parameters))
    parameters_to_solve = Int[]

    for parameter_index in eachindex(parameters)
        λ = parameters[parameter_index]
        if λ == Inf
            weights[parameter_index] = _uniform_WPF_weights(T)
        elseif λ == 0
            terminal_weights = zeros(T)
            terminal_weights[end] = 1
            weights[parameter_index] = terminal_weights
        else
            push!(parameters_to_solve, parameter_index)
        end
    end

    isempty(parameters_to_solve) && return weights

    distance_data = _WPF_distance_data(observations, WPF_norm, distances)
    Problem, p, entropy, transport_cost = _build_WPF_conic_model(T, distance_data)

    for (solve_index, parameter_index) in pairs(parameters_to_solve)
        λ = parameters[parameter_index]
        _set_WPF_objective!(Problem, entropy, transport_cost, λ)
        terminal_weights = _solve_WPF_conic_model(Problem, p, T)

        if isnothing(terminal_weights)
            weights[parameter_index] = Ipopt_WPF_weights(
                observations,
                λ,
                WPF_norm;
                distances = distance_data,
            )
            # A failed solve can leave the reusable model in an unknown state.
            # Rebuild it before continuing the parameter sweep.
            if solve_index < length(parameters_to_solve)
                Problem, p, entropy, transport_cost = _build_WPF_conic_model(T, distance_data)
            end
        else
            weights[parameter_index] = terminal_weights
        end
    end

    return weights
end


WPF_weights(observations, λs::AbstractVector, WPF_norm; distances = nothing) =
    WPF_weights_batch(observations, λs, WPF_norm; distances = distances)

function Ipopt_WPF_weights(observations, λ, WPF_norm; distances = nothing)
    T = length(observations)
    λ == Inf && return _uniform_WPF_weights(T)
    if λ == 0
        weights = zeros(T)
        weights[end] = 1
        return weights
    end

    # Ipopt's configured linear solver is not assumed to be reentrant. The
    # fallback is rare, so serialize it when outer WPF stages run concurrently.
    lock(WPF_ipopt_lock)
    try
        distance_data = _WPF_distance_data(observations, WPF_norm, distances)
        Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

        @variables(Problem, begin
            1 >= p[i = 1:T, t = 1:2] >= 0
            1 >= p_diag[1:T] >= 0
            1 >= γ[i = 1:T, j = 1:T; i < j] >= 0
        end)

        for t in 1:2
            @constraint(Problem, sum(p[i, t] for i in 1:T) == 1)
        end

        for t in 1:T
            @constraint(Problem, p[t, 1] + sum(γ[i, t] for i in 1:(t - 1)) == p_diag[t])
            @constraint(Problem, p_diag[t] == p[t, 2] + sum(γ[t, j] for j in (t + 1):T))
        end

        @objective(
            Problem,
            Max,
            sum(ifelse(p_diag[t] > 0, log(p_diag[t]), -Inf) for t in 1:T) -
            λ * sum(distance_data[i, j] * γ[i, j] for j in 2:T for i in 1:(j - 1)),
        )

        optimize!(Problem)
        return _terminal_WPF_weights(p, T)
    finally
        unlock(WPF_ipopt_lock)
    end
end

function smoothing_weights(observations, α, nothing)

    T = length(observations)

    if α == 0; weights = zeros(T); weights .= 1/T; return weights; end


    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights = weights/sum(weights)

    return weights
end


function windowing_weights(observations, s, nothing)

    T = length(observations)

    weights = zeros(T)

    if s >= T
        weights .= 1
    else
        for t in T:-1:T-(s-1)
            weights[t] = 1
        end
    end

    weights = weights/sum(weights)

    return weights
end

Ipoptimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

function Wp_DRO_weights(p, T, ρ╱ε)

    ε = 10.0
    ρ = ρ╱ε * ε

    if ρ == 0.0; weights = zeros(T); weights .= 1/T; return weights; end
    if ρ >= ε; weights = zeros(T); weights[T] = 1.0; return weights; end

    Problem = Model(Ipoptimizer)

    @variable(Problem, 1.0 >= w[t=1:T] >= 0.0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1.0)
    @constraint(Problem, sum(w[t]*(T-t+1)^p for t in 1:T)*ρ^p <= ε^p)

    @objective(Problem, Max,
        (1/(sum(w[t]^2 for t in 1:T)))*
            ((ε-(sum(w[t]*(T-t+1)^p for t in 1:T))^(1/p)*ρ)^(2*p)))

    optimize!(Problem)

    weights = max.(value.(w),0.0)
    weights = weights/sum(weights)

    return weights
end

function DLBA_W1_DRO_weights(observations, ρ╱ε, nothing)
    return Wp_DRO_weights(1, length(observations), ρ╱ε)
end

function DLBA_W2_DRO_weights(observations, ρ╱ε, nothing)
    return Wp_DRO_weights(2, length(observations), ρ╱ε)
end

#using LinearAlgebra
#d(ξ, ζ) = norm(ξ - ζ, 2)
#@assert sum(abs.(WPF_weights([6.13, 7.85, 6.47, 4.91, 5.54, 7.13], 4.0, d) - [0.0, 0.275, 0.021, 0.0, 0.325, 0.379])) <= 1e-3

DLBA_W1_DRO_cached_weights = let weights = Dict{Tuple{Int, Float64}, Vector{Float64}}(),
                                 weights_lock = ReentrantLock()

    function DLBA_W1_DRO_cached_weights(observations, ρ╱ε, nothing)
        key = (length(observations), Float64(ρ╱ε))

        lock(weights_lock)
        try
            return get!(weights, key) do
                Wp_DRO_weights(1, length(observations), ρ╱ε)
            end
        finally
            unlock(weights_lock)
        end
    end
end

DLBA_W2_DRO_cached_weights = let weights = Dict{Tuple{Int, Float64}, Vector{Float64}}(),
                                 weights_lock = ReentrantLock()

    function DLBA_W2_DRO_cached_weights(observations, ρ╱ε, nothing)
        key = (length(observations), Float64(ρ╱ε))

        lock(weights_lock)
        try
            return get!(weights, key) do
                Wp_DRO_weights(2, length(observations), ρ╱ε)
            end
        finally
            unlock(weights_lock)
        end
    end
end

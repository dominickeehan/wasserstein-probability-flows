# Wasserstein probability flow weights of WPF_weights in weights.jl found by flow shifting on its
# reduced network, without a conic solver or licence. Needs WPF_distance_matrix from weights.jl.
#
# Nodes are the source, the observations 1:T in time order and the sink. Flow s[t] enters
# observation t from the source, γ[i, j] moves from observation i to a later observation j at cost
# λ * D[i, j] per unit and e[t] leaves observation t for the sink. The flow through observation t is
# x[t] (p_diag in weights.jl) and the terminal weights are e. The solver finds the unit flow of
#       min - Σ_t log x[t] + λ Σ_{i<j} D[i, j] * γ[i, j].
# A route then costs λ times its distances minus 1/x[t] for each observation t it visits. At the
# optimum every route that carries flow into a node is a cheapest route into that node. Each sweep
# finds the cheapest route into every node and the dearest route that carries flow into it, and
# moves flow from the dearest to the cheapest where they split. This follows the flow shifting of
# Dial's Algorithm B for traffic assignment (doi:10.1016/j.trb.2006.02.008).



# Flow to move from one route segment onto another that minimizes the objective. Observations only
# on the segment gaining flow are A and observations only on the segment losing flow are B. The
# objective along the move is convex, so the zero of its derivative is found to machine precision by
# Newton's method kept inside a bracket.
function flow_to_move(x, A, B, cost_difference, available)

        derivative(δ) = sum((1 / (x[t] + δ) for t in A); init = 0.0) - sum((1 / (x[t] - δ) for t in B); init = 0.0) - cost_difference
        curvature(δ) = -sum((1 / (x[t] + δ)^2 for t in A); init = 0.0) - sum((1 / (x[t] - δ)^2 for t in B); init = 0.0)

        if derivative(available) >= 0; return available; end

        low, high, δ = 0.0, available, 0.0
        for _ in 1:200
                slope = derivative(δ)
                if slope > 0; low = δ; else; high = δ; end
                if slope == 0 || high - low <= 4 * eps(high); break; end
                newton_step = δ - slope / curvature(δ)
                δ = low < newton_step < high ? newton_step : (low + high) / 2

        end

        return δ
end


# Flow below which an arc counts as unused when finding dearest routes. Floating point leaves tiny
# flows such as 1e-18 on arcs that should be empty, and moving only that much flow per sweep would
# stall the solver.
const negligible_flow = 1e-15

# Cost of the cheapest route into each observation and of the dearest route that carries flow into
# it, with the predecessor of each observation on those routes. Predecessor 0 is the source.
function route_costs!(cheapest, cheapest_from, dearest, dearest_from, x, s, γ, D, λ, observations = 1:length(x))

        for t in observations
                best, best_from = 0.0, 0
                worst, worst_from = s[t] > negligible_flow ? 0.0 : -Inf, 0
                for i in 1:(t - 1)
                        arc_cost = λ * D[i, t]
                        if cheapest[i] + arc_cost < best; best, best_from = cheapest[i] + arc_cost, i; end
                        if γ[i, t] > negligible_flow && dearest[i] + arc_cost > worst; worst, worst_from = dearest[i] + arc_cost, i; end
                end
                cheapest[t] = best - 1 / x[t]
                cheapest_from[t] = best_from
                dearest[t] = worst - 1 / x[t]
                dearest_from[t] = worst_from

        end
end

# Observations and arcs of the route segment from node start back to node stop along the
# predecessors. Arcs are pairs (i, j) with i = 0 for the source and j = T + 1 for the sink.
function segment(start, stop, predecessor, last_observation)

        observations, arcs = Int[], Tuple{Int, Int}[]
        node, previous = start, last_observation
        while true
                push!(arcs, (previous, node))
                previous == stop && break
                push!(observations, previous)
                node, previous = previous, predecessor[previous]
        end

        return observations, arcs
end

arc_flow(s, γ, e, (i, j)) = i == 0 ? s[j] : j == length(s) + 1 ? e[i] : γ[i, j]

function add_flow!(s, γ, e, (i, j), δ)

        if i == 0; s[j] += δ; elseif j == length(s) + 1; e[i] += δ; else; γ[i, j] += δ; end
end


# Optimal unit flow for λ starting from the given flows, so a solve can start from the flow for a
# nearby λ. Stops once the best route beats the flow-weighted average route by less than tolerance,
# which bounds how far the objective is from its minimum.
function equilibrate!(s, γ, e, D, λ; tolerance = 1e-11, maximum_sweeps = 100000)

        T = length(s)
        x = s .+ vec(sum(γ, dims = 1))
        cheapest, dearest = zeros(T), zeros(T)
        cheapest_from, dearest_from = zeros(Int, T), zeros(Int, T)
        gap, sweeps = Inf, 0

        for sweep in 1:maximum_sweeps
                sweeps = sweep
                x .= s .+ vec(sum(γ, dims = 1))
                route_costs!(cheapest, cheapest_from, dearest, dearest_from, x, s, γ, D, λ)

                # Value of the best route against the flow-weighted average value T - λ Σ D γ.
                transport_cost = λ * sum(D[i, j] * γ[i, j] for j in 2:T for i in 1:(j - 1))
                gap = -minimum(cheapest) - (T - transport_cost)
                if gap <= tolerance; break; end

                # Shift flow into every observation and then into the sink. Route costs of earlier observations
                # whose flow changed are recomputed first, so each node sees exact route costs.
                first_stale = T + 1
                for node in 1:(T + 1)
                        if first_stale < node
                                route_costs!(cheapest, cheapest_from, dearest, dearest_from, x, s, γ, D, λ, first_stale:(node - 1))
                                first_stale = T + 1
                        end
                        if node <= T
                                cheap_from, dear_from = cheapest_from[node], dearest_from[node]
                        else
                                cheap_from = argmin(cheapest)
                                dear_from = argmax([e[t] > negligible_flow ? dearest[t] : -Inf for t in 1:T])
                        end
                        if cheap_from == dear_from; continue; end

                        # Node where the two routes last split.
                        a, b = cheap_from, dear_from
                        while a != b
                                if a > b; a = cheapest_from[a]; else; b = dearest_from[b]; end
                        end

                        A, cheap_arcs = segment(node, a, cheapest_from, cheap_from)
                        B, dear_arcs = segment(node, a, dearest_from, dear_from)
                        arc_cost(arcs) = λ * sum((D[i, j] for (i, j) in arcs if i != 0 && j != T + 1); init = 0.0)
                        cost_difference = arc_cost(cheap_arcs) - arc_cost(dear_arcs)
                        available = minimum(arc_flow(s, γ, e, arc) for arc in dear_arcs)
                        value_difference = sum((1 / x[t] for t in A); init = 0.0) - sum((1 / x[t] for t in B); init = 0.0) - cost_difference
                        if available <= 0 || value_difference <= 0; continue; end

                        δ = flow_to_move(x, A, B, cost_difference, available)
                        for arc in cheap_arcs; add_flow!(s, γ, e, arc, δ); end
                        for arc in dear_arcs
                                if δ == available && arc_flow(s, γ, e, arc) == available; add_flow!(s, γ, e, arc, -available); else; add_flow!(s, γ, e, arc, -δ); end
                        end
                        for t in A; x[t] += δ; end
                        for t in B; x[t] -= δ; end
                        first_stale = min(first_stale, node, minimum(A; init = T + 1), minimum(B; init = T + 1))

                end
        end

        if gap > tolerance; error("WPF flow shifting stopped with gap $gap after $sweeps sweeps"); end

        return sweeps, gap
end


# Unit flow along one route through every observation in time order.
function chain_flow(T)

        s, γ, e = zeros(T), zeros(T, T), zeros(T)
        s[1], e[T] = 1.0, 1.0
        for t in 1:(T - 1); γ[t, t + 1] = 1.0; end

        return s, γ, e
end

# Terminal weights for one λ, with the same interface as WPF_weights.
function network_WPF_weights(observations, λ, WPF_norm; distances = nothing)

        T = length(observations)
        if λ == Inf; return fill(1 / T, T); end
        if λ == 0; weights = zeros(T); weights[end] = 1; return weights; end

        D = isnothing(distances) ? WPF_distance_matrix(observations, WPF_norm) : distances
        s, γ, e = chain_flow(T)
        equilibrate!(s, γ, e, D, λ)

        return e / sum(e)
end

# Terminal weights for every λ in λs, each solve starting from the flow for the previous λ.
# Increasing λs are about a fifth faster than starting every solve afresh.
function network_WPF_weights_batch(observations, λs, WPF_norm; distances = nothing)

        T = length(observations)
        D = isnothing(distances) ? WPF_distance_matrix(observations, WPF_norm) : distances
        s, γ, e = chain_flow(T)
        weights = Vector{Vector{Float64}}(undef, length(λs))

        for (index, λ) in enumerate(λs)
                if λ == Inf; weights[index] = fill(1 / T, T); continue; end
                if λ == 0; weights[index] = zeros(T); weights[index][end] = 1; continue; end
                equilibrate!(s, γ, e, D, λ)
                weights[index] = e / sum(e)

        end

        return weights
end

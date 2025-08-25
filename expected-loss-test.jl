using Distributions, Statistics

# Parameters
Q = 100                      # Order quantity
μ = 90                       # Mean demand
σ = 15                       # Standard deviation of demand
Cu = 5                       # Underage cost per unit
Co = 2                       # Overage cost per unit
n = 10000000                  # Number of simulations

# Simulate demand
demand_dist = Normal(μ, σ)
demands = rand(demand_dist, n)

# Calculate overage and underage
overages = max.(Q .- demands, 0)
underages = max.(demands .- Q, 0)

# Compute total costs
total_costs = Cu .* underages .+ Co .* overages
average_cost_simulation = mean(total_costs)

# Analytical calculation
z = (Q - μ) / σ
ϕ = pdf(Normal(), z)
Φ = cdf(Normal(), z)
overage_analytical = σ * (z * Φ + ϕ)
underage_analytical = σ * (ϕ - z * (1 - Φ))
expected_cost_analytical = Cu * underage_analytical + Co * overage_analytical

# Output results
println("Average simulated cost: \$$(round(average_cost_simulation, digits=2))")
println("Expected cost (Analytical): \$$(round(expected_cost_analytical, digits=2))")

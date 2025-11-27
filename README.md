# Wasserstein Probability Flows
Code repository accompanying the paper:

***Nonstationary Distribution Estimation via Wasserstein Probability Flows***

## Abridged Paper Abstract
We study the problem of estimating a sequence of evolving probability distributions from historical data, where the underlying distribution changes over time in a nonstationary and nonparametric manner. To capture gradual changes, we introduce a model that penalises large deviations between consecutive distributions using the Wasserstein distance. This leads to a method in which we estimate the underlying series of distributions by maximizing the log-likelihood of the observations with a penalty applied to the sum of the Wasserstein distances between consecutive distributions. This can be reduced to a simple network-flow problem enabling efficient computation. We call this the Wasserstein Probability Flow method. We carry out numerical tests in different settings. Our results show that the Wasserstein Probability Flow method is a promising tool for applications in nonstationary stochastic optimization.

## Description
This repository provides implementations of our numerical tests and our reduced-form network-flow problem instance.

The following image (Figure 3 of the paper) shows an example Wasserstein probability flow estimate.

<img src="figures/radio-pulsar.svg" width="515pt">

For full details, see Subsection 5.1 of the paper.

## Dependencies
You must install a recent version of Julia from http://julialang.org/downloads/. The paper uses Julia 1.9.

A number of Julia packages are required. They can be added by commanding:

`using Pkg, Pkg.add("Random, Distributions, Statistics, StatsBase, LinearAlgebra, JuMP, MathOptInterface, COPT, Ipopt, IterTools, ProgressBars, Plots, Measures, CSV")`.*

*Note that a license for the COPT solver is also required; see https://www.shanshu.ai/copt. The paper uses COPT 7.2.

## Reproducing Output from the Paper
You can reproduce the following tables and figures by running the following scripts:
- Figure 3: `plot-radio-pulsar.jl`
- Table 2: `ex-post-multi-modal-and-multi-dimensional-newsvendor.jl`
- Table 3: `tabulate-ex-post-multi-modal-and-multi-dimensional-newsvendor.jl`
- Table 5 and Figures 4, 5, and 6: `train-test-dairy-prices.jl`
- Table 7 and Figures 7, 8, and 9: `train-test-stock-returns.jl`

The script `weights.jl` provides an implementation of our reduced-form network-flow problem instance. For a vector of historical observations `observations`; a scalar shift penalty parameter `λ`; and a distance function on the outcome space of the observations `d`, commanding `WPF_weights(observations, λ, d)` returns the terminal Wasserstein probability flow probability distribution using the COPT solver. We also provide an open source alternative `Ipopt_WPF_weights(observations, λ, d)` using the Ipopt solver.

## Thank You :pray:

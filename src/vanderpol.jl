using DynamicPolynomials, SumOfSquares, PolyJuMP, MosekTools
using Plots
using ForwardDiff: jacobian
using DifferentialEquations
using ControlSystems, LinearAlgebra

include("linearization.jl")

function vanderpol(state)
    x₁, x₂ = state
    return [
        -x₂
        x₁ + (x₁^2 - 1)*x₂
    ]
end

function linearize(f, x_G)
    A = jacobian(x -> f(x), x_G)
    return A
end

function optimize_ρ(ρ, J★, J̇̂★, n, ϵ)
    model =
        SOSModel(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    @polyvar x̅[1:n]
    #@variable model ρ
    X = monomials(x̅, 0:2)
    @variable model h SOSPoly(X)
    @constraint model h in SOSCone()
    @constraint model J̇̂★(x̅) + h * (ρ - J★(x̅)) <= -ϵ * (x̅' * x̅)
    #@objective(model, Max, ρ)
    optimize!(model)
    print("done")
    print(termination_status(model))
    return model
end

function main()
    # Take jacobian
    xG = [0;0]
    A = linearize(vanderpol, xG)
    Q = [1 0; 0 1]
    P = lyap(A,Q)

    ρ = 1
    m = optimize_ρ(ρ, x -> x'*P*x, x -> -2*x'*P*vanderpol(x), 2, 1e-3)
    m
end

main()

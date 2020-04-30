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

fun[1:ction linearize(f, x_G)
    A = jacobian(x -> f(x), x_G)
    return A
end

function main()
    # Take jacobian
    xG = [0;0]
    A = linearize(vanderpol, xG)
    Q = I
    P = lyap(A,Q)

    # SOS?
    #m = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))


end

main()

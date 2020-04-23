using DynamicPolynomials, SumOfSquares, PolyJuMP
using ForwardDiff: jacobian
using ControlSystems, LinearAlgebra

const g = 9.81
const m = 0.486
const r = 0.25
const I_z = 0.00383
const dim_x = 6
const dim_u = 2
const ϵ = 1e-3



function quad2d!(derivatives, state, control, t)
    x, y, θ, ẋ, ẏ, θ̇ = state
    u₁, u₂ = control
    dz[1] = ẋ
    dz[2] = ẏ
    dz[3] = θ̇
    dz[4] = -(u₁ + u₂) * sin(θ) / m
    dz[5] = (u₁ + u₂) * cos(θ) / m - g
    dz[6] = (u₁ - u₂) * r / I_z
end

function quad2d(state, control, t)
    (x, y, θ, ẋ, ẏ, θ̇) = state
    (u₁, u₂) = control
    return [
        ẋ
        ẏ
        θ̇
        -(u₁ + u₂) * sin(θ) / m
        (u₁ + u₂) * cos(θ) / m - g
        (u₁ - u₂) * r / I_z
    ]
end

quad2d(state, control) = quad2d(state, control, 0)

function vector_hessian(f, x)
    n = length(x)
    out = jacobian(x -> jacobian(f, x), x)
    return out
end

# function linearize(f, x_G, u_G)
#
# end

L2(x) = sum(y -> y .^ 2, x)

function find_max_rho(f, x_G, u_G, Q, R)
    A = jacobian(x -> f(x, u_G), x_G)
    B = jacobian(u -> f(x_G, u), u_G)
    S = care(A, B, Q, R)
    K = R \ B' * S
    J★(x̅) = x̅' * S * x̅
    f⁽ᶜˡ⁾(x̅) = f(x̅ + x_G, -K * (x̅ + x_G))
    f̂⁽ᶜˡ⁾(x̅) =
        jacobian(f⁽ᶜˡ⁾, x_G) * x̅ .+
        (1 / 2) *
        x̅' *
        reshape(vector_hessian(f⁽ᶜˡ⁾, x_G) * x̅, (length(x̅), length(x̅)))
    J̇̂★(x̅) = 2 * x̅' * S * f̂⁽ᶜˡ⁾(x̅)

    # model = SOSModel()
    # @polyvar x̅[1:6]
    # @variable(model, ρ)
    # X = monomials(x̅, 0:2)
    # @variable model h SOSPoly(X)
    # # cost = J̇̂★(x̅) .+ h .* (ρ .- J★(x̅)) .+ ϵ .* (x̅' * x̅)
    # @constraint model h in DSOSCone()
    # register(model, :J̇̂★, 1, J̇̂★, autodiff = true)
    # register(model, :J★, 1, J★, autodiff = true)
    # @constraint model h <= (ϵ .* (sum(x̅ .* x̅)) .+ J̇̂★(x̅)) ./ (J★(x̅) .- ρ)
    # optimize!(model)
end

function main()
    x_G = repeat([0.0], dim_x)
    u_G = repeat([m * g / 2], dim_u)
    Q = Diagonal([10, 10, 90, 1, 1, r / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]
    find_max_rho(quad2d, x_G, u_G, Q, R)
end

main()

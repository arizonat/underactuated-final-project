using JuMP, MosekTools
using PolyJuMP, SumOfSquares
include("linearization.jl")

# f: one-time-step system dynamics
# ℓ: loss function, ℓ: X × U → ℝ≥0
# x̂ᵢ: initial state at timestep i
# xᵢʳᵉᶠ: Desired state trajectory
# uᵢʳᵉᶠ: Desired control trajectory
# N: horizon length
function mpc(f, ℓ, x̂ᵢ, xᵢʳᵉᶠ, uᵢʳᵉᶠ, N)
    dim_x = size(xᵢʳᵉᶠ)[1]
    dim_u = size(uᵢʳᵉᶠ)[1]

    m = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))

    @variable m Δxᵢ[1:dim_x, 1:N+1] # States
    @variable m Δuᵢ[1:dim_u, 1:N] # Control efforts

    rᵢ = A * xᵢʳᵉᶠ[:, 1:N] + B * uᵢʳᵉᶠ - xᵢʳᵉᶠ[:, 2:N+1] # Dynamics error in desired trajectory
    cost = ℓ(Δxᵢ[:, 1:N], Δuᵢ)
    for n in 1:dim_x
        @constraint m Δxᵢ[n, 1] = x̂ᵢ[n] - xᵢʳᵉᶠ[n, 1] # Enforces start point
    end
    for k in 1:N
        Aᵢₖ, Bᵢₖ = linearize(f, xᵢʳᵉᶠ[:, k], uᵢʳᵉᶠ[:, k])
        rᵢ = Aᵢₖ * xᵢʳᵉᶠ[:, k] + Bᵢₖ * uᵢʳᵉᶠ[:, k] - xᵢʳᵉᶠ[:, k+1]
        Δxᵢₖ₊₁ = Aᵢₖ * Δxᵢ[:, k] + Bᵢₖ * Δuᵢ[:, k] - rᵢ # Dynamics
        for n in 1:dim_x
            @constraint m Δxᵢ[n, 2:N+1] = Δxᵢₖ₊₁[n] # Enforces dynamics
        end
    end
    @objective model Min cost
    optimize!(model)
end

function linear_mpc(A, B, Q, R, x̂ᵢ, xᵢʳᵉᶠ, uᵢʳᵉᶠ, N)
    dim_x = size(xᵢʳᵉᶠ)[1]
    dim_u = size(uᵢʳᵉᶠ)[1]
    m = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    @variable m Δxᵢ[1:dim_x, 1:N+1]
    @variable m Δuᵢ[1:dim_u, 1:N]


    for n in 1:dim_x
        @constraint m Δxᵢ[n, 1] = x̂ᵢ[n] - xᵢʳᵉᶠ[n, 1]
    end

    rᵢ = A * xᵢʳᵉᶠ[:, 1:N] + B * uᵢʳᵉᶠ - xᵢʳᵉᶠ[:, 2:N+1]
    Δxᵢₖ₊₁ = A * Δxᵢ[:, 1:N] + B * Δuᵢ - rᵢ
    for k in 1:N
        for n in 1:dim_x
            @constraint m Δxᵢ[n, 2:N+1] = Δxᵢₖ₊₁[n, k]
        end
    end
    cost = sum(
        Δxᵢ[:, k]' * Q * Δxᵢ[:, k] + Δuᵢ[:, k]' * R * Δuᵢ[:, k] for k in 1:N
    )
    @objective(model, Min, cost)
    optimize!(model)
    return model
end

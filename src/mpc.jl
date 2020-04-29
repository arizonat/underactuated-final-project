using JuMP, MosekTools
using PolyJuMP, SumOfSquares
using Plots
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
        @constraint(m, Δxᵢ[n, 1] == x̂ᵢ[n] - xᵢʳᵉᶠ[n, 1]) # Enforces start point
    end
    for k in 1:N
        Aᵢₖ, Bᵢₖ = linearize(f, xᵢʳᵉᶠ[:, k], uᵢʳᵉᶠ[:, k])
        rᵢ = Aᵢₖ * xᵢʳᵉᶠ[:, k] + Bᵢₖ * uᵢʳᵉᶠ[:, k] - xᵢʳᵉᶠ[:, k+1]
        Δxᵢₖ₊₁ = Aᵢₖ * Δxᵢ[:, k] + Bᵢₖ * Δuᵢ[:, k] - rᵢ # Dynamics
        for n in 1:dim_x
            @constraint(m, Δxᵢ[n, 2:N+1] == Δxᵢₖ₊₁[n]) # Enforces dynamics
        end
    end
    @objective(model, Min, cost)
    optimize!(model)
end

function linear_mpc(A, B, Q, R, x̂ᵢ, xᵢʳᵉᶠ, uᵢʳᵉᶠ, N)
    dim_x = size(xᵢʳᵉᶠ)[1]
    dim_u = size(uᵢʳᵉᶠ)[1]
    m = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    @variable m Δxᵢ[1:dim_x, 1:N+1]
    @variable m Δuᵢ[1:dim_u, 1:N]


    for n in 1:dim_x
        @constraint(m, Δxᵢ[n, 1] == x̂ᵢ[n] - xᵢʳᵉᶠ[n, 1])
    end

    rᵢ = A * xᵢʳᵉᶠ[:, 1:N] + B * uᵢʳᵉᶠ - xᵢʳᵉᶠ[:, 2:N+1]
    Δxᵢₖ₊₁ = A * Δxᵢ[:, 1:N] + B * Δuᵢ - rᵢ
    for k in 1:N
        for n in 1:dim_x
            @constraint(m, Δxᵢ[n, k+1] == Δxᵢₖ₊₁[n, k])
        end
    end
    cost = sum(
        Δxᵢ[:, k]' * Q * Δxᵢ[:, k] + Δuᵢ[:, k]' * R * Δuᵢ[:, k] for k in 1:N
    )
    @objective(m, Min, cost)
    optimize!(m)
    return Δxᵢ, Δuᵢ, m
end

function main()
    num_iters = 10 # Number of MPC optimizations to run
    reject_ratio = 0.8 # Fraction of trajectory to throw out
    N = 15 # number of iterations per MPC optimization; dt = Δt / N
    Δt = 0.75 # Timespan of single MPC optimization

    k = 16.0
    b = 0.8
    m = 1.0
    A = [0.0 1.0; -k / m b / m] * Δt + [1.0 0.0; 0.0 1.0]
    B = [0.0; 1.0] * Δt
    B = reshape(B, (length(B), 1))
    Q = [0.001 0.0; 0.0 0.001]
    R = reshape([100.0], (1, 1))
    x₀ = [0.0; 0.0]


    xG = [4.0 0.0]
    uG = [(k / m) * xG[1]]
    xᵢʳᵉᶠ = repeat(xG, N + 1)' |> collect
    uᵢʳᵉᶠ = repeat(uG, N)' |> collect
    us = reshape(typeof(uG)[], (1, 0))
    xs = reshape(typeof(xG)[], (2, 0))

    reject = Int(reject_ratio * N)
    for i in 1:num_iters
        x̂ᵢ = (i == 1) ? x₀ : xs[:, end]
        x, u, m = linear_mpc(A, B, Q, R, x̂ᵢ, xᵢʳᵉᶠ, uᵢʳᵉᶠ, N)
        xs = hcat(xs, value.(x)[:, 1:end-reject] + xᵢʳᵉᶠ[:, 1:end-reject])
        us = hcat(us, value.(u) + uᵢʳᵉᶠ)
    end

    t = collect(1:size(xs)[2]) * Δt / N
    display(plot(t, xs[1, :]))

    return xs, us
end

main()

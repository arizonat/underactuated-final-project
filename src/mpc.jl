


# f: one-time-step system dynamics
# ℓ: loss function, ℓ: X × U → ℝ≥0
# x̂ᵢ: initial state at timestep i
# xᵢʳᵉᶠ: Desired state trajectory
# uᵢʳᵉᶠ: Desired control trajectory
# N: horizon length
function mpc(f, ℓ, x̂ᵢ, xᵢʳᵉᶠ, uᵢʳᵉᶠ, N, dt)
    X = size(xᵢʳᵉᶠ)[1]
    U = size(uᵢʳᵉᶠ)[1]

    m = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))

    @variable m Δxᵢ[1:X, 1:N+1] # States
    @variable m Δuᵢ[1:U, 1:N] # Control efforts

    cost = sum(ℓ(Δxᵢ[:, k], Δuᵢ[:, k]) for k in 1:N)
    for n in 1:X
        @constraint(m, Δxᵢ[n, 1] == x̂ᵢ[n] - xᵢʳᵉᶠ[n, 1]) # Enforces start point
    end
    for k in 1:N
        Aᵢₖ, Bᵢₖ = linearize(f, xᵢʳᵉᶠ[:, k], uᵢʳᵉᶠ[:, k])
        A = I(X) + Aᵢₖ * dt
        B = Bᵢₖ * dt
        rᵢ = A * xᵢʳᵉᶠ[:, k] + B * uᵢʳᵉᶠ[:, k] - xᵢʳᵉᶠ[:, k+1] # Error from unphysical trajectories
        Δxᵢₖ₊₁ = A * Δxᵢ[:, k] + B * Δuᵢ[:, k] - rᵢ # Dynamics
        for n in 1:X
            @constraint(m, Δxᵢ[n, k+1] == Δxᵢₖ₊₁[n]) # Enforces dynamics
        end
    end
    @objective(m, Min, cost)
    optimize!(m)
    return Δxᵢ, Δuᵢ, m
end

function linear_mpc(A, B, Q, R, x̂ᵢ, xᵢʳᵉᶠ, uᵢʳᵉᶠ, N)
    X = size(xᵢʳᵉᶠ)[1]
    U = size(uᵢʳᵉᶠ)[1]
    m = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    @variable m Δxᵢ[1:X, 1:N+1]
    @variable m Δuᵢ[1:U, 1:N]


    for n in 1:X
        @constraint(m, Δxᵢ[n, 1] == x̂ᵢ[n] - xᵢʳᵉᶠ[n, 1])
    end

    rᵢ = A * xᵢʳᵉᶠ[:, 1:N] + B * uᵢʳᵉᶠ - xᵢʳᵉᶠ[:, 2:N+1]
    Δxᵢₖ₊₁ = A * Δxᵢ[:, 1:N] + B * Δuᵢ - rᵢ
    for k in 1:N
        for n in 1:X
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

function nonlinear_mpc_optimal_control(
    f,
    ℓ,
    x₀,
    xG,
    uG,
    num_iters = 50,
    reject_ratio = 0.8,
    N = 100,
    Δt = 0.75,
)
    num_iters = 50 # Number of MPC optimizations to run
    reject_ratio = 0.8 # Fraction of trajectory to throw out
    N = 100 # number of iterations per MPC optimization; dt = Δt / N
    Δt = 0.75 # Timespan of single MPC optimization

    dt = Δt / N
    dim_x = size(xG)[1]
    dim_u = size(uG)[1]

    xᵢʳᵉᶠ = repeat(xG', N + 1)' |> collect
    uᵢʳᵉᶠ = repeat(uG', N)' |> collect
    us = reshape(Float64[], (dim_u, 0))
    xs = reshape(Float64[], (dim_x, 0))

    rej = Int(reject_ratio * N)
    for i in 1:num_iters
        x̂ᵢ = (i == 1) ? x₀ : xs[:, end]
        x, u, m = mpc(f, ℓ, x̂ᵢ, xᵢʳᵉᶠ, uᵢʳᵉᶠ, N, dt)
        xs = hcat(xs, value.(x)[:, 1:end-rej] + xᵢʳᵉᶠ[:, 1:end-rej])
        us = hcat(us, value.(u) + uᵢʳᵉᶠ)
    end

    t = collect(1:size(xs)[2]) * dt
    return t, xs, us
end

using UnderactuatedFinalProject
using Plots

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

    display(plot(
        t,
        xs[1, :],
        title = "Model predictive control of mass-spring-damper",
        xlabel = "Time",
        ylabel = "Position",
    ))

    return xs, us
end

main()

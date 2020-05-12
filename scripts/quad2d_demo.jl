using UnderactuatedFinalProject
using LinearAlgebra: Diagonal
using DifferentialEquations
using Plots

const g = 9.81
const μ = 0.486
const r = 0.25
const I_z = 0.00383
const dim_x = 6
const dim_u = 2

# function main()
#     x_G = repeat([0.0], dim_x)
#     u_G = repeat([m * g / 2], dim_u)
#     Q = Diagonal([10, 10, 90, 1, 1, r / (2 * pi)])
#     R = [0.1 0.05; 0.05 0.1]
#     find_max_rho(quad2d, x_G, u_G, Q, R)
# end

function quad2d_mpc()
    num_iters = 50 # Number of MPC optimizations to run
    reject_ratio = 0.8 # Fraction of trajectory to throw out
    N = 100 # number of iterations per MPC optimization; dt = Δt / N
    Δt = 0.75 # Timespan of single MPC optimization

    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]
    ℓ(x, u) = x' * Q * x + u' * R * u


    x₀ = [1.0; 1.0; 0.0; 0.0; 0.0; 0.0]

    xG = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    uG = [2.38383; 2.38383]

    t, xs, us = nonlinear_mpc_optimal_control(
        UnderactuatedFinalProject.quad2d,
        ℓ,
        x₀,
        xG,
        uG,
        num_iters,
        reject_ratio,
        N,
        Δt,
    )

    plt = plot_quad2D_frame(xs, Int(round(size(xs)[2] / 2)))
    anim = plot_quad2D_animation(xs)
    gif(anim, "quad2D_mpc.gif", fps = 60)
end

function quad2d_lqr()
    N = 50 * 1000 # number of iterations per MPC optimization; dt = Δt / N
    dt = 0.75 / 100 # Timespan of single MPC optimization
    t = cumsum(repeat([dt], N))
    tspan = (minimum(t), maximum(t))

    #Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * π)]) #Q3 = 0.001
    #R = [10.0 0.05; 0.05 10.0]
    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]

    x₀ = [1.0; 1.0; 0.0; 0.0; 0.0; 0.0]

    xG = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    #uG = [2.38383; 2.38383]
    #uG = μ*g/2.0 * [1; 1]
    uG = [0;0]

    A, B = linearize(quad2d_shifted, xG, uG)
    K, ~ = do_lqr(A, B, Q, R)

    control(x, t) = -K * x

    prob = ODEProblem(quad2d_shifted!, x₀, tspan, control)
    sol = solve(prob, saveat = t)

    xs = hcat(sol.u...)

    print("Maximum input used: ", maximum(abs.(-K*xs)))

    plt = plot_quad2D_frame(xs, Int(round(size(xs)[2] / 2)))
    anim = plot_quad2D_animation(xs[:, 1:600])
    gif(anim, "quad2D_mpc.gif", fps = 120)
end

quad2d_lqr()

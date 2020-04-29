using UnderactuatedFinalProject
using LinearAlgebra: Diagonal
using Plots

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

    Q = Diagonal([10, 10, 90, 1, 1, 0.25 / (2 * pi)])
    R = [10.0 0.05; 0.05 10.0]
    ℓ(x, u) = x' * Q * x + u' * R * u


    x₀ = [0.0; 2.0; π; 0.0; 0.0; 0.0]

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


quad2d_mpc()

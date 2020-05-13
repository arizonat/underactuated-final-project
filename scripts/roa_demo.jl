using UnderactuatedFinalProject
using LinearAlgebra: Diagonal
using Plots, ColorSchemes
using DifferentialEquations

function main()
    clibrary(:colorbrewer)
    f̂ = quad2d_approx
    x_G = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    u_G = [0.0; 0.0]
    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]
    ρ, K, S = find_max_rho(f̂, x_G, u_G, Q, R)
    print(ρ)
    lyap(x, y) = sum([x y] * S[1:2, 1:2] * [x; y])

    lyap(x, y) = sum([x y] * S[1:2, 1:2] * [x; y])
    pl = contourf(
        -1:0.05:1,
        -5:0.05:5,
        lyap,
        levels = [0, ρ, Inf],
        palette = :Pastel1,
    )
    plot!(pl, traj_x, traj_y)
    display(pl)
end

main()

using UnderactuatedFinalProject
using LinearAlgebra: Diagonal
using Plots

function main()
    f̂ = quad2d_approx
    x_G = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    u_G = [0.0; 0.0]
    Q = Diagonal([10, 10, 90, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]
    ρ, K, S = find_max_rho(f̂, x_G, u_G, Q, R)
    lyap(x, y) = sum([x y] * S[1:2, 1:2] * [x; y])

end

main()

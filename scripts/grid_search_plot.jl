using UnderactuatedFinalProject, DifferentialEquations, Plots
using LinearAlgebra

function check_goal_reached(xG, xs)
    # todo: actually try to see if it's stably there?
    ϵ = 3e-2
    d = sqrt.(sum((xs .- xG) .^ 2, dims = 1))
    inds = findall(x -> x <= ϵ, d)
    if isempty(inds)
        return false, 0
    end
    return true, inds[1][2]
end

function main()

    xG = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    #uG = [2.38383; 2.38383]
    #uG = μ*g/2.0 * [1; 1]
    uG = [0; 0]

    A, B = linearize(quad2d_shifted, xG, uG)
    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]
    K, S = do_lqr(A, B, Q, R)

    control(x, t) = -K * x
    xs = -10:0.2:10
    ys = -10:0.2:10
    u0s = [[x; y; 0.0; 0.0; 0.0; 0.0] for x in xs for y in ys]
    tspan = (0.0, 100.0)
    results = Float64[]
    for u0 in u0s
        prob = ODEProblem(quad2d_shifted!, u0, tspan, control)
        sol = solve(prob, Tsit5()) |> collect
        push!(results, check_goal_reached(xG, sol)[1] * 1.0)
    end
    rs = reshape(results, (length(xs), length(ys)))
    pl = contour(xs, ys, rs, fill = true)
    display(pl)
    savefig("pl1.png")
    ρ = 0.7805688476562498
    V(x, y) = sum([x y] * S[1:2, 1:2] * [x; y])
    pl = contourf(xs, ys, V, fill = false, levels = [0.0, ρ, maximum(Z)])
    display(pl)
    savefig("pl2.png")

    pl = contourf(xs, ys, V, fill = true, levels = [0.0, ρ, maximum(Z)])
    display(pl)
    savefig("pl3.png")

    return xs, ys, rs
end

main()

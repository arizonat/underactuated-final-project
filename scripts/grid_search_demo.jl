using DifferentialEquations, Plots, UnderactuatedFinalProject

function grid_search_demo(f, u0s, tspan, xmin, xmax, ymin, ymax; p = nothing)
    sols = []
    for u0 in u0s
        prob = ODEProblem(f, u0, tspan, p)
        sol = solve(prob, Tsit5()) |> collect
        push!(sols, sol)
    end
    pl = plot(
        sols[1][1, :],
        sols[1][2, :],
        xlims = (xmin, xmax),
        ylims = (ymin, ymax),
        legend = false,
    )
    for sol in sols[2:end]
        plot!(
            pl,
            sol[1, :],
            sol[2, :],
            xlims = (xmin, xmax),
            ylims = (ymin, ymax),
            legend = false,
        )
    end
    display(pl)
end

function grid_search_demo3D(
    f,
    u0s,
    tspan,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax;
    p = nothing,
)
    sols = []
    for u0 in u0s
        prob = ODEProblem(f, u0, tspan, p)
        sol = solve(prob, Tsit5()) |> collect
        push!(sols, sol)
    end
    pl = plot3d(
        sols[1][1, :],
        sols[1][2, :],
        sols[1][3, :],
        xlims = (xmin, xmax),
        ylims = (ymin, ymax),
        zlims = (zmin, zmax),
        legend = false,
    )
    for sol in sols[2:end]
        plot3d!(
            pl,
            sol[1, :],
            sol[2, :],
            sol[3, :],
            xlims = (xmin, xmax),
            ylims = (ymin, ymax),
            zlims = (zmin, zmax),
            legend = false,
        )
    end
    display(pl)
end

function sys(du, u, p, t)
    du[1] = u[2]
    du[2] = -0.01 * u[2] - 10.0 * sin(u[1] * pi / 180)
end

function main()
    xG = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    #uG = [2.38383; 2.38383]
    #uG = μ*g/2.0 * [1; 1]
    uG = [0; 0]

    A, B = linearize(quad2d_shifted, xG, uG)
    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]
    K, ~ = do_lqr(A, B, Q, R)

    control(x, t) = -K * x

    u0s = [
        [x; y; θ; 0.0; 0.0; 0.0]
        for x in -10:5.0:10, y in -10:5.0:10, θ in -π:π/2:π
    ]
    tspan = (0.0, 0.1)
    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10
    θmin = -π
    θmax = π
    grid_search_demo3D(
        quad2d_shifted!,
        u0s,
        tspan,
        xmin,
        xmax,
        ymin,
        ymax,
        θmin,
        θmax,
        p = control,
    )
end

main()

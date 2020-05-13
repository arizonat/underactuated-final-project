using DifferentialEquations, Plots

function grid_search_demo(f, u0s, tspan, xmin, xmax, ymin, ymax)
    sols = []
    for u0 in u0s
        prob = ODEProblem(f, u0, tspan)
        sol = solve(prob, Tsit5()) |> collect
        push!(sols, sol)
    end
    pl = plot(
        sols[1][1, :],
        sols[1][2, :],
        xlims = (xmin, xmax),
        ylims = (ymin, ymax),
        legend=false
    )
    for sol in sols[2:end]
        plot!(
            pl,
            sol[1, :],
            sol[2, :],
            xlims = (xmin, xmax),
            ylims = (ymin, ymax),
            legend=false
        )
    end
    display(pl)
end

function sys(du, u, p, t)
    du[1] = u[2]
    du[2] = -0.01 * u[2] - 10.0 * sin(u[1]) - 9.81
end

function main()
    u0s = [[i; j] for i in -5.0:0.5:5.0, j in -5.0:0.5:5.0]
    xmin = -5
    xmax = 5
    ymin = -5
    ymax = 5
    tspan = (0.0, 10.0)
    grid_search_demo(sys, u0s, tspan, xmin, xmax, ymin, ymax)
end

main()

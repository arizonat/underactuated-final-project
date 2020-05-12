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

function find_max_stable_input(ρ, K, S)
    m = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    @variable m x[1:6]
    @constraint(m, x'*S*x <=ρ)
    cost = sum((-K*x).^2)
    @objective(m, Max, cost)
    optimize!(m)
    return x, cost
end

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
        5
    )

    plt = plot_quad2D_frame(xs, Int(round(size(xs)[2] / 2)))
    anim = plot_quad2D_animation(xs[:,1:600])
    gif(anim, "quad2D_mpc.gif", fps = 120)
    return t, xs, us
end

function quad2d_lqr()
    N = 50 * 100 # number of iterations per MPC optimization; dt = Δt / N
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
    us = -K*xs

    plt = plot_quad2D_frame(xs, Int(round(size(xs)[2] / 2)))
    anim = plot_quad2D_animation(xs[:, 1:600])
    gif(anim, "quad2D_lqr.gif", fps = 120)
    return t, xs, us
end

function check_goal_reached(xG, xs)
    # todo: actually try to see if it's stably there?
    ϵ = 3e-2
    d = sqrt.(sum((xs .- xG).^2, dims=1))
    inds = findall(x->x<=ϵ, d)
    if isempty(inds)
        return false, 0
    end
    return true, inds[1][2]
end

function sample_quad2d_ROA_surface(S, ρ, n, zero_vels=false)
    # not necessarily uniform, especially given theta, but probably good enough
    x = rand(6,n)
    if zero_vels
        x[4:6,:] .= 0
    end
    a = sqrt.(ρ./diag((x'*S*x)))
    return a'.*x
end

function compare_quad2d_lqr_mpc()
    # sample starting locations
    # ρ=0.7805688476562498 if Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)]), R = [0.1 0.05; 0.05 0.1]

    # Setup shared constants
    ρ=0.7805688476562498 # TODO: this should be computed live
    nₓ = 10 # number of starting conditions to test

    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]

    # Setup starting points
    x₀s = sample_quad2d_ROA_surface(ρ,S,nₓ,zero_vels=true)
    x₀s .*= 1.1 # step slightly outside the RoA, try some in the boundary as well

    # Setup LQR
    xG_lqr = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    uG_lqr = [0;0]

    A, B = linearize(quad2d_shifted, xG, uG)
    K, S = do_lqr(A, B, Q, R)

    N = 50 * 100 # number of iterations per MPC optimization; dt = Δt / N
    dt = 0.75 / 100 # Timespan of single MPC optimization
    t = cumsum(repeat([dt], N))
    tspan = (minimum(t), maximum(t))
    control(x, t) = -K * x

    # Setup MPC
    xG_mpc = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    uG_mpc = [2.38383; 2.38383]
    ℓ(x, u) = x' * Q * x + u' * R * u

    for i=1:nₓ
        # try with LQR
        # try with non-saturated mpc
        # try with saturated mpc (how should we pick saturations?)
        # check if it reached the goal
    end
end

println("Starting lqr...")
ts, xs, us = quad2d_lqr()
println("max (shifted) thrust single rotor: ",maximum(abs.(us.+μ*g/2)))
println("total thrust: ",norm(us))
println("time step goal reached: ",check_goal_reached([0.0; 0.0; 0.0; 0.0; 0.0; 0.0],xs))
#quad2d_mpc()
println("Starting mpc...")
ts, xs, us = quad2d_mpc()
println("max (shifted) thrust single rotor: ",maximum(abs.(us.+μ*g/2)))
println("total thrust: ", norm(us))
println("time step goal reached: ", check_goal_reached([0.0; 0.0; 0.0; 0.0; 0.0; 0.0],xs))

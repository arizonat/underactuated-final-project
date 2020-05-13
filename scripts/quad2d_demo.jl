using UnderactuatedFinalProject
using LinearAlgebra: Diagonal, diag, norm
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

function quad2d_mpc(sat, sat_number)
    num_iters = 50 # Number of MPC optimizations to run
    reject_ratio = 0.8 # Fraction of trajectory to throw out
    N = 100 # number of iterations per MPC optimization; dt = Δt / N
    Δt = 0.75 # Timespan of single MPC optimization

    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]
    ℓ(x, u) = x' * Q * x + u' * R * u

    x₀ = [2.0; 2.0; 0.0; 0.0; 0.0; 0.0]

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
        sat,
        sat_number
    )

    plt = plot_quad2D_frame(xs, Int(round(size(xs)[2] / 2)))
    anim = plot_quad2D_animation(xs[:,1:900])
    gif(anim, string("quad2D_mpc_",sat,"_",sat_number,".gif"), fps = 120)
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

    x₀ = [2.0; 2.0; 0.0; 0.0; 0.0; 0.0]

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
    anim = plot_quad2D_animation(xs[:, 1:900])
    gif(anim, "quad2D_lqr.gif", fps = 120)
    return t, xs, us
end

function quad2d_lqr_saturated(sat)
    N = 50 * 100 # number of iterations per MPC optimization; dt = Δt / N
    dt = 0.75 / 100 # Timespan of single MPC optimization
    t = cumsum(repeat([dt], N))
    tspan = (minimum(t), maximum(t))

    #Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * π)]) #Q3 = 0.001
    #R = [10.0 0.05; 0.05 10.0]
    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]

    x₀ = [2.0; 2.0; 0.0; 0.0; 0.0; 0.0]

    xG = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    #uG = [2.38383; 2.38383]
    #uG = μ*g/2.0 * [1; 1]
    uG = [0;0]

    A, B = linearize(quad2d_shifted, xG, uG)
    K, ~ = do_lqr(A, B, Q, R)

    control(x, t) = min.(max.(-K * x,-sat-2.38383),sat-2.38383)

    prob = ODEProblem(quad2d_shifted!, x₀, tspan, control)
    sol = solve(prob, saveat = t)

    xs = hcat(sol.u...)
    us = min.(max.(-K * xs,-sat-2.38383),sat-2.38383)

    plt = plot_quad2D_frame(xs, Int(round(size(xs)[2] / 2)))
    anim = plot_quad2D_animation(xs[:, 1:900])
    gif(anim, "quad2D_lqr_saturated.gif", fps = 120)
    return t, xs, us
end

function quad2d_lqr(Q, R, x₀)
    N = 50 * 100 # number of iterations per MPC optimization; dt = Δt / N
    dt = 0.75 / 100 # Timespan of single MPC optimization
    t = cumsum(repeat([dt], N))
    tspan = (minimum(t), maximum(t))

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
    us = -K*xs.+μ*g/2.0 # since we're using shifted dynamics
    return t, xs, us
end

function quad2d_mpc(Q, R, x₀, sat)
    num_iters = 50 # Number of MPC optimizations to run
    reject_ratio = 0.8 # Fraction of trajectory to throw out
    N = 100 # number of iterations per MPC optimization; dt = Δt / N
    Δt = 0.75 # Timespan of single MPC optimization

    ℓ(x, u) = x' * Q * x + u' * R * u

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
        sat
    )

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
    nₓ = 10 # number of starting conditions to test
    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]

    # Setup starting points
    ρ=0.7805688476562498 # TODO: this should be computed live
    ρ*=20
    xG = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    uG = [0;0]
    A, B = linearize(quad2d_shifted, xG, uG)
    K, S = do_lqr(A, B, Q, R)
    x₀s = sample_quad2d_ROA_surface(S,ρ,nₓ,true)
    #x₀s .*= 1.1 # step slightly outside the RoA

    for i=1:nₓ
        x₀ = x₀s[:,i]
        println("trying with: ",x₀)
        # try each controller
        t_lqr, xs_lqr, us_lqr = quad2d_lqr(Q, R, x₀)
        t_mpc, xs_mpc, us_mpc = quad2d_mpc(Q, R, x₀, 0)
        t_mpc_s, xs_mpc_s, us_mpc_s = quad2d_mpc(Q, R, x₀, 5)

        # check if they reached the goal
        success_lqr = check_goal_reached(xG, xs_lqr)
        success_mpc = check_goal_reached(xG, xs_mpc)
        success_mpc_s = check_goal_reached(xG, xs_mpc_s)

        # print results
        println("lqr: ", success_lqr)
        println("mpc: ", success_mpc)
        println("mpc_sat: ", success_mpc_s)
        # save results
    end

    # plot results
end

function compare_quad2d_lqr_mpc_grid()
    # sample starting locations
    # ρ=0.7805688476562498 if Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)]), R = [0.1 0.05; 0.05 0.1]

    # Setup shared constants
    Q = Diagonal([10, 10, 10, 1, 1, 0.25 / (2 * pi)])
    R = [0.1 0.05; 0.05 0.1]

    for i=1:nₓ
        x₀ = x₀s[:,i]
        println("trying with: ",x₀)
        # try each controller
        t_lqr, xs_lqr, us_lqr = quad2d_lqr(Q, R, x₀)
        t_mpc, xs_mpc, us_mpc = quad2d_mpc(Q, R, x₀, 0)
        t_mpc_s, xs_mpc_s, us_mpc_s = quad2d_mpc(Q, R, x₀, 5)

        # check if they reached the goal
        success_lqr = check_goal_reached(xG, xs_lqr)
        success_mpc = check_goal_reached(xG, xs_mpc)
        success_mpc_s = check_goal_reached(xG, xs_mpc_s)

        # print results
        println("lqr: ", success_lqr)
        println("mpc: ", success_mpc)
        println("mpc_sat: ", success_mpc_s)
        # save results
    end

    # plot results
end

println("Starting lqr...")
ts, xs, us = quad2d_lqr()
println("max (shifted) thrust single rotor: ",maximum(abs.(us.+μ*g/2)))
println("total thrust: ",norm(us))
println("time step goal reached: ",check_goal_reached([0.0; 0.0; 0.0; 0.0; 0.0; 0.0],xs))
println("Starting lqr saturated...")
ts, xs, us = quad2d_lqr_saturated(5)
println("max (shifted) thrust single rotor: ",maximum(abs.(us.+μ*g/2)))
println("total thrust: ",norm(us))
println("time step goal reached: ",check_goal_reached([0.0; 0.0; 0.0; 0.0; 0.0; 0.0],xs))
#quad2d_mpc()
println("Starting mpc...")
ts, xs, us = quad2d_mpc(0,0)
println("max (shifted) thrust single rotor: ",maximum(abs.(us)))
println("total thrust: ", norm(us))
println("time step goal reached: ", check_goal_reached([0.0; 0.0; 0.0; 0.0; 0.0; 0.0],xs))

println("Starting mpc sat 1...")
ts, xs, us = quad2d_mpc(5,1)
println("max (shifted) thrust single rotor: ",maximum(abs.(us)))
println("total thrust: ", norm(us))
println("time step goal reached: ", check_goal_reached([0.0; 0.0; 0.0; 0.0; 0.0; 0.0],xs))

println("Starting mpc sat 2...")
ts, xs, us = quad2d_mpc(5,2)
println("max (shifted) thrust single rotor: ",maximum(abs.(us)))
println("total thrust: ", norm(us))
println("time step goal reached: ", check_goal_reached([0.0; 0.0; 0.0; 0.0; 0.0; 0.0],xs))

#compare_quad2d_lqr_mpc()

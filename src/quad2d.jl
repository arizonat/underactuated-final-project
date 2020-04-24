using DynamicPolynomials, SumOfSquares, PolyJuMP, MosekTools
using Plots
using ForwardDiff: jacobian
using DifferentialEquations
using ControlSystems, LinearAlgebra

const g = 9.81
const m = 0.486
const r = 0.25
const I_z = 0.00383
const dim_x = 6
const dim_u = 2
const ϵ = 1e-3

const approx_hess_error_msg = "x₀ and x must have the same dimensions"

function quad2d!(derivatives, state, control, t)
    x, y, θ, ẋ, ẏ, θ̇ = state
    u₁, u₂ = control
    dz[1] = ẋ
    dz[2] = ẏ
    dz[3] = θ̇
    dz[4] = -(u₁ + u₂) * sin(θ) / m
    dz[5] = (u₁ + u₂) * cos(θ) / m - g
    dz[6] = (u₁ - u₂) * r / I_z
end

function quad2d(state, control, t)
    (x, y, θ, ẋ, ẏ, θ̇) = state
    (u₁, u₂) = control
    return [
        ẋ
        ẏ
        θ̇
        -(u₁ + u₂) * sin(θ) / m
        (u₁ + u₂) * cos(θ) / m - g
        (u₁ - u₂) * r / I_z
    ]
end

quad2d(state, control) = quad2d(state, control, 0)

function vector_hessian(f, x)
    n = length(x)
    out = jacobian(x -> jacobian(f, x), x)
    return out
end

function hessian(f, x₀::AbstractArray, x::AbstractArray)
    length(x) == length(x₀) || throw(ArgumentError(approx_hess_error_msg))
    n = length(x₀)
    return (1 / 2) * reshape(vector_hessian(f, x₀) * x, (n, n))'
end


function linearize(f, x_G, u_G)
    A = jacobian(x -> f(x, u_G), x_G)
    B = jacobian(u -> f(x_G, u), u_G)
    return A, B
end

function do_lqr(A, B, Q, R)
    S = care(A, B, Q, R)
    K = R \ B' * S
    return K, S
end

L2(x) = sum(y -> y .^ 2, x)

function line_search(ρ_init, f, min_step_size, init_step_size)
    ρ = ρ_init
    step_size = init_step_size
    if !(f(ρ))
        ρ = -1
    else
        while step_size > min_step_size
            if f(ρ)
                ρ += step_size
            else
                ρ -= step_size
                step_size /= 2

            end
        end
    end
    return ρ
end

function optimize_ρ(ρ, J★, J̇̂★, n)
    model =
        SOSModel(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    @polyvar x̅[1:n]
    #@variable model ρ
    X = monomials(x̅, 0:2)
    @variable model h SOSPoly(X)
    @constraint model h in SOSCone()
    @constraint model J̇̂★(x̅) + h * (ρ - J★(x̅)) <= -ϵ * (x̅' * x̅)
    #@objective(model, Max, ρ)
    optimize!(model)
    return model
end

function ρ_feasible_func(J★, J̇̂★, n)
    function f(ρ)
        model = optimize_ρ(ρ, J★, J̇̂★, n)
        termination_status(model) == MOI.TerminationStatusCode(1)
    end
    return f
end

function find_max_rho(f, x_G, u_G, Q, R)
    n = length(x_G)

    A, B = linearize(f, x_G, u_G)
    K, S = do_lqr(A, B, Q, R)

    J★(x̅) = x̅' * S * x̅
    f⁽ᶜˡ⁾(x̅) = f(x̅ + x_G, -K * (x̅ + x_G))

    𝕆 = zero(x_G)
    f̂⁽ᶜˡ⁾(x̅) = (jacobian(f⁽ᶜˡ⁾, 𝕆) .+ hessian(f⁽ᶜˡ⁾, 𝕆, x̅)) * x̅

    J̇̂★(x̅) = 2 * x̅' * S * f̂⁽ᶜˡ⁾(x̅)

    ρ_init = 0.01
    min_step_size = 0.0001
    init_step_size = 10.0

    ρ_feasible = ρ_feasible_func(J★, J̇̂★, n)
    ρ = line_search(ρ_init, ρ_feasible, min_step_size, init_step_size)
    return ρ, K, S
end

const Iᵣ = 10.0
const mᵣ = 10.0
const b = 0.01
const ℓ = 50.0

pendulum_eos(τ, θ, θ̇) = (τ .- mᵣ .* g .* sin.(θ)) ./ Iᵣ

function pendulum!(dx, x, u, t)
    θ, θ̇ = x
    τ = u[1]
    θ̈ = pendulum_eos(τ, θ, θ̇)
    dx[1] = θ̇
    dx[2] = θ̈

end

function pendulum(x, u, t)
    θ, θ̇ = x
    τ = u[1]
    θ̈ = pendulum_eos(τ, θ, θ̇)
    [θ̇; θ̈]
end

pendulum(x, τ) = pendulum(x, τ, 0)

# function main()
#     x_G = repeat([0.0], dim_x)
#     u_G = repeat([m * g / 2], dim_u)
#     Q = Diagonal([10, 10, 90, 1, 1, r / (2 * pi)])
#     R = [0.1 0.05; 0.05 0.1]
#     find_max_rho(quad2d, x_G, u_G, Q, R)
# end

function main()
    x_G = [π; 0.0]
    u_G = [0.0]
    Q = Diagonal([1.0, 10.0])
    R = reshape([100.0], (1, 1))
    ρ, K, S = find_max_rho(pendulum, x_G, u_G, Q, R)
    f_cl(x) = pendulum(x, -K * (x - x_G))

    tmin = -pi
    tmax = 3 * pi
    tdmin = -4.0
    tdmax = 4.0

    dt = 0.2
    dtd = 0.8
    dt_d = 0.05
    dtd_d = 0.05

    θ = collect(tmin:dt:tmax)
    θ̇ = collect(tdmin:dtd:tdmax)

    θd = collect(tmin:dt_d:tmax)
    θ̇d = collect(tdmin:dtd_d:tdmax)

    grid_θ = [i for i in θ, j in θ̇]
    grid_θ̇ = [j for i in θ, j in θ̇]
    grid_θ̈ = [
        f_cl([grid_θ[i, j]; grid_θ̇[i, j]])[2]
        for i in 1:length(θ), j in 1:length(θ̇)
    ]
    J(x) = ((x - x_G)'*S*(x - x_G))[1]
    grid_J = [J([i; j]) for i in θd, j in θ̇d]
    norm_grid = sqrt.(grid_θ̇ .^ 2 + grid_θ̈ .^ 2) * 10

    θlims = (minimum(θ), maximum(θ))
    θ̇lims = (minimum(θ̇), maximum(θ̇))
    theme(:juno)
    p = contour(
        θd,
        θ̇d,
        vec(grid_J),
        fill = false,
        xlims = θlims,
        ylims = θ̇lims,
    )
    if ρ != -1
        contour!(
            p,
            θd,
            θ̇d,
            vec(grid_J),
            fill = true,
            levels = [0, ρ, maximum(grid_J)],
            xlims = θlims,
            ylims = θ̇lims,
        )
    end
    contour!(
        p,
        θd,
        θ̇d,
        vec(grid_J),
        fill = false,
        xlims = θlims,
        ylims = θ̇lims,
    )

    cl_pendulum!(dx, x, p, t) = pendulum!(dx, x, -K * (x - x_G), t)
    for (θ₀, θ̇₀) in zip(vec(grid_θ), vec(grid_θ̇))
        u0 = [θ₀; θ̇₀]
        tspan = (0.0, 10.0)
        prob = ODEProblem(cl_pendulum!, u0, tspan)
        sol = DifferentialEquations.solve(prob)
        plot!(
            p,
            sol,
            plotdensity = 10000,
            vars = (1, 2),
            legend = false,
            xlims = θlims,
            ylims = θ̇lims,
        )
    end
    # quiver!(p,
    #     vec(grid_θ),
    #     vec(grid_θ̇),
    #     quiver = (vec(grid_θ̇ ./ norm_grid), vec(grid_θ̈ ./ norm_grid)),
    #     xlims = θlims,
    #     ylims = θ̇lims,
    # )
    display(p)
    println(ρ)
end

function test_sos_solver()
    @polyvar x y # variable with name y
    motzkin = x^4 * y^2 + x^2 * y^4 + 1 - 3 * x^2 * y^2
    model = SOSModel(Mosek.Optimizer)
    X = monomials([x, y], 0:2)
    @variable model p Poly(X)
    # p should be strictly positive
    @constraint model p >= 1
    @constraint model p * motzkin >= 0
    optimize!(model)
end

main()

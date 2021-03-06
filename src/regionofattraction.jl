
const FEASIBLE = MOI.TerminationStatusCode(1)

function line_search(ρ_init, f, min_step_size, init_step_size)
    ρ = ρ_init
    step_size = init_step_size
    if !(f(ρ))
        ρ = -1
    else
        prog = ProgressMeter.ProgressThresh(min_step_size, "Minimizing: ")
        while step_size > min_step_size
            if f(ρ)
                ρ += step_size
            else
                ρ -= step_size
                step_size /= 2

            end
            ProgressMeter.update!(prog, step_size)
        end
    end
    return ρ
end

function optimize_ρ(ρ, J★, J̇̂★, n, ϵ; polynomial_order = 2)
    model =
        SOSModel(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    @polyvar x̅[1:n]
    #@variable model ρ
    X = monomials(x̅, 0:polynomial_order)
    @variable model h SOSPoly(X)
    @constraint model h in SOSCone()
    @constraint model J̇̂★(x̅) + h * (ρ - J★(x̅)) <= -ϵ * (x̅' * x̅)
    #@objective(model, Max, ρ)
    optimize!(model)
    return model
end

ρ_feasible_func(opt, a...; b...) =
    ρ -> termination_status(opt(ρ, a...; b...)) == FEASIBLE

function find_max_rho(
    f̂,
    x_G,
    u_G,
    Q,
    R,
    ρ_init = 0.01,
    Δmin = 0.1,
    Δ₀ = 10.0,
    ϵ = 1e-3,
)
    n = length(x_G)
    polynomial_order = 3

    A, B = linearize(f̂, x_G, u_G)
    K, S = do_lqr(A, B, Q, R)

    J★(x̅) = x̅' * S * x̅
    f⁽ᶜˡ⁾(x̅) = f̂(x̅ + x_G, -K * (x̅ + x_G))

    #𝕆 = zero(x_G)
    #f̂⁽ᶜˡ⁾(x̅) = (jacobian(f⁽ᶜˡ⁾, 𝕆) .+ hessian(f⁽ᶜˡ⁾, 𝕆, x̅)) * x̅

    J̇̂★(x̅) = 2 * x̅' * S * f⁽ᶜˡ⁾(x̅)

    ρ_feasible = ρ_feasible_func(
        optimize_ρ,
        J★,
        J̇̂★,
        n,
        ϵ,
        polynomial_order = polynomial_order,
    )
    ρ = line_search(ρ_init, ρ_feasible, Δmin, Δ₀)
    return ρ, K, S
end

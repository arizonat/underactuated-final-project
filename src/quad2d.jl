const g = 9.81
const μ = 0.486
const r = 0.25
const I_z = 0.00383
const dim_x = 6
const dim_u = 2


function quad2d!(derivatives, state, control, t)
    x, y, θ, ẋ, ẏ, θ̇ = state
    u₁, u₂ = control(state, t)
    derivatives[1] = ẋ
    derivatives[2] = ẏ
    derivatives[3] = θ̇
    derivatives[4] = -(u₁ + u₂) * sin(θ) / μ
    derivatives[5] = (u₁ + u₂) * cos(θ) / μ - g
    derivatives[6] = (u₁ - u₂) * r / I_z
end

function quad2d(state, control, t)
    (x, y, θ, ẋ, ẏ, θ̇) = state
    (u₁, u₂) = control
    return [
        ẋ
        ẏ
        θ̇
        -(u₁ + u₂) * sin(θ) / μ
        (u₁ + u₂) * cos(θ) / μ - g
        (u₁ - u₂) * r / I_z
    ]
end

quad2d(state, control) = quad2d(state, control, 0)

function quad2d_approx!(derivatives, state, control, t)
    x, y, θ, ẋ, ẏ, θ̇ = state
    u₁, u₂ = control(state, t)
    derivatives[1] = ẋ
    derivatives[2] = ẏ
    derivatives[3] = θ̇
    derivatives[4] = -(μ * g + u₁ + u₂) * (θ - (1 // 6) * θ .^ 3) / μ
    derivatives[5] = (μ * g + u₁ + u₂) * (1 - (1 // 2) * θ .^ 2) / μ - g
    derivatives[6] = (u₁ - u₂) * r / I_z
end

function quad2d_approx(state, control, t)
    (x, y, θ, ẋ, ẏ, θ̇) = state
    (u₁, u₂) = control
    return [
        ẋ
        ẏ
        θ̇
        -(μ * g + u₁ + u₂) * (θ - (1 // 6) * θ .^ 3) / μ
        (μ * g + u₁ + u₂) * (1 - (1 // 2) * θ .^ 2) / μ - g
        (u₁ - u₂) * r / I_z
    ]
end

quad2d_approx(state, control) = quad2d(state, control, 0)

const prop_r = 0.1
const dx = r
const dy = 0.01
const prop_rod_width = 0.005
const prop_height = 0.025
const prop_thickness = 0.01

function quad2D_shape(x, y, θ)
    pts = [
        -dx -dy
        -dx dy
        -dx / 2 - prop_rod_width dy
        -dx / 2 - prop_rod_width dy + prop_height
        -dx / 2 - prop_r dy + prop_height + prop_thickness / 2
        -dx / 2 - prop_rod_width dy + prop_height + prop_thickness
        -dx / 2 + prop_rod_width dy + prop_height + prop_thickness
        -dx / 2 + prop_r dy + prop_height + prop_thickness / 2
        -dx / 2 + prop_rod_width dy + prop_height
        -dx / 2 + prop_rod_width dy
        dx / 2 - prop_rod_width dy
        dx / 2 - prop_rod_width dy + prop_height
        dx / 2 - prop_r dy + prop_height + prop_thickness / 2
        dx / 2 - prop_rod_width dy + prop_height + prop_thickness
        dx / 2 + prop_rod_width dy + prop_height + prop_thickness
        dx / 2 + prop_r dy + prop_height + prop_thickness / 2
        dx / 2 + prop_rod_width dy + prop_height
        dx / 2 + prop_rod_width dy
        dx dy
        dx -dy
        -dx -dy
    ]

    pts = pts * [cos(θ) sin(θ); -sin(θ) cos(θ)]

    pts .+= [x y]

    return Shape(pts[:, 1], pts[:, 2])

end

function plot_quad2D_frame(xs, idx)
    max_idx = size(xs)[2]
    x = xs[1, :]
    y = xs[2, :]
    θ = xs[3, idx]

    xlims = (minimum(x) - 1.0, maximum(x) + 1.0)
    ylims = (minimum(y) - 1.0, maximum(y) + 1.0)
    plt = plot(
        x[1:idx],
        y[1:idx],
        c = "red",
        xlim = xlims,
        ylim = ylims,
        legend = false,
    )
    if idx < max_idx
        plot!(plt, x[idx+1:end], y[idx+1:end], c = "blue", legend = false)
    end
    plot!(plt, quad2D_shape(x[idx], y[idx], θ), c = "black", legend = false)
    return plt
end

function plot_quad2D_animation(xs)
    plt = plot_quad2D_frame(xs, 1)
    anim = Animation()
    for t in 1:size(xs)[2]
        plot_quad2D_frame(xs, t)
        frame(anim)
    end
    return anim
end

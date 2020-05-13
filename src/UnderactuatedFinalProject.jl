module UnderactuatedFinalProject

using DynamicPolynomials, SumOfSquares, JuMP, PolyJuMP, MosekTools
using ForwardDiff: jacobian
using ControlSystems, LinearAlgebra
using Plots
using DifferentialEquations
using ProgressMeter



export find_max_rho
export mpc, linear_mpc, nonlinear_mpc_optimal_control
export quad2d,
    quad2d!,
    quad2d_approx,
    quad2d_approx!,
    quad2d_shifted,
    quad2d_shifted!,
    plot_quad2D_frame,
    plot_quad2D_animation
export do_lqr, linearize

include("linearization.jl")
include("regionofattraction.jl")
include("mpc.jl")
include("quad2d.jl")

end # module

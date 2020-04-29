module UnderactuatedFinalProject

using DynamicPolynomials, SumOfSquares, JuMP, PolyJuMP, MosekTools
using ForwardDiff: jacobian
using ControlSystems, LinearAlgebra
using Plots


export find_max_rho
export mpc, linear_mpc, nonlinear_mpc_optimal_control
export quad2d, plot_quad2D_frame, plot_quad2D_animation

include("linearization.jl")
include("regionofattraction.jl")
include("mpc.jl")
include("quad2d.jl")

end # module

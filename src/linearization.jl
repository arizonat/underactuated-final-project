using ForwardDiff: jacobian

const approx_hess_error_msg = "x₀ and x must have the same dimensions"

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

make_linear_system(f, x_G, u_G) = LinearSystem(linearize(f, x_G, u_G)...)

to_col(x::AbstractVector{T}) where {T} = reshape(x, (length(x), 1))
to_row(x::AbstractVector{T}) where {T} = reshape(x, (1, length(x)))


struct LinearSystem{T<:Real,M<:AbstractMatrix{T}}
    A::M
    B::M
    function LinearSystem{T,M}(
        A::M,
        B::M,
    ) where {T<:Real} where {M<:AbstractMatrix{T}}
        size(A)[1] == size(A)[2] || throw(ArgumentError("A must be square"))
        size(A)[1] == size(B)[1] ||
        throw(ArgumentError("A and B must have the same number of rows"))
        new{T,M}(A, B)
    end
end

LinearSystem(A <: AbstractMatrix{T}, B <: AbstractVector{T}) where {T<:Real} =
    LinearSystem(A, to_col(B))

(m::LinearSystem{T,M})(x::M, u::M) = m.A * x + m.B * u
(m::LinearSystem{T,M})(x::V{<:T}, u::V{<:T}) where {V<:AbstractVector} =
    m.A * x + m.B * u
(m::LinearSystem{T,M})(x, u, t) = m(x, u)

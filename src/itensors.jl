# TNQS-owned tensor utilities and the legacy `ITensors.jl`-style API, implemented over the
# next-gen `ITensorBase` / `TensorAlgebra` / `MatrixAlgebraKit` stack.

import Base: truncate
import ITensorBase: scalartype, uniqueinds
import MatrixAlgebraKit as MAK
import TensorAlgebra: matricize
using Adapt: Adapt
using ITensorBase: ITensorBase, AbstractITensor, ITensor, Index, NamedUnitRange, commoninds,
    dimnames, hascommoninds, id, inds, name, nameddims, noncommoninds, noprime, plev, prime,
    replacedimnames, sim, tags, unioninds, unnamed
using LinearAlgebra: LinearAlgebra
using TensorAlgebra: TensorAlgebra, project, scalar, tryproject

function project_aux(v::AbstractVector{<:Number}, i::Index)
    length(v) == length(i) ||
        error(
        "state vector has dimension $(length(v)) but the site index has dimension $(length(i))"
    )
    ψ = tryproject(v, (i,))
    isnothing(ψ) || return ψ
    raw = project(reshape(v, (length(v), 1)), (unnamed(i),), ())
    aux = Index(TensorAlgebra.axes(raw, 2))
    return nameddims(raw, (ITensorBase.name(i), ITensorBase.name(aux)))
end

function onehot(eltype::Type, (i, p)::Pair{<:Index})
    v = zeros(eltype, length(i))
    v[p] = one(eltype)
    return project_aux(v, i)
end
onehot(p::Pair{<:Index}) = onehot(Float64, p)

function inner end

function contract end
function contract(tensors::AbstractVector; sequence = nothing)
    return isnothing(sequence) ? reduce(*, tensors) : _contract_sequence(tensors, sequence)
end
_contract_sequence(tensors, s::Integer) = tensors[s]
_contract_sequence(tensors, s) = reduce(*, (_contract_sequence(tensors, x) for x in s))

# The single shared/unique index, or `nothing` when there is not exactly one. `commoninds`
# and `uniqueinds` come from ITensorBase, which keys index-set algebra by name so a graded
# bond still matches its dual.
commonind(a, b) = (cs = commoninds(a, b); isempty(cs) ? nothing : first(cs))
noncommonind(a, b) = (us = uniqueinds(a, b); isempty(us) ? nothing : first(us))

replaceind(t, p::Pair) = replaceinds(t, p)
replaceind(t, from::Index, to::Index) = replaceinds(t, from => to)

const _IndexColl = Union{Tuple{Vararg{Index}}, AbstractVector{<:Index}}
function replaceinds(t, pairs::Pair...)
    return replacedimnames(t, map(p -> name(first(p)) => name(last(p)), pairs)...)
end
function replaceinds(t, from::_IndexColl, to::_IndexColl)
    return replaceinds(t, map(=>, from, to)...)
end
function replaceinds(t::AbstractITensor, p::Pair{<:_IndexColl, <:_IndexColl})
    return replaceinds(t, first(p), last(p))
end

diaglength(a::AbstractArray) = minimum(size(a))
function diagstride(a::AbstractArray)
    s = 1
    p = 1
    for i in 1:(ndims(a) - 1)
        p *= size(a, i)
        s += p
    end
    return s
end
function diagindices(a::AbstractArray)
    maxdiag = LinearIndices(a)[CartesianIndex(ntuple(Returns(diaglength(a)), ndims(a)))]
    return 1:diagstride(a):maxdiag
end
diagview(a::AbstractArray) = @view a[diagindices(a)]

function diagonaltensor(diag::AbstractVector, ax::Tuple{Vararg{AbstractUnitRange}})
    a = similar(diag, ax)
    fill!(a, zero(eltype(a)))
    diagview(a) .= diag
    return a
end
function diagonaltensor(
        diag::AbstractVector,
        is::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return nameddims(diagonaltensor(diag, unnamed.(is)), name.(is))
end

function similar_map(prototype::AbstractITensor, eltype::Type, codomain, domain)
    raw = TensorAlgebra.similar_map(
        unnamed(prototype), eltype, unnamed.(codomain), unnamed.(domain)
    )
    return nameddims(raw, (name.(codomain)..., name.(domain)...))
end
function similar_map(prototype::AbstractITensor, codomain, domain)
    return similar_map(prototype, scalartype(prototype), codomain, domain)
end

delta(eltype::Type, is::Tuple) = diagonaltensor(ones(eltype, minimum(length, is)), is)
delta(eltype::Type, is::Index...) = delta(eltype, is)
delta(eltype::Type, is::AbstractVector{<:Index}) = delta(eltype, Tuple(is))
delta(is::Tuple) = delta(Float64, is)
delta(is::Index...) = delta(Float64, is)
delta(is::AbstractVector{<:Index}) = delta(Float64, Tuple(is))

# The codomain/domain bipartition of an operator tensor: each plev-0 index paired with its
# prime. Viewing the operator as this square map is what `tr` and `eigen` factor through.
function operator_inds(a::AbstractITensor)
    domain = filter(i -> plev(i) == 0, inds(a))
    return prime.(domain), domain
end

apply(o::AbstractITensor, ψ::AbstractITensor) = noprime(o * ψ)

function eigen(m::AbstractITensor, codomain, domain; ishermitian = false, cutoff = nothing)
    ishermitian ||
        error("the compat `eigen` only supports the hermitian case (ishermitian = true)")
    isnothing(cutoff) || error(
        "the compat `eigen` does not yet translate the `cutoff` truncation kwarg to MatrixAlgebraKit's `trunc` spec"
    )
    D, U = MAK.eigh_full(m, codomain, domain)
    u = only(commoninds(D, U))
    t = only(uniqueinds(D, U))
    D = replaceinds(D, t => ITensorBase.prime(u))
    return D, U
end

function eigen(m::AbstractITensor; kwargs...)
    codomain, domain = operator_inds(m)
    D, U = eigen(m, codomain, domain; kwargs...)
    return D, conj(U)
end

function itensor_trunc(; maxdim = nothing, cutoff = nothing)
    trunc = MAK.notrunc()
    isnothing(maxdim) || (trunc &= MAK.truncrank(maxdim))
    isnothing(cutoff) || iszero(cutoff) || (trunc &= MAK.truncerror(; rtol = sqrt(cutoff), p = 2))
    return trunc
end

function factorize(
        a::AbstractITensor,
        codomain;
        ortho = "left",
        cutoff = nothing,
        maxdim = nothing,
        tags = nothing
    )
    # `left_orth` / `right_orth` take the codomain indices and infer the domain, and `trunc`
    # covers both the exact (no cutoff/maxdim) and truncating cases.
    trunc = itensor_trunc(; cutoff, maxdim)
    if ortho == "left"
        L, R = MAK.left_orth(a, codomain; trunc)
    elseif ortho == "right"
        L, R = MAK.right_orth(a, codomain; trunc)
    else
        error("compat `factorize` supports ortho = \"left\" / \"right\" (got $(repr(ortho)))")
    end
    if !isnothing(tags)
        b = only(commoninds(L, R))
        bnew = settags(b, tags)
        L, R = replaceind(L, b, bnew), replaceind(R, b, bnew)
    end
    return L, R
end

datatype(T::AbstractITensor) = typeof(unnamed(T))
array(T::AbstractITensor) = convert(Array, unnamed(T))
data(T::AbstractITensor) = unnamed(T)

struct ScalarTypeAdaptor{T} end
ScalarTypeAdaptor(T::Type) = ScalarTypeAdaptor{T}()
adapt_scalartype(T::Type) = Adapt.adapt(ScalarTypeAdaptor(T))
adapt_scalartype(T::Type, x) = Adapt.adapt(ScalarTypeAdaptor(T), x)
function Adapt.adapt_structure(::ScalarTypeAdaptor{elt}, T::AbstractITensor) where {elt}
    eltype(T) === elt && return T
    return nameddims(convert(AbstractArray{elt}, unnamed(T)), ITensorBase.dimnames(T))
end

swapind(T::AbstractITensor, i::Index, j::Index) = replaceinds(T, i => j, j => i)

hasqns(i::Index) = conj(unnamed(i)) != unnamed(i)
hasqns(t::AbstractITensor) = any(hasqns, inds(t))
hasqns(::Any) = false

function directsum(out_inds, pairs::Pair...)
    out_inds = Tuple(out_inds)
    t1, s1 = first(pairs[1]), Tuple(last(pairs[1]))
    shared = Tuple(filter(i -> !(i in s1), collect(inds(t1))))
    target = (shared..., out_inds...)
    out = zeros(scalartype(t1), length.(target))
    offsets = zeros(Int, length(out_inds))
    for p in pairs
        t, sinds = first(p), Tuple(last(p))
        order = (shared..., sinds...)
        cur = collect(ITensorBase.dimnames(t))
        perm = [findfirst(==(name(o)), cur) for o in order]
        a = permutedims(unnamed(t), perm)
        ranges = (
            Base.OneTo.(length.(shared))...,
            ntuple(k -> (offsets[k] + 1):(offsets[k] + length(sinds[k])), length(sinds))...,
        )
        out[ranges...] .= a
        offsets .+= length.(sinds)
    end
    return out[target...]
end

struct Algorithm{Alg, Kwargs <: NamedTuple}
    kwargs::Kwargs
end
Algorithm{Alg}(kwargs::NamedTuple) where {Alg} = Algorithm{Alg, typeof(kwargs)}(kwargs)
Algorithm{Alg}(; kwargs...) where {Alg} = Algorithm{Alg}((; kwargs...))
Algorithm(alg::Symbol; kwargs...) = Algorithm{alg}(; kwargs...)
Algorithm(alg::AbstractString; kwargs...) = Algorithm(Symbol(alg); kwargs...)
Algorithm(alg::Algorithm) = alg
function Base.getproperty(alg::Algorithm, name::Symbol)
    return if name === :kwargs
        getfield(alg, :kwargs)
    else
        getfield(getfield(alg, :kwargs), name)
    end
end
macro Algorithm_str(s)
    return :(Algorithm{$(Expr(:quote, Symbol(s)))})
end

function settags(i::Index, tagstr::AbstractString)
    for t in split(tagstr, ",")
        s = String(strip(t))
        isempty(s) || (i = ITensorBase.settag(i, s, ""))
    end
    return i
end
function settags(i::Index, d::AbstractDict)
    for (k, v) in d
        i = ITensorBase.settag(i, k, v)
    end
    return i
end
function hastags(i::Index, tagstr::AbstractString)
    return all(
        haskey(tags(i), String(strip(t))) for t in split(tagstr, ",") if !isempty(strip(t))
    )
end

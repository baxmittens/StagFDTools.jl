struct Allocs{t,n,p,M,NT,DC,DV,G}
    type::t
    number::n
    pattern::p
    M::M
    𝐊::ExtendableSparseMatrix
    𝐐::ExtendableSparseMatrix
    𝐐ᵀ::ExtendableSparseMatrix
    𝐏::ExtendableSparseMatrix
    dx::Vector{Float64}
    r::Vector{Float64}
    R::NT
    V::NT
    Vi::NT
    η::NT
    λ̇::NT
    ε̇::NT
    τ0::NT
    τ::NT
    ξ::NT
    G::NT
    β::NT
    ρ::NT
    Pt::Matrix{Float64}
    Pti::Matrix{Float64}
    Pt0::Matrix{Float64}
    ΔPt::NT
    Dc::DC
    Dv::DV
    𝐷::NT
    D_ctl_c::DC
    D_ctl_v::DV
    𝐷_ctl::NT
    phases::NT
    Grid::G
end

function Allocs(nc, config, x, y, Δ)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    type = Fields(
        fill(:out, (nc.x + 3, nc.y + 4)),
        fill(:out, (nc.x + 4, nc.y + 3)),
        fill(:out, (nc.x + 2, nc.y + 2)),
    )
    set_boundaries_template!(type, config, nc)

    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]), @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])),
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1; 1 1 1]), @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0; 0 1 0]), @SMatrix([0 0; 1 1; 0 0]), @SMatrix([1]))
    )

    nVx = maximum(number.Vx)
    nVy = maximum(number.Vy)
    nPt = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)),
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)),
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊 = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐 = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏 = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r = zeros(nVx + nVy + nPt)

    R = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V = (x=zeros(size_x...), y=zeros(size_y...))
    Vi = (x=zeros(size_x...), y=zeros(size_y...))
    η = (c=ones(size_c...), v=ones(size_v...))
    λ̇ = (c=zeros(size_c...), v=zeros(size_v...))
    ε̇ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    τ0 = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...))
    τ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...), θ=zeros(size_c...))
    ξ = (c=ones(size_c...), v=ones(size_v...))
    G = (c=zeros(size_c...), v=zeros(size_v...))
    β = (c=zeros(size_c...), v=zeros(size_v...))
    ρ = (c=zeros(size_c...), v=zeros(size_v...))

    Pt = zeros(size_c...)
    Pti = zeros(size_c...)
    Pt0 = zeros(size_c...)
    ΔPt = (c=zeros(size_c...), Vx=zeros(size_x...), Vy=zeros(size_y...))

    Dc = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    Dv = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷 = (c=Dc, v=Dv)
    D_ctl_c = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    D_ctl_v = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    𝐷_ctl = (c=D_ctl_c, v=D_ctl_v)
    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))

    # Mesh coordinates
    Grid = GenerateGrid(x, y, Δ, nc)

    return Allocs(
        type, number, pattern, M,
        𝐊, 𝐐, 𝐐ᵀ, 𝐏,
        dx, r,
        R, V, Vi, η, λ̇, ε̇, τ0, τ, ξ, G, β, ρ,
        Pt, Pti, Pt0, ΔPt,
        Dc, Dv, 𝐷, D_ctl_c, D_ctl_v, 𝐷_ctl,
        phases, Grid
    )
end
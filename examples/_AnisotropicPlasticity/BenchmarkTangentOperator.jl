using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, StaticArrays, LinearAlgebra
using BenchmarkTools

@views function benchmark_tangent_operator(nc)
    #--------------------------------------------#

    # Boundary loading type
    config = :free_slip
    ε̇bg = 5e-11
    D_BC = @SMatrix([ -ε̇bg 0.; 0  ε̇bg ])

    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; plasticity=DruckerPrager, compressible=true)
    
    params_bg = (ρ=1.0, n=1.0, η0=2e50, G=1.0, C=1.74e-4, ϕ=30., ηvp=2e3, β=0.5, ψ=10., ε̇=5e-11, rad=25e-4)
    params_in = (ρ=1.0, n=1.0, η0=2e50, G=0.25, C=1.74e-4, ϕ=30., ηvp=2e3, β=0.5, ψ=10.)

    materials.g .= [0., 0.]
    materials.ρ .= [params_bg.ρ, params_in.ρ]
    materials.n .= [params_bg.n, params_in.n]
    materials.η0 .= [params_bg.η0, params_in.η0]
    materials.G .= [params_bg.G, params_in.G]
    materials.β .= [params_bg.β, params_in.β]
    materials.plasticity.C .= [params_bg.C, params_in.C]
    materials.plasticity.ϕ .= [params_bg.ϕ, params_in.ϕ]
    materials.plasticity.ηvp .= [params_bg.ηvp, params_in.ηvp]
    materials.plasticity.ψ .= [params_bg.ψ, params_in.ψ]
    preprocess!(materials)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Boundary conditions
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    set_boundaries_template!(type, config, nc)

    # Equation numbering
    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    # Initialize field
    L = (x=1.0, y=0.7)
    x_bounds = (min=-L.x/2, max=L.x/2)
    y_bounds = (min=-L.y/2, max=L.y/2)
    Δ = (x=L.x/nc.x, y=L.y/nc.y, t=1e5)
    Grid = GenerateGrid(x_bounds, y_bounds, Δ, nc)

    # Allocations
    V = (x=zeros(size_x...), y=zeros(size_y...))
    η = (c=ones(size_c...), v=ones(size_v...))
    λ̇ = (c=zeros(size_c...), v=zeros(size_v...))
    ε̇ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    τ0 = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...))
    τ = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...), II=zeros(size_c...))
    G = (c=zeros(size_c...), v=zeros(size_v...))
    β_field = (c=zeros(size_c...), v=zeros(size_v...))
    ρ = (c=zeros(size_c...), v=zeros(size_v...))
    Pt = zeros(size_c...)
    Pt0 = zeros(size_c...)
    ΔPt = (c=zeros(size_c...), Vx=zeros(size_x...), Vy=zeros(size_y...))

    Dc = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    Dv = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    D = (c=Dc, v=Dv)
    D_ctl_c = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xx, 1), _ in axes(ε̇.xx, 2)]
    D_ctl_v = [@MMatrix(zeros(4, 4)) for _ in axes(ε̇.xy, 1), _ in axes(ε̇.xy, 2)]
    D_ctl = (c=D_ctl_c, v=D_ctl_v)

    phases = (c=ones(Int64, size_c...), v=ones(Int64, size_v...))

    # Initial velocity & pressure field
    @views V.x[inx_Vx, iny_Vx] .= D_BC[1, 1] * Grid.v.x .+ D_BC[1, 2] * Grid.c.x'
    @views V.y[inx_Vy, iny_Vy] .= D_BC[2, 1] * Grid.c.x .+ D_BC[2, 2] * Grid.v.y'
    @views Pt[inx_c, iny_c] .= 0.0

    # Boundary conditions
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx] .= (type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1] * Grid.v.x .+ D_BC[1, 2] * Grid.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1] * Grid.v.y .+ D_BC[1, 2] * Grid.v.y[end])
        BC.Vy[inx_Vy, 2] .= (type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * Grid.v.x[1] .+ D_BC[2, 2] * Grid.v.y)
        BC.Vy[end-1, iny_Vy] .= (type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1] * Grid.v.x[end] .+ D_BC[2, 2] * Grid.v.y)
    end

    # Material geometry
    ccord = (x=-L.x/2, y=-L.y/2)
    @views phases.c[inx_c, iny_c][((Grid.c.x .- ccord.x).^2 .+ ((Grid.c.y') .- ccord.y).^2) .<= (25e-4)] .= 2
    @views phases.v[inx_v, iny_v][((Grid.v.x .- ccord.x).^2 .+ ((Grid.v.y') .- ccord.y).^2) .<= (25e-4)] .= 2
    phase_ratios = InitialisePhaseRatios(phases, nphases)

    # Pre-compute grid fields
    compute_grid_fields!(G, β_field, ρ, (c=ones(size_c...), v=ones(size_v...)), materials, phase_ratios, nc, size_c, size_v, nphases)

    # Warmup
    TangentOperator!(D, D_ctl, τ, τ0, ε̇, λ̇, η, G, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)

    # Benchmark
    println("\n" * "="^80)
    println("Benchmarking TangentOperator! with grid size: $(nc.x) × $(nc.y)")
    println("="^80 * "\n")
    
    bench = @benchmark TangentOperator!($D, $D_ctl, $τ, $τ0, $ε̇, $λ̇, $η, $G, $V, $Pt, $Pt0, $ΔPt, $type, $BC, $materials, $phase_ratios, $Δ)
    
    println(bench)
    println("\n" * "="^80)
    println("Summary:")
    println("  Mean time:    $(round(mean(bench.times)/1e6, digits=3)) ms")
    println("  Std dev:      $(round(std(bench.times)/1e6, digits=3)) ms")
    println("  Min time:     $(round(minimum(bench.times)/1e6, digits=3)) ms")
    println("  Max time:     $(round(maximum(bench.times)/1e6, digits=3)) ms")
    println("  Allocations:  $(bench.allocs) per call")
    println("  Memory:       $(round(bench.memory/1024, digits=2)) KB per call")
    println("="^80 * "\n")

    return bench
end

# Run benchmarks for different grid sizes
let
    resolutions = [(x=32, y=32), (x=64, y=64), (x=128, y=128)]
    
    for nc in resolutions
        benchmark_tangent_operator(nc)
        GC.collect()
    end
end

using StagFDTools, StagFDTools.StokesJustPIC, StagFDTools.Rheology
using JLD2, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using TimerOutputs
using ExactFieldSolutions

# ------------------------------------------------------------------------------------------------------
# Anisotropic VEP 
# ------------------------------------------------------------------------------------------------------

# Main function
function main(res)

    # --------------------------------------------------------------------------------------------------- #
    # Initialise

    # Scaling
    sc = (t = 1., σ = 1., L = 1.)

    # Resolution
    nc = (x=res, y=res)
    nmpc = (x = 4, y = 4)

    # Parameters
    params_bg = (ρ=1.0, g=0., n=1.0, η0=2e50, G=1.0, C=1.74e-4, ϕ=30., ηvp=1., β=0.5, ψ=10., ε̇=5e-11, rad=25e-4)
    params_in = (ρ=1.0, g=0., n=1.0, η0=2e50, G=1.0, C=1.74e-4, ϕ=30., ηvp=1., β=0.5, ψ=10.)

    # Process material material properties 
    materials_pr = initialize_materials(2; plasticity=DruckerPrager, compressible=true)
    materials_pr.ρ .= [params_bg.ρ, params_in.ρ]
    materials_pr.g .= [params_bg.g, params_in.g]
    materials_pr.n .= [params_bg.n, params_in.n]
    materials_pr.η0 .= [params_bg.η0, params_in.η0]    
    materials_pr.G .= [params_bg.G, params_in.G]
    materials_pr.C .= [params_bg.C, params_in.C]
    materials_pr.ψ .= [params_bg.ψ, params_in.ψ]
    materials_pr.ηvp .= [params_bg.ηvp, params_in.ηvp]
    materials_pr.β .= [params_bg.β, params_in.β]
    materials_pr.ϕ .= [params_bg.ϕ, params_in.ϕ]
    materials = preprocess_materials(materials_pr)

    # Markers 
    noise = true

    # Boundary conditions
    config = :free_slip
    D_BC = @SMatrix([params_bg.ε̇ 0.;
                     0. params_bg.ε̇])
    ε̇_bulk = D_BC[1]

    # Time stepping
    nt = 100
    Δt0 = 1e5

    # Newton Solver 
    niter = 5
    ϵ_nl  = 1e-10
    α     = LinRange(0.05, 1.0, 10)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # --------------------------------------------------------------------------------------------------- #
    # Numbering and assembling

    # Define node types and set BC flags
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

    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏  = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r  = zeros(nVx + nVy + nPt)

    # --------------------------------------------------------------------------------------------------- #
    # Initialisation

    # Intialise field
    L   = (x=1., y=1.)
    x   = (min=-L.x/2, max=L.x/2)
    y   = (min=-L.y/2, max=L.y/2)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    ξ       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    ΔPt     = (c=zeros(size_c...), Vx = zeros(size_x...), Vy = zeros(size_y...))

    Dc      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    # --------------------------------------------------------------------------------------------------- #
    # Grid, BC

    # Mesh coordinates
    X = GenerateGrid(x, y, Δ, nc)

    # Initial velocity & pressure field
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'
    Pt[inx_c, iny_c ]  .= 0.                 
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[1]   .+ D_BC[2,2]*X.v.y)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[end] .+ D_BC[2,2]*X.v.y)

    # --------------------------------------------------------------------------------------------------- #
    # Markers

    # Initialise marker field
    m = InitialiseParticleField(nc, nmpc, L, Δ, materials, noise)
    phase_ratios, phase_weights = InitialisePhaseRatios(m, ε̇)
    mphase = ones(Int64,m.num...)

    # Set phase geometry
    ccord = (x=x.min, y=y.max)
    mphase[((m.Xm .-ccord.x).^2 .+ (m.Ym .-ccord.y).^2) .<= params_bg.rad] .=2

    # Set phase ratios
    PhaseRatios!(phase_ratios, phase_weights, m, mphase, X.c_e.x, X.c_e.y, X.v_e.x, X.v_e.y, Δ)
    # Check sum 
    for I in CartesianIndices(phase_ratios.center)
        s = sum(phase_ratios.center[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.center at $I: sum = $s, values = $(phase_ratios.center[I])"
        end
    end

    # Cut ghost cells
    phase_ratios = (
        center   = phase_ratios.center[2:end-1,2:end-1],
        vertex   = phase_ratios.vertex[2:end-1,2:end-1],
    )

    # --------------------------------------------------------------------------------------------------- #
    # Pre-processing
    # rvec = zeros(length(α))
    # err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    # to   = TimerOutput()

    # --------------------------------------------------------------------------------------------------- #
    # Main loop 
    for it=1:nt

        @printf("Step %04d\n", it)

        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        # Compute bulk and shear moduli
        compute_shear_bulk_moduli!(G, β, materials, phase_ratios, nc, size_c, size_v, m.nphases)

        for iter=1:niter

            @printf("Iteration %04d\n", iter)
            
            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, β, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
                @show extrema(λ̇.c)
                @show extrema(λ̇.v)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(R.p[inx_c,iny_c])/sqrt(nPt)

            @show  max(err.x[iter], err.y[iter], err.p[iter])

            max(err.x[iter], err.y[iter]) < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt
            
            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = @views -r[1:size(𝐊,1)]
            fp   = @views -r[size(𝐊,1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            @views dx[1:size(𝐊,1)]     .= u
            @views dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, β, V, Pt, Pt0, ΔPt, type, BC, materials, phase_ratios, Δ)
        end
    end

end

let 
    main(101)
    # @time main(101)
end
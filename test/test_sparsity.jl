using Test, StaticArrays, LinearAlgebra, SparseArrays, ExtendableSparse
import StagFDTools.Poisson
import StagFDTools.Stokes

@testset "Sparsity patterns" verbose=true begin

    @testset "Poisson" begin
        nc = (x = 3, y = 4)

        type = Poisson.Fields( fill(:out, (nc.x+2, nc.y+2)) )
        type.u[2:end-1,2:end-1] .= :in
        type.u[:,1]             .= :Dirichlet
        type.u[:,end]           .= :Neumann

        number = Poisson.Fields( fill(0, (nc.x+2, nc.y+2)) )
        Poisson.Numbering!(number, type, nc)

        nu = maximum(number.u)
        @test nu == nc.x*nc.y

        # 5-point stencil: symmetric pattern with a full diagonal
        pattern = Poisson.Fields( Poisson.Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )
        M = Poisson.Fields( Poisson.Fields( ExtendableSparseMatrix(nu, nu) ))
        Poisson.SparsityPattern!(M, number, pattern, nc)
        A = sparse(M.u.u)
        @test norm(A - A') == 0.0
        @test all(diag(A) .== 1.0)
        # Interior rows couple to at most 5 unknowns
        @test maximum(sum(A .!= 0, dims=2)) <= 5

        # 9-point stencil strictly contains the 5-point one
        pattern9 = Poisson.Fields( Poisson.Fields( @SMatrix([1 1 1; 1 1 1; 1 1 1]) ) )
        M9 = Poisson.Fields( Poisson.Fields( ExtendableSparseMatrix(nu, nu) ))
        Poisson.SparsityPattern!(M9, number, pattern9, nc)
        A9 = sparse(M9.u.u)
        @test norm(A9 - A9') == 0.0
        @test nnz(A9) > nnz(A)
        @test all(A9[findall(!iszero, A)] .== 1.0)
    end

    @testset "Stokes" begin
        nc = (x = 4, y = 3)

        type = Stokes.Fields(
            fill(:out, (nc.x+3, nc.y+4)),
            fill(:out, (nc.x+4, nc.y+3)),
            fill(:out, (nc.x+2, nc.y+2)),
        )
        # -------- Vx -------- #
        type.Vx[2:end-1,3:end-2] .= :in
        type.Vx[2,2:end-1]       .= :Dirichlet_normal
        type.Vx[end-1,2:1:end-1] .= :Dirichlet_normal
        type.Vx[2:end-1,2]       .= :Dirichlet
        type.Vx[2:end-1,end-1]   .= :Dirichlet
        # -------- Vy -------- #
        type.Vy[2:end-2,2:end-1] .= :in
        type.Vy[2,2:end-1]       .= :Dirichlet
        type.Vy[end-1,2:end-1]   .= :Dirichlet
        type.Vy[2:end-1,2]       .= :Dirichlet_normal
        type.Vy[2:end-1,end-1]   .= :Dirichlet_normal
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in

        pattern = Stokes.Fields(
            Stokes.Fields(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0;  0 1 0])),
            Stokes.Fields(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0])),
            Stokes.Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
        )

        number = Stokes.Fields(
            fill(0, (nc.x+3, nc.y+4)),
            fill(0, (nc.x+4, nc.y+3)),
            fill(0, (nc.x+2, nc.y+2)),
        )
        Stokes.Numbering!(number, type, nc)

        nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
        @test nPt == nc.x*nc.y
        @test nVx > 0 && nVy > 0

        M = Stokes.Fields(
            Stokes.Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)),
            Stokes.Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)),
            Stokes.Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
        )
        Stokes.SparsityPattern!(M, number, pattern, nc)

        # Velocity block pattern is symmetric
        K = [sparse(M.Vx.Vx) sparse(M.Vx.Vy); sparse(M.Vy.Vx) sparse(M.Vy.Vy)]
        @test norm(K - K') == 0.0
        @test all(diag(K) .== 1.0)

        # Gradient and divergence patterns are transposes of each other
        Q  = [sparse(M.Vx.Pt); sparse(M.Vy.Pt)]
        Qᵀ = [sparse(M.Pt.Vx) sparse(M.Pt.Vy)]
        @test norm(Q' - Qᵀ) == 0.0
    end

end

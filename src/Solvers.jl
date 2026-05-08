using SparseArrays

# @views function KSP_GCR_Stokes!( x, M, b, noisy, Kuu, Kup, Kpu, Kpp; ηb=1e3, ϵ_l=1e-9, restart = 25
#  )

#     if nnz(Kpp) == 0 # incompressible limit
#         𝐏inv  = ηb .* I(size(Kpp,1))
#     else # compressible case
#         𝐏inv  = spdiagm(1.0 ./diag(Kpp))
#     end

#     # KSP GCR solver
#     norm_r, norm0 = 0.0, 0.0
#     N         = length(x)
#     maxit     = 1000
#     ncyc, its = 0, 0
#     i1, i2, success=0,0,0
#     # Arrays for coupled problem
#     f      = zeros(Float64, N)
#     v      = zeros(Float64, N)
#     s      = zeros(Float64, N)
#     val    = zeros(Float64, restart)
#     VV     = zeros(Float64, (restart,N))
#     SS     = zeros(Float64, (restart,N))
#     # Coupled
#     # Initial residual
#     f      = b - M*x 
#     norm_r = norm(f)
#     norm0  = norm_r;
#     #
#     ndofu = size(Kup,1)
#     ndofp = size(Kup,2)
#     Kuusc = Kuu - Kup*(𝐏inv*Kpu) # OK
#     PC    =  0.5*(Kuusc + Kuusc') 
#     t = @elapsed Kf    = cholesky(Hermitian(PC),check = false)
#     @printf("Cholesky took = %02.2e s\n", t)
#     # Arrays for decoupled problem
#     su    = zeros(Float64, ndofu)
#     fusc  = zeros(Float64, ndofu)
#     sp    = zeros(Float64, ndofp)
#     fu    = zeros(Float64, ndofu)
#     fp    = zeros(Float64, ndofp)
#     fu     .= f[1:ndofu]
#     fp     .= f[ndofu+1:end]
#     if (noisy > 1) @printf("       %1.4d KSP GCR Residual %1.12e %1.12e\n", 0, norm_r, norm_r/norm0); end
#     # Solving procedure
#     while ( success == 0 && its<maxit ) 
#         for i1=1:restart
#             # Apply preconditioner, s = PC^{-1} f
#             # s = PC\f
#             fusc .= fu  - Kup*(𝐏inv*fp + sp)
#             su   .= Kf\fusc
#             sp   .+= 𝐏inv*(fp - Kpu*su)
#             s[1:ndofu]     .= su
#             s[ndofu+1:end] .= sp
#             # Action of Jacobian on s: v = J*s
#             # JacobianAction!(v, M, s; r,kv,T,fc,TW,TE,dx,n)
#             v .= M*s
#             # Approximation of the Jv product
#             for i2=1:i1
#                 val[i2] = v' * VV[i2,:]
#             end
#             # Scaling
#             for i2=1:i1
#                 v .-= val[i2] * VV[i2,:]
#                 s .-= val[i2] * SS[i2,:]
#             end
#             # -----------------
#             r_dot_v = f'*v
#             nrm     = norm(v)
#             r_dot_v = r_dot_v / nrm
#             # -----------------
#             fact    = 1.0/nrm
#             v     .*= fact
#             s     .*= fact
#             # -----------------
#             fact    = r_dot_v;
#             x     .+= fact*s
#             f     .-= fact*v
#             # -----------------
#             norm_r  = norm(f) 
#             fu     .= f[1:ndofu]
#             fp     .= f[ndofu+1:end]
#             @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", its, norm(fu)/sqrt(length(fu)), norm(fp)/sqrt(length(fp)))
#             if norm(fu)/(length(fu)) < ϵ_l && norm(fp)/(length(fu)) < ϵ_l #(norm_r < eps * norm0 )
#                 success = 1
#                 println("converged")
#                 break
#             end
#             # Store 
#             VV[i1,:] .= v
#             SS[i1,:] .= s
#             its              += 1
#         end
#         its  += 1
#         ncyc += 1
#     end
#     if (noisy>1) @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
#     return its
# end

function KSP_GCR_Stokes!(
    x, M, b, noisy, Kuu, Kup, Kpu, Kpp;
    ηb      = 1e3, ϵ_l     = 1e-9, restart = 25, maxit   = 1000
)

    @views begin

        Kuu = sparse(Kuu)
        Kup = sparse(Kup)
        Kpu = sparse(Kpu)
        Kpp = sparse(Kpp)
        M   = sparse(M)

        ndofu = size(Kup,1)
        ndofp = size(Kup,2)
        N     = length(x)

        Pinv = nnz(Kpp) == 0 ? fill(ηb, ndofp) : 1.0 ./ diag(Kpp)

        Kuusc = Kuu - Kup * spdiagm(Pinv) * Kpu
        Kf    = cholesky(Hermitian(Kuusc), check=false)

        f = similar(x)
        s = similar(x)
        v = similar(x)

        mul!(f, M, x)
        @. f = b - f

        norm0 = norm(f)

        fu = f[1:ndofu]
        fp = f[ndofu+1:end]

        su = s[1:ndofu]
        sp = s[ndofu+1:end]

        VV  = zeros(eltype(x), N, restart)
        SS  = zeros(eltype(x), N, restart)

        tmpu = zeros(eltype(x), ndofu)
        tmpp = zeros(eltype(x), ndofp)
        fusc = zeros(eltype(x), ndofu)

        its = 0

        while its < maxit

            for k = 1:restart

                fill!(s, 0.0)
                @. tmpp = Pinv * fp

                mul!(tmpu, Kup, tmpp)
                @. fusc = fu - tmpu

                ldiv!(su, Kf, fusc)

                mul!(tmpp, Kpu, su)

                @. sp += Pinv * (fp - tmpp)

                mul!(v, M, s)   

                for j = 1:k-1
                    hj = dot(v, VV[:,j])
                    BLAS.axpy!(-hj, VV[:,j], v)
                    BLAS.axpy!(-hj, SS[:,j], s)
                end

                nrm = norm(v)

                @. v /= nrm
                @. s /= nrm

                α = dot(f, v)

                BLAS.axpy!( α, s, x)
                BLAS.axpy!(-α, v, f)

                if norm(fu)/sqrt(ndofu) < ϵ_l &&
                   norm(fp)/sqrt(ndofp) < ϵ_l

                    noisy > 0 && println("KSP converged in $its iterations")
                    return its
                end

                copyto!(VV[:,k], v)
                copyto!(SS[:,k], s)

                its += 1
            end
        end

        noisy > 0 && println("KSP failed after $its iterations")

        return its
    end
end

function DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e3, niter_l=10, ϵ_l=1e-11, 𝐊_PC=I(size(𝐊,1)))
    
    if nnz(𝐏) == 0 # incompressible limit
        𝐏inv  = ηb .* I(size(𝐏,1))
    else # compressible case
        𝐏inv  = spdiagm(1.0 ./diag(𝐏))
    end
    𝐊sc      = 𝐊    .- 𝐐*(𝐏inv*𝐐ᵀ)
    𝐊sc_PC   = 𝐊_PC .- 𝐐*(𝐏inv*𝐐ᵀ)

    if fact == :chol
        L_PC  = I(size(𝐊sc,1))
        𝐊fact = cholesky(Hermitian(L_PC*𝐊sc), check=false)
    elseif fact == :symchol
        L_PC  = 𝐊sc'
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
        @time Ksym = L_PC*𝐊sc
        @time 𝐊fact = cholesky(Hermitian(Ksym), check=false)
    elseif fact == :PCchol
        L_PC  = I(size(𝐊sc,1))
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
    elseif fact == :lu
        L_PC  = I(size(𝐊sc,1))
        @time 𝐊fact = lu(L_PC*𝐊sc)
    end
    ru    = zeros(size(𝐊,1))
    u     = zeros(size(𝐊,1))
    ru    = zeros(size(𝐊,1))
    fusc  = zeros(size(𝐊,1))
    p     = zeros(size(𝐐,2))
    rp    = zeros(size(𝐐,2))
    # Iterations
    for rit=1:niter_l           
        ru   .= fu .- 𝐊*u  .- 𝐐*p
        rp   .= fp .- 𝐐ᵀ*u .- 𝐏*p
        nrmu, nrmp = norm(ru), norm(rp)
        @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
        if nrmu/sqrt(length(ru)) < ϵ_l && nrmp/sqrt(length(rp)) < ϵ_l
            break
        end
        fusc .= fu  .- 𝐐*(𝐏inv*fp .+ p)
        u    .= 𝐊fact\(L_PC*fusc)

        # # Iterative refinement
        # ϵ_ref = 1e-7
        # for iter_ref=1:10
        #     ru .= 𝐊sc*u .- fusc
        #     @printf("  --> Iterative refinement %02d\n res.   = %2.2e\n", iter_ref, norm(ru)/sqrt(length(ru)))
        #     norm(ru)/sqrt(length(ru)) < ϵ_ref && break
        #     du  = 𝐊fact\(L_PC*ru)
        #     u  .-= du
        # end
   
        p   .+= 𝐏inv*(fp .- 𝐐ᵀ*u .- 𝐏*p)
    end
    return u, p
end
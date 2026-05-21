using ForwardDiff


function line(p, K, dt, η_ve, ψ, p1, t1)
    p2 = p1 + K * dt * sind(ψ)  # introduce sinϕ ?
    t2 = t1 - η_ve
    a = (t2 - t1) / (p2 - p1)
    b = t2 - a * p2
    return a * p + b
end

# Yield and Potential Functions ----------------------------------
yield_DruckerPrager(τ, P, C, cosΨ, sinΨ) = τ - C * cosΨ - P * sinΨ

function Yield(x, p, model::DruckerPrager1)
    (; C, cosϕ, sinϕ, cosψ, sinψ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    F = yield_DruckerPrager(τ, P, p.C, p.cosϕ, p.sinϕ)
    return (F - λ̇ * ηvp) * (F > ϵ) + (F < ϵ) * λ̇ * ηvp
end

function Potential(x, p, model::DruckerPrager1)
    (; C, cosϕ, sinϕ, cosψ, sinψ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    Q = yield_DruckerPrager(τ, P, C, cosψ, sinψ)
    return Q
end

yield_Hyperbolic(τ, P, C, cosΨ, sinΨ, σT) = sqrt(τ^2 + (C * cosΨ - σT * sinΨ)^2) - (P * sinΨ + C * cosΨ)

function Yield(x, p, model::DruckerHyperbolic)
    (; C, cosϕ, sinϕ, cosψ, sinψ, σT, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    F = yield_Hyperbolic(τ, P, C, cosϕ, sinϕ, σT)
    return (F - λ̇ * ηvp) * (F >= ϵ) + (F < ϵ) * λ̇ * ηvp
end

function Potential(x, p, model::DruckerHyperbolic)
    (; C, cosϕ, sinϕ, cosψ, sinψ, σT, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    Q = yield_Hyperbolic(τ, P, C, cosψ, sinψ, σT)
    return Q
end

@inline Af(p, pc, pt, γ) = (pc - pt) / (2 * π) * (2 * atan(γ * (pc + pt - 2p) / (2 * pc)) + π)
@inline Bf(p, pc, pt, M, C, α) = M * C * exp(α * (p - C) / (pc - pt))
@inline Cf(pc, pt, γ) = (pc - pt) / π * atan(γ / 2) + (pc + pt) / 2

yield_Golchin(τ, P, A, B, C, β, λ̇, ηvp) = B * (P - λ̇ * ηvp - C)^2 / A + A * (τ - λ̇ * ηvp - β * (P - λ̇ * ηvp))^2 / B - A * B

function Yield(x, p, model::Golchin2021)
    (; M, N, Pt, Pc, α, β, γ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    C = Cf(Pc, Pt, γ)
    B = Bf(P, Pc, Pt, M, C, α)
    A = Af(P, Pc, Pt, γ)
    F = yield_Golchin(τ, P, A, B, C, β, λ̇, 0 * ηvp)
    return (F - λ̇ * ηvp) * (F >= ϵ) + (F < ϵ) * λ̇ * ηvp
    # return (F)*(F>=ϵ) + (F<ϵ)*λ̇*ηvp
end

function Potential(x, p, model::Golchin2021)
    (; M, N, Pt, Pc, α, β, γ, ηvp) = p
    ϵ = -1e-13
    τ, P, λ̇ = x[1], x[2], x[3]
    C = Cf(Pc, Pt, γ)
    B = Bf(P, Pc, Pt, N, C, α)
    A = Af(P, Pc, Pt, γ)
    Q = yield_Golchin(τ, P, A, B, C, β, λ̇, 0 * ηvp)
    return Q
end

# Residual -------------------------------------------
function ResidualDeviator(x, τ_trial, ε̇_eff, ηve, p, model)
    τ, P, λ̇ = x[1], x[2], x[3]
    ∂Q∂σ = ad_gradient(Potential, x, p, model)
    # return ε̇_eff -  τ/2/ηve  - λ̇/2*∂Q∂σ[1][1]
    return τ - τ_trial + ηve * λ̇ * ∂Q∂σ[1]
end

function ResidualVolume(x, P_trial, Dkk, P0, K, Δt, p, model)
    τ, P, λ̇ = x[1], x[2], x[3]
    ∂Q∂σ = ad_gradient(Potential, x, p, model)
    return P - P_trial + K * Δt * λ̇ * ∂Q∂σ[2]
end

function RheologyResidual(x, trial, plastic, model)
    τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt = trial
    return @SVector([
        ResidualDeviator(x, τ_trial, ε̇_eff, ηve, plastic, model),
        ResidualVolume(x, P_trial, Dkk, P0, K, Δt, plastic, model),
        Yield(x, plastic, model),
    ])
end

function bt_line_search(Δx, J, x, r, trial, plastic, model; α=1.0, ρ=0.5, c=1.0e-4, α_min=1.0e-8)
    # Borrowed from RheologicalCalculator
    perturbed_x = @. x + α * Δx
    perturbed_r = RheologyResidual(x, trial, plastic, model)

    J_times_Δx = -J * Δx
    while sqrt(sum(perturbed_r .^ 2)) > sqrt(sum((r + (c * α * (J_times_Δx))) .^ 2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_x = @. x + α * Δx
        perturbed_r = RheologyResidual(x, trial..., plastic, model)
    end
    return α
end

# Return mapping functions ------------------------------------
function NonLinearReturnMapping(τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, plastic, model)

    tol = 1e-5
    λ̇ = zero(τII)
    K = 1 / β
    τ_trial = τII
    P_trial = P
    itermax = 100

    T = typeof(τII)
    x = SVector{3,T}(τII, P, λ̇)
    αvec = @SVector([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    Fvec = MVector{length(αvec),T}(zeros(T, length(αvec)))

    trial = (τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt)

    R = RheologyResidual(x, trial, plastic, model)
    nR = abs(R[3])#norm(R)
    iter, nR0 = 0, nR
    R0 = copy(R)

    while nR > tol && (nR / nR0) > tol && iter < itermax

        iter += 1
        x0 = copy(x)
        R, J = ad_value_and_jacobian(RheologyResidual, x, trial, plastic, model)
        δx = -J \ R
        nR = abs(R[3])

        # x .= x0 .+  1*δx

        # α = bt_line_search(δx, J.derivs[1], x0, J.val, trial, plastic, model)
        # x .= x0 .+  α*δx

        for ils in eachindex(αvec)
            x = @. x0 + αvec[ils] * δx
            R = RheologyResidual(x, trial, plastic, model)
            Fvec[ils] = norm(ForwardDiff.value.(R))
        end
        # ibest = argmin(Fvec)
        # x .= x0 .+  αvec[ibest]*δx

        # @show iter, nR,  αvec[ibest], x

        # if isnan(norm(δx))
        #     @show R0
        #     @show J.val
        #     @show J.derivs[1]
        #     @show δx
        #     @show iter, nR,  αvec[ibest]
        #     error()
        # end
    end

    if iter == itermax && (nR > tol && (nR / nR0) > tol)
        R = RheologyResidual(x, trial, plastic, model)
        @warn "Failed return mapping after $iter iterations"
        @show τII, P, ε̇_eff, ηve, β
        @show nR, nR0, nR / nR0, tol
        @show trial
        @show plastic
        @show R0
        @show R
        @show x
        # Relax tolerance and retry once
        tol_relax = tol * 100
        if nR > tol_relax || (nR / nR0) > tol_relax
            error("Failed return mapping")
        end
    end

    if x[1] < 0
        @show R, x
        error()
    end

    return x[1], x[2], x[3]
end

function Kiss2023ReturnMapping(τ, P, η_ve, comp, β, Δt, C, φ, ψ, ηvp, σ_T, δσ_T, pc1, τc1, pc2, τc2)
    K = 1 / β
    λ̇ = zero(τ)
    Pc = P
    τc = τ
    l1 = line(P, K, Δt, η_ve, 90.0, pc1, τc1)
    l2 = line(P, K, Δt, η_ve, 90.0, pc2, τc2)
    l3 = line(P, K, Δt, η_ve, ψ, pc2, τc2)
    if max(τ - P * sind(φ) - C * cosd(φ), τ - P - σ_T, -P - (σ_T - δσ_T)) > 0.0
        if τ <= τc1
            # pressure limiter 
            dqdp = -1.0
            f = -P - (σ_T - δσ_T)
            λ̇ = f / (K * Δt)
            τc = τ
            Pc = P - K * Δt * λ̇ * dqdp
            f = -Pc - (σ_T - δσ_T)
            domain_pl = 1.0
        elseif τc1 < τ <= l1
            # corner 1 
            τc = τ - η_ve * (τ - τc1) / (η_ve + ηvp)
            Pc = P - K * Δt * (P - pc1) / (K * Δt + ηvp)
            domain_pl = 2.0
        elseif l1 < τ <= l2            # mode-1
            # tension
            dqdp = -1.0
            dqdτ = 1.0
            f = τ - P - σ_T
            λ̇ = f / (K * Δt + η_ve + ηvp)
            τc = τ - η_ve * λ̇ * dqdτ
            Pc = P - K * Δt * λ̇ * dqdp
            domain_pl = 3.0
        elseif l2 < τ <= l3 # 2nd corner
            # corner 2
            τc = τ - η_ve * (τ - τc2) / (η_ve + ηvp)
            Pc = P - K * Δt * (P - pc2) / (K * Δt + ηvp)
            domain_pl = 4.0
        elseif l3 < τ
            # Drucker-Prager
            dqdp = -sind(ψ)
            dqdτ = 1.0
            f = τ - P * sind(φ) - C * cosd(φ)
            λ̇ = f / (K * Δt * sind(φ) * sind(ψ) + η_ve + ηvp)
            τc = τ - η_ve * λ̇ * dqdτ
            Pc = P - K * Δt * λ̇ * dqdp
            domain_pl = 5.0
        end
    end
    return τc, Pc, λ̇
end

function AnalyticalReturnMapping(τII, P, ηve, comp, β, Δt, C, cosϕ, sinϕ, sinψ, ηvp)
    λ̇ = zero(τII)
    F = τII - C * cosϕ - P * sinϕ - λ̇ * ηvp
    if F > 1e-10
        λ̇ = F / (ηve + ηvp + comp * Δt / β * sinϕ * sinψ)
        τII -= λ̇ * ηve
        P += comp * λ̇ * sinψ * Δt / β
        F = τII - C * cosϕ - P * sinϕ - λ̇ * ηvp
        (F > 1e-10) && error("Failed return mapping")
    end
    return τII, P, λ̇
end

function TensileReturnMapping(τII, P, ηve, comp, β, Δt, σT, ηvp)
    λ̇ = zero(τII)
    F = τII - σT - P - λ̇ * ηvp
    if F > 1e-10
        λ̇ = F / (ηve + ηvp + comp * Δt / β)
        τII -= λ̇ * ηve
        P += comp * λ̇ * Δt / β
        F = τII - σT - P - λ̇ * ηvp
        (F > 1e-10) && error("Failed return mapping")
        (τII < 0.0) && error("Plasticity without condom")
    end
    return τII, P, λ̇
end


# Return mapping --------------------------------------------
return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, ::NoPlasticity, phases) = τII, P, 0.0

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::VonMises, phases)
    return AnalyticalReturnMapping(τII, P, ηvep, comp, β, Δt,
        pl.C[phases], pl.cosϕ[phases], 0.0, 0.0, pl.ηvp[phases])
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerPrager, phases)
    return AnalyticalReturnMapping(τII, P, ηvep, comp, β, Δt,
        pl.C[phases], pl.cosϕ[phases], pl.sinϕ[phases], pl.sinψ[phases], pl.ηvp[phases])
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerPrager1, phases)
    p = (C=pl.C[phases], cosϕ=pl.cosϕ[phases], sinϕ=pl.sinϕ[phases], sinψ=pl.sinψ[phases], cosψ=pl.cosψ[phases], ηvp=pl.ηvp[phases])
    return NonLinearReturnMapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, p, DruckerPrager1())
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::DruckerHyperbolic, phases)
    p = (C=pl.C[phases], cosϕ=pl.cosϕ[phases], sinϕ=pl.sinϕ[phases], sinψ=pl.sinψ[phases], cosψ=pl.cosψ[phases], σT=pl.σT[phases], ηvp=pl.ηvp[phases])
    return NonLinearReturnMapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, p, DruckerHyperbolic())
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::Golchin2021, phases)
    Pt = -pl.σT[phases]
    p = (M=pl.M[phases], N=pl.N[phases], Pt, Pc=pl.Pc[phases], α=pl.a[phases], β=pl.b[phases], γ=pl.c[phases], ηvp=pl.ηvp[phases])
    return NonLinearReturnMapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, p, Golchin2021())
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::Kiss2023, phases)
    return Kiss2023ReturnMapping(τII, P, ηvep, comp, β, Δt,
        pl.C[phases], pl.ϕ[phases], pl.ψ[phases], pl.ηvp[phases],
        pl.σT[phases], pl.δσT[phases], pl.P1[phases], pl.τ1[phases], pl.P2[phases], pl.τ2[phases])
end

function return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δt, comp, pl::Tensile, phases)
    return TensileReturnMapping(τII, P, ηvep, comp, β, Δt, pl.σT[phases], pl.ηvp[phases])
end

# Strain rate trial ------------------------------------------
function StrainRateTrial(τII, G, Δt, B, n)
    ε̇II_vis = B * τII^n
    ε̇II_trial = ε̇II_vis + τII / (2 * G * Δt)
    return ε̇II_trial
end

# Phase average ----------------------------------------------
function PhaseAverage_summand(a, phase_ratio, averaging)
    if averaging === :harmonic && a != 0.0
        # Hⱼ = w′ᵢ * aᵢ⁻¹
        a_j = phase_ratio / a
    elseif averaging === :geometric && a > 0.0
        # Gⱼ = w′ᵢ * ln(aᵢ)
        a_j = phase_ratio * log(a)
    else # arithmetic
        # Aⱼ =w′ᵢ * aᵢ
        a_j = phase_ratio * a
    end
    return a_j
end

function PhaseAverage(a_average, averaging)
    if averaging === :harmonic && a_average != 0.0
        # H = (Σⁿᵢ₌₁ w′ᵢ * aᵢ⁻¹)⁻¹ = (Σⁿᵢ₌₁ Hⱼ)⁻¹
        a_avg = 1 / a_average
    elseif averaging === :geometric
        # G = exp(Σⁿᵢ₌₁ w′ᵢ * ln(aᵢ)) = exp(Σⁿᵢ₌₁ Gⱼ)
        a_avg = exp(a_average)
    else # arithmetic
        # A = Σⁿᵢ₌₁ w′ᵢ * aᵢ = Σⁿᵢ₌₁ Aⱼ
        a_avg = a_average
    end
    return a_avg
end

function LocalRheology(ε̇, Dkk, P0, materials, phase_ratios, Δ)

    nphases = length(materials.n)
    phase_avg = materials.phase_avg
    eps0 = 1e-17

    # Effective strain rate & pressure
    ε̇II = sqrt((ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1] - ε̇[2])^2) / 2 + ε̇[3]^2) + eps0
    P = ε̇[4]

    η_average = zero(ε̇II)
    λ̇_average = zero(ε̇II)
    P_average = zero(ε̇II)
    τ_average = zero(ε̇II)

    for phases = 1:nphases

        phase_ratios[phases] < eps() && continue

        # P = P_trial
        # Parameters
        ϵ = 1e-10 # tolerance
        n = materials.n[phases]
        η0 = materials.η0[phases]
        B = materials.B[phases]
        G = materials.G[phases]
        β = materials.β[phases]
        comp = materials.compressible

        # Initial guess
        η = (η0.*ε̇II .^ (1 ./ n.-1.0))[1]
        ηvep = inv(1 / η + 1 / (G * Δ.t))
        τII = 2 * ηvep * ε̇II
        # P = P0 - comp * Δ.t / β * P_trial

        # Visco-elastic powerlaw
        for it = 1:20
            r = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
            # @show abs(r)
            (abs(r) < ϵ) && break
            ∂ε̇II∂τII = ad_derivative(StrainRateTrial, τII, G, Δ.t, B, n)
            ∂τII∂ε̇II = inv(∂ε̇II∂τII)
            τII += ∂τII∂ε̇II * r
        end
        isnan(τII) && error()

        # ηvep for analytical solution
        ηvep = τII / 2 / ε̇II

        # Viscoplastic return mapping
        τII, P, λ̇ = return_mapping(τII, P, ε̇II, Dkk, P0, ηvep, β, Δ.t, comp, materials.plasticity, phases)

        # Effective viscosity
        ηvep = τII / (2 * ε̇II)

        # Phase averaging
        η_average += PhaseAverage_summand(ηvep, phase_ratios[phases], phase_avg)
        P_average += PhaseAverage_summand(P, phase_ratios[phases], phase_avg)
        λ̇_average += PhaseAverage_summand(λ̇, phase_ratios[phases], phase_avg)
        τ_average += PhaseAverage_summand(τII, phase_ratios[phases], phase_avg)
    end

    η_average = PhaseAverage(η_average, phase_avg)
    P_average = PhaseAverage(P_average, phase_avg)
    λ̇_average = PhaseAverage(λ̇_average, phase_avg)
    τ_average = PhaseAverage(τ_average, phase_avg)

    return η_average, λ̇_average, P_average, τ_average
end

function StressVector!(ε̇::SVector{N,T}, ε̇kk, P0, materials, phase_ratios, Δ) where {N,T}
    η, λ̇, P, τII = LocalRheology(ε̇, ε̇kk, P0, materials, phase_ratios, Δ)
    τ = SVector{4,T}(2 * η * ε̇[1], 2 * η * ε̇[2], 2 * η * ε̇[3], P)
    return τ, η, λ̇, τII
end


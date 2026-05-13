function InitialiseParticleField(nc, nmpc, L, Δ, materials, noise)
    nphases = length(materials.n)
    num = (x=nmpc.x * (nc.x + 2), y=nmpc.y * (nc.y + 2))
    Δm = (x=L.x / num.x, y=L.y / num.y)
    xm = LinRange(-L.x / 2 - Δ.x + Δm.x / 2, L.x / 2 + Δ.x - Δm.x / 2, num.x)
    ym = LinRange(-L.y / 2 - Δ.y + Δm.y / 2, L.y / 2 + Δ.y - Δm.y / 2, num.y)
    Xm = [xm[i] for i in eachindex(xm), j in eachindex(ym)]
    Ym = [ym[j] for i in eachindex(xm), j in eachindex(ym)]

    # Add noise to marker coordinates
    if noise
        for ind = 1:(num.x*num.y)
            Xm[ind] += (rand() - 0.5) * Δm.x
            Ym[ind] += (rand() - 0.5) * Δm.y
        end
    end
    return (Xm=Xm, Ym=Ym, xm=xm, ym=ym, Δm=Δm, num=num, nphases=nphases)
end

function InitialisePhaseRatios(markers, f)
    phase_ratios = (
        c=[zeros(markers.nphases) for _ in axes(f.xx, 1), _ in axes(f.xx, 2)],
        v=[zeros(markers.nphases) for _ in axes(f.xy, 1), _ in axes(f.xy, 2)],
    )
    phase_weights = (
        c=[zeros(markers.nphases) for _ in axes(f.xx, 1), _ in axes(f.xx, 2)],
        v=[zeros(markers.nphases) for _ in axes(f.xy, 1), _ in axes(f.xy, 2)],
    )
    return phase_ratios, phase_weights
end

function InitialisePhaseRatios(phases::NamedTuple, nphases::Int)
    c = [
        let r = zeros(nphases)
            r[phases.c[i, j]] = 1.0
            r
        end
        for i in axes(phases.c, 1), j in axes(phases.c, 2)
    ]
    v = [
        let r = zeros(nphases)
            r[phases.v[i, j]] = 1.0
            r
        end
        for i in axes(phases.v, 1), j in axes(phases.v, 2)
    ]
    return (c=c, v=v)
end

function MarkerWeight(xm, x, Δx)
    # Compute marker-grid distance and weight
    dst = abs(xm - x)
    w = 1.0 - 2 * dst / Δx
    return w
end

function MarkerWeight_phase!(phase_ratio, phase_weight, x, y, xm, ym, Δ, phase, nphases)
    w_x = MarkerWeight(xm, x, Δ.x)
    w_y = MarkerWeight(ym, y, Δ.y)
    for k = 1:nphases
        phase_ratio[k] += (k === phase) * w_x * w_y
        phase_weight[k] += w_x * w_y
    end
end
function PhaseRatios!(phase_ratios, phase_weights, m, mphase, xce, yce, xve, yve, Δ)

    for I in CartesianIndices(mphase)
        # find indices of grid centroid
        ic = Int64(ceil((m.Xm[I] - xve[1]) / Δ.x))
        jc = Int64(ceil((m.Ym[I] - yve[1]) / Δ.y))
        # find indices of grid verteces
        iv = Int64(ceil((m.Xm[I] - xve[1]) / Δ.x + 0.5))
        jv = Int64(ceil((m.Ym[I] - yve[1]) / Δ.y + 0.5))

        MarkerWeight_phase!(phase_ratios.c[ic, jc], phase_weights.c[ic, jc], xce[ic], yce[jc], m.Xm[I], m.Ym[I], Δ, mphase[I], m.nphases)
        MarkerWeight_phase!(phase_ratios.v[iv, jv], phase_weights.v[iv, jv], xve[iv], yve[jv], m.Xm[I], m.Ym[I], Δ, mphase[I], m.nphases)
    end

    # centroids
    for i in axes(phase_ratios.c, 1), j in axes(phase_ratios.c, 2)
        #  normalize weights and assign to phase ratios
        for k = 1:m.nphases
            phase_ratios.c[i, j][k] = phase_ratios.c[i, j][k] / (phase_weights.c[i, j][k] == 0.0 ? 1 : phase_weights.c[i, j][k])
        end
    end
    # vertices
    for i in axes(phase_ratios.v, 1), j in axes(phase_ratios.v, 2)
        #  normalize weights and assign to phase ratios
        for k = 1:m.nphases
            phase_ratios.v[i, j][k] = phase_ratios.v[i, j][k] / (phase_weights.v[i, j][k] == 0.0 ? 1 : phase_weights.v[i, j][k])
        end
    end
end

function compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, size_c, size_v, nphases)
    nxc, nyc = size(G.c)
    @inbounds for j in 1:nyc, i in 1:nxc
        if 1 < i < nc.x + 2 && 1 < j < nc.y + 2
            βc = 0.0
            Gc = 0.0
            ρc = 0.0
            ξc = 0.0
            pr = phase_ratios.c[i-1, j-1]
            for p = 1:nphases
                r = pr[p]
                βc += r * materials.β[p]
                Gc += r * materials.G[p]
                ρc += r * materials.ρ[p]
                ξc += r * materials.ξ0[p]
            end
            β.c[i, j] = βc
            G.c[i, j] = Gc
            ρ.c[i, j] = ρc
            ξ.c[i, j] = ξc
        else
            β.c[i, j] = 0.0
            G.c[i, j] = 0.0
            ρ.c[i, j] = 0.0
            ξ.c[i, j] = 0.0
        end
    end

    @inbounds for j in 1:nyc
        G.c[1, j] = G.c[2, j]
        G.c[nxc, j] = G.c[nxc-1, j]
        β.c[1, j] = β.c[2, j]
        β.c[nxc, j] = β.c[nxc-1, j]
        ρ.c[1, j] = ρ.c[2, j]
        ρ.c[nxc, j] = ρ.c[nxc-1, j]
        ξ.c[1, j] = ξ.c[2, j]
        ξ.c[nxc, j] = ξ.c[nxc-1, j]
    end
    @inbounds for i in 1:nxc
        G.c[i, 1] = G.c[i, 2]
        G.c[i, nyc] = G.c[i, nyc-1]
        β.c[i, 1] = β.c[i, 2]
        β.c[i, nyc] = β.c[i, nyc-1]
        ρ.c[i, 1] = ρ.c[i, 2]
        ρ.c[i, nyc] = ρ.c[i, nyc-1]
        ξ.c[i, 1] = ξ.c[i, 2]
        ξ.c[i, nyc] = ξ.c[i, nyc-1]
    end

    nxv, nyv = size(G.v)
    @inbounds for j in 1:nyv, i in 1:nxv
        if 1 < i < nc.x + 3 && 1 < j < nc.y + 3
            Gv = 0.0
            pr = phase_ratios.v[i-1, j-1]
            for p = 1:nphases
                Gv += pr[p] * materials.G[p]
            end
            G.v[i, j] = Gv
        else
            G.v[i, j] = 0.0
        end
    end

    @inbounds for j in 1:nyv
        G.v[1, j] = G.v[2, j]
        G.v[nxv, j] = G.v[nxv-1, j]
    end
    @inbounds for i in 1:nxv
        G.v[i, 1] = G.v[i, 2]
        G.v[i, nyv] = G.v[i, nyv-1]
    end
    return nothing
end
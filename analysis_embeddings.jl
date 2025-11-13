# -----------------------------------------------------------------
# 1. SETUP: INSTALL AND LOAD PACKAGES
# -----------------------------------------------------------------
using Pkg

# Activate the local environment
Pkg.activate(".")

# Add necessary packages (idempotent if already installed)
Pkg.add([
    "CSV",
    "DataFrames",
    "Distances",
    "Clustering",
    "Plots",
    "StatsPlots",
    "TSne",
    "MultivariateStats",
    "Distributions",
    "NearestNeighbors",
])

using CSV
using DataFrames
using Distances            # pairwise distances, CosineDist
using Clustering           # hclust, dbscan
using Plots                # plotting (scatter, quiver)
using StatsPlots           # dendrogram recipe, hexbin
using TSne                 # t-SNE
using MultivariateStats    # PCA
using Statistics           # mean, std, quantile
using Random               # seeding
using Dates                # evolution plot labels
using LinearAlgebra        # eigen, cov, etc.
using Distributions        # Chi-square quantile for ellipses
using NearestNeighbors     # KDTree for kNN on reduced PCs

# Ensure output directory exists and set some plotting defaults
isdir("results") || mkpath("results")
default(size=(1000, 850), dpi=150)

# Reproducibility for t-SNE (and any RNG)
Random.seed!(42)

println("Julia packages loaded successfully.")

# ----------------------- SMALL HELPERS -----------------------
# DataFrame column helpers (String-based; robust across DataFrames versions)
hascol(df::DataFrame, col::AbstractString) = col in names(df)
getcol(df::DataFrame, col::AbstractString) = df[!, col]

"""
Robustly coerce a year-like column to Vector{Union{Missing,Int}}.
Accepts Int, "2024", "2024-05", etc.; returns missing if no 4-digit year.
"""
function _to_year_vector(ycol)::Vector{Union{Missing,Int}}
    n = length(ycol)
    out = Vector{Union{Missing,Int}}(undef, n)
    for i in 1:n
        v = ycol[i]
        if v === missing || v === nothing
            out[i] = missing
        elseif v isa Integer
            out[i] = Int(v)
        elseif v isa AbstractString
            m = match(r"\b(\d{4})\b", v)
            out[i] = isnothing(m) ? missing : parse(Int, m.captures[1])
        else
            p = tryparse(Int, string(v))
            out[i] = isnothing(p) ? missing : p
        end
    end
    return out
end

"""
Extract vector matrix (rows=samples, cols=features) and metadata columns.
Returns (X, df_meta, vcols) where df_meta has "dataset" and "year", and vcols are feature names (Vector{String}).
"""
function _load_vectors_and_meta(path::String, dataset_name::String)
    df = CSV.read(path, DataFrame)

    # Normalize names to String (handles Symbol/ String differences across versions)
    name_list = String.(names(df))               # Vector{String}
    vector_cols = [c for c in name_list if startswith(c, "A")]
    if isempty(vector_cols)
        error("No vector columns (starting with 'A') found in $path")
    end

    meta_names = ["system:index", "month", "year", "countryCode"]
    meta_cols  = intersect(meta_names, name_list)
    df_meta = isempty(meta_cols) ? DataFrame() : select(df, meta_cols, copycols=false)

    # Build matrix with selected feature columns (by String names)
    X = Matrix(coalesce.(df[:, vector_cols], 0.0))  # rows=samples, cols=features

    # Ensure "year" exists in meta (if missing, fill with missing)
    if !("year" in names(df_meta))
        df_meta.year = fill(missing, size(X, 1))
    end
    df_meta.dataset = fill(dataset_name, size(X, 1))

    return X, df_meta, vector_cols   # vector_cols::Vector{String}
end

"""
Compute centroids (means) of rows for each group key.
`scores` must be n×p (rows=samples, cols=PCs), `groups` is Vector{String} length n.
Returns Dict{String, Vector{Float64}} mapping group -> centroid Vector (length p).
"""
function _centroids(scores::AbstractMatrix, groups::AbstractVector{<:AbstractString})
    @assert size(scores, 1) == length(groups)
    G = unique(groups)
    bygroup = Dict{String, Vector{Float64}}()
    for g in G
        idx = findall(==(g), groups)
        bygroup[g] = vec(mean(scores[idx, :], dims=1))
    end
    return bygroup
end

# Cosine similarity via Distances.jl (no LinearAlgebra norms needed)
_cosine(u::AbstractVector, v::AbstractVector) = 1 - (CosineDist())(u, v)

"""
Compute ellipse coordinates (x,y) for bivariate 95% covariance ellipse.
"""
function _ellipse(μ::Vector{<:Real}, Σ::AbstractMatrix; level=0.95, npts=200)
    r2 = quantile(Chisq(2), level)
    evals, evecs = eigen(Symmetric(Σ))
    evals = max.(evals, 0) # guard tiny negatives
    R = evecs * Diagonal(sqrt.(evals))
    θ = range(0, 2π; length=npts)
    circ = [cos.(θ)'; sin.(θ)']
    E = (sqrt(r2) .* (R * circ)) .+ μ
    return E[1, :], E[2, :]
end

# ------------------- CROSS-DATASET PCA -------------------
"""
compare_datasets(file_list::Vector{Tuple{String,String}}; k=10, standardize=false)

Performs a joint PCA across all files to identify dominant shared components
and visualize evolution between datasets/years.

Outputs:
- results/cross_pca_scatter_pc1_pc2.png
- results/cross_pca_hexbin_pc1_pc2.png
- results/cross_pca_centroids_pc1_pc2.png
- results/cross_pca_centroids_ellipses.png
- results/cross_pca_trajectories_pc1_pc2.png
- results/cross_pca_explained_variance.png
- results/cross_centroid_cosine_topK.csv
- results/cross_pca_pc1_centroids.csv
- results/cross_pca_pc1_evolution.png
- results/cross_pca_centroid_shifts.csv
- results/prominent_vectors_summary.csv
- results/prominent_vectors_<dataset>.csv (per dataset)
- results/pc_loadings_full.csv
- results/pc1_top_features.csv
- results/pc2_top_features.csv
"""
function compare_datasets(file_list::Vector{Tuple{String,String}}; k::Int=10, standardize::Bool=false)
    println("\n-------------------------------------------------")
    println("🔗 Cross-dataset analysis (PCA)")
    println("-------------------------------------------------")

    # 1) Load and stack
    matrices   = Vector{Matrix{Float64}}()
    metas      = Vector{DataFrame}()
    vcols_all  = Vector{Vector{String}}()     # store String feature names
    for (path, prefix) in file_list
        try
            X, meta, vcols = _load_vectors_and_meta(path, prefix)
            push!(matrices, X)
            push!(metas, meta)
            push!(vcols_all, vcols)
            println("   ... loaded $(path)  →  $(size(X,1)) samples, $(size(X,2)) features")
        catch e
            @warn "Skipping $path due to error: $e"
        end
    end
    if isempty(matrices)
        println("❌ No datasets could be loaded for comparison.")
        return
    end

    # Check feature name consistency across files (optional)
    feature_names = vcols_all[1]  # Vector{String}
    for vnames in vcols_all[2:end]
        if length(vnames) != length(feature_names) || any(vnames .!= feature_names)
            @warn "Feature columns differ across files; using first file's feature ordering for loadings export."
            break
        end
    end

    X_all    = vcat(matrices...)    # n×d (rows=samples)
    meta_all = vcat(metas...)       # n×(meta)
    n, d = size(X_all)
    println("   ... combined matrix: $n samples, $d features")

    # Labels
    dataset_labels = String.(meta_all.dataset)
    year_labels    = "year" in names(meta_all) ? _to_year_vector(meta_all.year) : fill(missing, n)

    # 2) Optional standardization across all samples (z-score per feature)
    X_for_pca = if standardize
        Xμ = mean(X_all, dims=1)
        Xσ = std(X_all, dims=1; corrected=true)
        Xσ[Xσ .== 0] .= 1.0
        (X_all .- Xμ) ./ Xσ
    else
        X_all
    end

    # 3) PCA with observations as columns (MultivariateStats expects d×n)
    Xt = transpose(X_for_pca)                    # d×n
    maxout = min(k, min(d, n))                  # guard k
    pca_model = fit(PCA, Xt; maxoutdim=maxout)  # center=true by default

    # Scores of training data: p×n -> transpose to n×p (rows=samples, cols=PCs)
    scores = transpose(MultivariateStats.transform(pca_model, Xt))
    p = size(scores, 2)

    # Explained variance ratios
    prinvars = pca_model.prinvars
    evr = prinvars ./ sum(prinvars)

    # ================== Prominent vectors & feature loadings ==================
    try
        # ---- Sample-level prominence: PC1 extremes & representative (medoid in top PCs) ----
        pc1 = scores[:, 1]
        Krep = min(5, size(scores, 2))
        repspace = scores[:, 1:Krep]
        has_sid = hascol(meta_all, "system:index")
        sid_col  = has_sid ? getcol(meta_all, "system:index") : string.(1:size(scores,1))

        summ = DataFrame(dataset=String[],
                         prominent_abs_pc1_index=Int[], prominent_abs_pc1_value=Float64[],
                         prominent_pos_pc1_index=Int[], prominent_pos_pc1_value=Float64[],
                         prominent_neg_pc1_index=Int[], prominent_neg_pc1_value=Float64[],
                         representative_index=Int[], representative_dist=Float64[])

        for ds in sort(unique(dataset_labels))
            idxs = findall(==(ds), dataset_labels)
            # 1) ABS(PC1) leader
            absvals = abs.(pc1[idxs])
            i_abs_local = argmax(absvals)
            i_abs = idxs[i_abs_local]
            v_abs = pc1[i_abs]

            # 2) Directional leaders: max and min PC1
            i_pos = idxs[argmax(pc1[idxs])]
            v_pos = pc1[i_pos]
            i_neg = idxs[argmin(pc1[idxs])]
            v_neg = pc1[i_neg]

            # 3) Representative medoid in first Krep PCs (closest to centroid)
            μ = vec(mean(repspace[idxs, :], dims=1))
            d2 = [sum((repspace[j, :] .- μ).^2) for j in idxs]  # squared Euclidean
            i_rep_local = argmin(d2)
            i_rep = idxs[i_rep_local]
            rep_dist = sqrt(d2[i_rep_local])

            push!(summ, (ds, i_abs, v_abs, i_pos, v_pos, i_neg, v_neg, i_rep, rep_dist))

            # Per-dataset CSV with metadata for the picks
            ds_out = DataFrame(
                role = ["abs_pc1_leader", "pos_pc1_leader", "neg_pc1_leader", "representative_medoid"],
                global_index = [i_abs, i_pos, i_neg, i_rep],
                system_index = [sid_col[i_abs], sid_col[i_pos], sid_col[i_neg], sid_col[i_rep]],
                year = [year_labels[i_abs], year_labels[i_pos], year_labels[i_neg], year_labels[i_rep]],
                PC1_score = [v_abs, v_pos, v_neg, pc1[i_rep]]
            )
            CSV.write("results/prominent_vectors_$(ds).csv", ds_out)
        end

        # Add human-readable ids & years to summary
        summ.system_index_abs = [sid_col[i] for i in summ.prominent_abs_pc1_index]
        summ.system_index_pos = [sid_col[i] for i in summ.prominent_pos_pc1_index]
        summ.system_index_neg = [sid_col[i] for i in summ.prominent_neg_pc1_index]
        summ.system_index_rep = [sid_col[i] for i in summ.representative_index]
        summ.year_abs = [year_labels[i] for i in summ.prominent_abs_pc1_index]
        summ.year_pos = [year_labels[i] for i in summ.prominent_pos_pc1_index]
        summ.year_neg = [year_labels[i] for i in summ.prominent_neg_pc1_index]
        summ.year_rep = [year_labels[i] for i in summ.representative_index]

        CSV.write("results/prominent_vectors_summary.csv", summ)
        println("   ✅ Saved results/prominent_vectors_summary.csv and per-dataset CSVs")
    catch e
        @warn "Failed to compute/save prominent vectors: $e"
    end

    try
        # ---- Feature-level prominence: loading vectors and top features ----
        # Loadings matrix: d × p (feature loadings for each PC)
        W = MultivariateStats.projection(pca_model)  # d×p
        feats = feature_names                        # Vector{String}
        pc_cols = [Symbol("PC"*string(i)) for i in 1:size(W,2)]
        dfW = DataFrame(feature = feats)
        for j in 1:size(W,2)
            dfW[!, pc_cols[j]] = W[:, j]
        end
        CSV.write("results/pc_loadings_full.csv", dfW)
        println("   ✅ Saved results/pc_loadings_full.csv")

        # Top-N features by |loading| for PC1 and PC2
        function _top_features(W::AbstractMatrix, pc::Int; N::Int=30)
            vals = W[:, pc]
            idx = sortperm(abs.(vals); rev=true)[1:min(N, length(vals))]
            DataFrame(rank = 1:length(idx),
                      feature = feats[idx],
                      loading = vals[idx],
                      abs_loading = abs.(vals[idx]))
        end
        CSV.write("results/pc1_top_features.csv", _top_features(W, 1; N=30))
        println("   ✅ Saved results/pc1_top_features.csv")
        if size(W,2) >= 2
            CSV.write("results/pc2_top_features.csv", _top_features(W, 2; N=30))
            println("   ✅ Saved results/pc2_top_features.csv")
        end
    catch e
        @warn "Failed to compute/save feature loadings: $e"
    end
    # ================== END Prominent vectors & loadings ==================

    # 4) PC1–PC2 scatter (transparent, no strokes)
    try
        plt1 = scatter(
            scores[:, 1], scores[:, 2];
            group = dataset_labels,
            title = "Cross-dataset PCA: PC1 vs PC2",
            xlabel = "PC1 ($(round(evr[1]*100, digits=1))% var)",
            ylabel = "PC2 ($(round(evr[2]*100, digits=1))% var)",
            markersize = 1.5,
            markeralpha = 0.2,
            markerstrokewidth = 0,
            legend = :outertopright
        )
        savefig(plt1, "results/cross_pca_scatter_pc1_pc2.png")
        println("   ✅ Saved results/cross_pca_scatter_pc1_pc2.png")
    catch e
        @warn "Failed to save PC1–PC2 scatter: $e"
    end

    # 5) Hexbin density for PC1–PC2
    try
        plt_hex = hexbin(scores[:,1], scores[:,2];
                         nbins=60,
                         title="PC1–PC2 density (hexbin)",
                         xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                         ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)")
        savefig(plt_hex, "results/cross_pca_hexbin_pc1_pc2.png")
        println("   ✅ Saved results/cross_pca_hexbin_pc1_pc2.png")
    catch e
        @warn "Failed to save PC1–PC2 hexbin density: $e"
    end

    # 6) Centroids per dataset on PC1–PC2 (points + labels)
    cent_by_dataset = _centroids(scores[:, 1:2], dataset_labels)
    try
        ds_names = sort(collect(keys(cent_by_dataset)))
        cx = [cent_by_dataset[ds][1] for ds in ds_names]
        cy = [cent_by_dataset[ds][2] for ds in ds_names]
        plt2 = scatter(
            cx, cy;
            seriestype = :scatter,
            markershape = :circle,
            title = "Centroids by dataset in PC space",
            xlabel = "PC1 ($(round(evr[1]*100, digits=1))% var)",
            ylabel = "PC2 ($(round(evr[2]*100, digits=1))% var)",
            legend = false,
            markersize = 8
        )
        for (i, name) in pairs(ds_names)
            annotate!(cx[i], cy[i], text(name, 9, :left))
        end
        savefig(plt2, "results/cross_pca_centroids_pc1_pc2.png")
        println("   ✅ Saved results/cross_pca_centroids_pc1_pc2.png")
    catch e
        @warn "Failed to save centroid plot: $e"
    end

    # 7) Per-dataset 95% ellipses around centroids
    try
        plt_ell = plot(title="PC1–PC2 centroids with 95% ellipses",
                       xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                       ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)")
        for ds in sort(unique(dataset_labels))
            idx = findall(==(ds), dataset_labels)
            X = scores[idx, 1:2]
            μ = vec(mean(X, dims=1))
            Σ = cov(X; corrected=true)
            ex, ey = _ellipse(μ, Σ; level=0.95)
            plot!(ex, ey; label=false)
            scatter!([μ[1]], [μ[2]]; label=ds, markersize=7)
        end
        savefig(plt_ell, "results/cross_pca_centroids_ellipses.png")
        println("   ✅ Saved results/cross_pca_centroids_ellipses.png")
    catch e
        @warn "Failed to save centroid ellipses: $e"
    end

    # 8) Trajectories by dataset kind across years (PC1–PC2) with arrows
    base_kind = map(dataset_labels) do s
        parts = split(s, "_")
        length(parts) >= 2 ? parts[end] : s
    end
    try
        plt3 = plot(title = "Centroid trajectories by dataset kind (PC1–PC2)",
                    xlabel = "PC1 ($(round(evr[1]*100, digits=1))% var)",
                    ylabel = "PC2 ($(round(evr[2]*100, digits=1))% var)")
        added_any = false
        for kind in unique(base_kind)
            idxs = findall(base_kind .== kind)
            ys = year_labels[idxs]
            valid = .!(ismissing.(ys))
            isempty(findall(valid)) && continue

            sub_pc1 = scores[idxs[valid], 1]
            sub_pc2 = scores[idxs[valid], 2]
            sub_year = Int.(ys[valid])

            dfk = DataFrame(pc1=sub_pc1, pc2=sub_pc2, year=sub_year)
            years = sort(unique(dfk.year))
            xs = Float64[]; ys2 = Float64[]
            for y in years
                sub = dfk[dfk.year .== y, :]
                μ = vec(mean(Matrix(select(sub, [:pc1, :pc2])), dims=1))
                push!(xs, μ[1]); push!(ys2, μ[2])
            end

            plot!(xs, ys2; label=kind, markershape=:circle, markersize=6)
            if length(xs) > 1
                dx = diff(xs); dy = diff(ys2)
                quiver!(xs[1:end-1], ys2[1:end-1]; quiver=(dx, dy), arrow=true, label=false)
            end
            for (i, y) in enumerate(years)
                annotate!(xs[i], ys2[i], text(string(y), 9, :left))
            end
            added_any = true
        end
        if added_any
            savefig(plt3, "results/cross_pca_trajectories_pc1_pc2.png")
            println("   ✅ Saved results/cross_pca_trajectories_pc1_pc2.png")
        else
            @warn "No valid year data to plot trajectories."
        end
    catch e
        @warn "Failed to save trajectories: $e"
    end

    # 9) Explained variance bar chart
    try
        m = min(10, length(evr))
        plt4 = bar(1:m, evr[1:m];
                   xlabel = "Principal component",
                   ylabel = "Explained variance ratio",
                   title = "Explained variance (top $m PCs)")
        savefig(plt4, "results/cross_pca_explained_variance.png")
        println("   ✅ Saved results/cross_pca_explained_variance.png")
    catch e
        @warn "Failed to save explained variance bar chart: $e"
    end

    # 10) Cosine similarity between dataset centroids on top K PCs
    try
        topK = min(5, size(scores, 2))  # use first 5 PCs if available
        centK = _centroids(scores[:, 1:topK], dataset_labels)
        ds = sort(collect(keys(centK)))
        S = Matrix{Float64}(undef, length(ds), length(ds))
        for i in eachindex(ds), j in eachindex(ds)
            S[i, j] = _cosine(centK[ds[i]], centK[ds[j]])
        end
        out = DataFrame([:dataset => ds])
        for (j, name) in enumerate(ds)
            out[!, Symbol(name)] = S[:, j]
        end
        outpath = "results/cross_centroid_cosine_topK.csv"
        CSV.write(outpath, out)
        println("   ✅ Saved $outpath")
    catch e
        @warn "Failed to compute/save centroid cosine table: $e"
    end

    # 11) PC1 “dominant component” report: centroid per dataset + rank
    try
        pc1_scores = scores[:, 1]
        cent_pc1 = _centroids(reshape(pc1_scores, :, 1), dataset_labels)  # centroids on PC1 only
        ds = sort(collect(keys(cent_pc1)))
        pc1vals = [cent_pc1[d][1] for d in ds]
        order = sortperm(pc1vals; rev=true)
        inv = invperm(order) # 1 = highest
        report = DataFrame(dataset = ds, PC1_centroid = pc1vals, PC1_rank = inv)
        CSV.write("results/cross_pca_pc1_centroids.csv", report)
        println("   ✅ Saved results/cross_pca_pc1_centroids.csv")
    catch e
        @warn "Failed to save PC1 centroid report: $e"
    end

    # 12) Year-by-year evolution on PC1 (compact, non-empty)
    try
        base_kind2 = map(dataset_labels) do s
            parts = split(s, "_")
            length(parts) >= 2 ? parts[end] : s
        end
        valid = .!(ismissing.(year_labels))
        if any(valid)
            df_evo = DataFrame(kind = base_kind2[valid],
                               year = Int.(year_labels[valid]),
                               pc1  = scores[valid, 1])
            gk = groupby(df_evo, [:kind, :year])
            evo = combine(gk, :pc1 => mean => :pc1_mean)

            kinds = unique(evo.kind)
            plt_evo = plot(title="Evolution of PC1 centroid by kind",
                           xlabel="Year", ylabel="PC1 centroid")
            for k in kinds
                sub = evo[evo.kind .== k, :]
                ys = sort(unique(sub.year))
                μ = [sub[sub.year .== y, :pc1_mean][1] for y in ys]
                plot!(ys, μ; label=k, markershape=:circle, markersize=6)
            end
            savefig(plt_evo, "results/cross_pca_pc1_evolution.png")
            println("   ✅ Saved results/cross_pca_pc1_evolution.png")
        else
            @warn "No valid year values for PC1 evolution plot."
        end
    catch e
        @warn "Failed to save PC1 evolution: $e"
    end

    # 13) Per-kind centroid shift summary from earliest→latest year
    try
        base_kind3 = map(dataset_labels) do s
            parts = split(s, "_"); length(parts) >= 2 ? parts[end] : s
        end
        valid = .!(ismissing.(year_labels))
        if any(valid)
            df = DataFrame(kind = base_kind3[valid],
                           year = Int.(year_labels[valid]),
                           pc1  = scores[valid, 1],
                           pc2  = scores[valid, 2])
            g = groupby(df, [:kind, :year])
            cent = combine(g, [:pc1, :pc2] .=> mean .=> [:pc1_mean, :pc2_mean])
            out = DataFrame(kind=String[], year_start=Int[], year_end=Int[],
                            d_pc1=Float64[], d_pc2=Float64[], shift_norm=Float64[], angle_deg=Float64[])
            for k in unique(cent.kind)
                sub = cent[cent.kind .== k, :]
                yrs = sort(unique(sub.year))
                if length(yrs) >= 2
                    a = sub[sub.year .== first(yrs), [:pc1_mean, :pc2_mean]]
                    b = sub[sub.year .== last(yrs),  [:pc1_mean, :pc2_mean]]
                    d1 = b.pc1_mean[1] - a.pc1_mean[1]
                    d2 = b.pc2_mean[1] - a.pc2_mean[1]
                    push!(out, (k, first(yrs), last(yrs), d1, d2, hypot(d1,d2), atan(d2, d1)*180/pi))
                end
            end
            CSV.write("results/cross_pca_centroid_shifts.csv", out)
            println("   ✅ Saved results/cross_pca_centroid_shifts.csv")
        end
    catch e
        @warn "Failed to save centroid shift summary: $e"
    end

    println("🏁 Finished cross-dataset PCA.")
    return nothing
end

# -----------------------------------------------------------------
# 2. DEFINE THE PER-FILE ANALYSIS FUNCTION
#     — with DBSCAN on top PCs + auto-tuned eps —
# -----------------------------------------------------------------
"""
    analyze_vectors(input_csv::String, output_prefix::String; nPC=30, k_percentile=0.98)

Loads a CSV file containing vectors, runs a full similarity analysis,
and saves the output plots with the specified prefix.

DBSCAN is run on the first `nPC` principal components (default up to 30).
`eps` is auto-tuned from the k-distance curve (98th percentile of kNN distances).
Also saves the k-distance curve plot.
"""
function analyze_vectors(input_csv::String, output_prefix::String; nPC::Int=30, k_percentile::Float64=0.98)
    println("-------------------------------------------------")
    println("🚀 Starting Analysis for: $input_csv")
    println("   Saving outputs with prefix: $output_prefix")
    println("-------------------------------------------------")

    # === 2a. Load and Prepare Data ===
    local df
    try
        df = CSV.read(input_csv, DataFrame)
        println("   ... Successfully loaded $input_csv.")
    catch e
        println("❌ ERROR: Could not read file '$input_csv'.")
        println(e)
        println("   Skipping this file.")
        return
    end

    # Normalize names to String and detect columns
    name_list = String.(names(df))                  # Vector{String}
    vector_cols = [c for c in name_list if startswith(c, "A")]
    meta_names  = ["system:index", "month", "year", "countryCode"]
    metadata_cols = intersect(meta_names, name_list)

    # Check if we found vector columns
    if isempty(vector_cols)
        println("❌ ERROR: No vector columns (starting with 'A') found in $input_csv.")
        println("   Skipping this file.")
        return
    end

    # Separate data and handle missing values
    vector_data_df = df[!, vector_cols]
    metadata = isempty(metadata_cols) ? DataFrame() : df[!, metadata_cols]
    X = Matrix(coalesce.(vector_data_df, 0.0))   # n×d
    n_samples, d = size(X)

    println("   ... Data separated ($n_samples samples, $(length(vector_cols)) features).")

    # === 2b. Analysis 1: Cosine Similarity Heatmap ===
    println("   ... Generating Heatmap")
    cosine_dist_matrix = pairwise(CosineDist(), X, dims=1)
    similarity_matrix = 1.0 .- cosine_dist_matrix

    hm = heatmap(
        similarity_matrix,
        title = "Cosine Similarity: $output_prefix",
        c = :viridis,
        aspect_ratio = :equal,
        xticks = false,
        yticks = false
    )
    heatmap_name = "results/$(output_prefix)_cosine_heatmap.png"
    savefig(hm, heatmap_name)
    println("   ✅ Saved '$heatmap_name'")

    # Create labels for plots (String-based helpers)
    labels = (!isempty(metadata) && hascol(metadata, "system:index") && hascol(metadata, "year")) ?
        string.(getcol(metadata, "system:index")) .* " (" .* string.(getcol(metadata, "year")) .* ")" :
        string.(1:n_samples)

    # === 2c. Analysis 2: Hierarchical Clustering ===
    if n_samples < 2
        println("   ... Skipping Dendrogram (need at least 2 samples).")
    elseif n_samples > 2000
        println("   ... Skipping Dendrogram (dataset > 2000 samples is too large).")
    else
        println("   ... Generating Dendrogram")
        dist_matrix = pairwise(Euclidean(), X, dims=1)
        hclust_result = hclust(dist_matrix, linkage=:ward)
        dg = plot(
            hclust_result,
            labels = labels,
            title = "Dendrogram: $output_prefix",
            yflip = true,    # leaves at bottom
            leaf_font_size = 8,
            rotation = 90
        )
        dendrogram_name = "results/$(output_prefix)_hierarchical_dendrogram.png"
        savefig(dg, dendrogram_name)
        println("   ✅ Saved '$dendrogram_name'")
    end

    # === 2d. Analysis 3: DBSCAN on top PCs (auto-tuned eps) & t-SNE on raw ===
    if n_samples <= 3
        println("   ... Skipping t-SNE & DBSCAN (need more than 3 samples).")
    else
        println("   ... Computing per-file PCA for DBSCAN")
        # Fit PCA with observations as columns
        Xt = transpose(X)                                         # d×n
        outdim = min(nPC, min(d, n_samples))
        pca_model = fit(PCA, Xt; maxoutdim=outdim)                # center=true
        Z = transpose(MultivariateStats.transform(pca_model, Xt)) # n×outdim (reduced)

        # Build KDTree on reduced space (features × samples)
        println("   ... Building KDTree on top $outdim PCs")
        data_t = permutedims(Z)                 # outdim × n
        tree = KDTree(data_t)

        # Heuristic min_neighbors and auto eps from k-distance curve
        min_neighbors = 10
        knn_k = min_neighbors + 1               # +1 so self is first neighbor to skip
        println("   ... Computing k-distance curve (k = $min_neighbors)")

        # Indices (ignored) and distances (matrix OR vector-of-vectors depending on NN version)
        idxs, dists = knn(tree, data_t, knn_k, true)  # true => sorted results

        # Extract the k-th neighbor distance for each sample (no sqrt: these are true distances)
        k_dists = dists isa AbstractMatrix ? vec(dists[knn_k, :]) : [di[knn_k] for di in dists]

        # Save k-distance plot
        try
            kd_sorted = sort(k_dists)
            plt_k = plot(1:length(kd_sorted), kd_sorted;
                         xlabel = "Points (sorted)",
                         ylabel = "k-distance (k = $min_neighbors)",
                         title  = "k-distance curve: $output_prefix")
            savefig(plt_k, "results/$(output_prefix)_kdist_curve.png")
            println("   ✅ Saved 'results/$(output_prefix)_kdist_curve.png'")
        catch e
            @warn "Failed to save k-distance plot: $e"
        end

        # Auto-tune eps from a high percentile (robust to outliers)
        eps_auto = quantile(k_dists, k_percentile)
        println("   ... Auto-chosen eps from k-distance @ p=$(round(k_percentile*100,digits=1))% → $(round(eps_auto,digits=5))")

        # Run DBSCAN on reduced space (observations as columns = outdim × n)
        println("   ... Running DBSCAN on top PCs")
        try
            db = dbscan(data_t, eps_auto; min_neighbors=min_neighbors)
            cluster_labels = db.assignments
            num_clusters = length(unique(string.(cluster_labels)))
            println("   ... DBSCAN complete. Found $num_clusters clusters (eps=$(round(eps_auto,digits=5)), min_neighbors=$min_neighbors).")
        catch e
            @warn "DBSCAN on reduced space failed ($e). Falling back to eps=1.0 on reduced space."
            eps_fallback = 1.0
            db = dbscan(data_t, eps_fallback; min_neighbors=min_neighbors)
            cluster_labels = db.assignments
            num_clusters = length(unique(string.(cluster_labels)))
            println("   ... DBSCAN fallback complete. Found $num_clusters clusters (eps=$(eps_fallback), min_neighbors=$min_neighbors).")
        end

        # t-SNE on raw space (as before)
        println("   ... Running t-SNE (this may take a minute...)")
        perplexity_val = clamp(30, 5, max(5, n_samples - 1))
        tsne_results = tsne(X, 2, perplexity_val, 500)  # (n_samples, 2)
        tsne_x = tsne_results[:, 1]
        tsne_y = tsne_results[:, 2]
        cluster_groups = string.(cluster_labels)

        tsne_plot = scatter(
            tsne_x, tsne_y;
            group = cluster_groups,
            title = "t-SNE ($output_prefix) - Colored by DBSCAN Cluster (on PCs)",
            xlabel = "t-SNE Component 1",
            ylabel = "t-SNE Component 2",
            markersize = 2,
            markeralpha = 0.5,
            markerstrokewidth = 0,
            legend = :outertopright
        )
        tsne_name = "results/$(output_prefix)_tsne_visualization_colored.png"
        savefig(tsne_plot, tsne_name)
        println("   ✅ Saved '$tsne_name'")
    end

    println("🏁 Finished analysis for $output_prefix.")
    return nothing
end

# -----------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------
files_to_analyze = [
    ("data/2019_embeddings_lantana_native.csv", "2019_native"),
    ("data/2019_embeddings_lantana_test.csv",   "2019_test"),
    ("data/2020_embeddings_lantana_native.csv", "2020_native"),
    ("data/2020_embeddings_lantana_test.csv",   "2020_test"),
    ("data/2021_embeddings_lantana_native.csv", "2021_native"),
    ("data/2021_embeddings_lantana_test.csv",   "2021_test"),
    ("data/2022_embeddings_lantana_native.csv", "2022_native"),
    ("data/2022_embeddings_lantana_test.csv",   "2022_test"),
    ("data/2023_embeddings_lantana_native.csv", "2023_native"),
    ("data/2023_embeddings_lantana_test.csv",   "2023_test"),
    ("data/2024_embeddings_lantana_native.csv", "2024_native"),
    ("data/2024_embeddings_lantana_test.csv",   "2024_test"),
]

# Per-file analyses (DBSCAN on top PCs with auto eps)
for (file_path, prefix) in files_to_analyze
    Base.invokelatest(analyze_vectors, file_path, prefix; nPC=30, k_percentile=0.98)
end

# Joint comparison across all files
# Set `standardize=true` if features may be on different scales across files.
compare_datasets(files_to_analyze; k=10, standardize=false)

println("\n🎉 All analyses complete.")

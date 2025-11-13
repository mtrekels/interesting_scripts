# -----------------------------------------------------------------
# 1) SETUP: INSTALL AND LOAD PACKAGES
# -----------------------------------------------------------------
using Pkg

# Activate the local environment
Pkg.activate(".")

# Idempotent add
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
using Distances             # pairwise distances, CosineDist
using Clustering            # hclust, dbscan
using Plots                 # plotting
using StatsPlots            # (we'll use hexbin if backend supports it)
using TSne                  # t-SNE
using MultivariateStats     # PCA
using Statistics            # mean, std, quantile
using Random                # seeding
using LinearAlgebra         # eigen, cov
using Distributions         # chi-square quantiles (ellipses)
using NearestNeighbors      # KDTree / kNN

# Output dir & plot defaults
isdir("results") || mkpath("results")
default(size=(1000, 850), dpi=150)

# Reproducibility
Random.seed!(42)

println("Julia packages loaded successfully.")

# -----------------------------------------------------------------
# 2) HELPERS (METADATA COLUMNS IGNORED)
# -----------------------------------------------------------------

# Parse (year, condition) from a label like "2020_native"
function parse_year_condition(lbl::AbstractString)
    m = match(r"\b(\d{4})\b", lbl)
    year = isnothing(m) ? missing : try parse(Int, m.captures[1]) catch; missing end
    parts = split(lbl, "_")
    cond  = length(parts) >= 2 ? parts[end] : missing
    return year, cond
end

# Fallback: parse from filename like ".../2020_embeddings_lantana_test.csv"
function parse_year_condition_from_path(path::AbstractString)
    base = splitpath(path)[end]
    m1 = match(r"(\d{4})", base)
    yr = isnothing(m1) ? missing : try parse(Int, m1.captures[1]) catch; missing end
    m2 = match(r"_([A-Za-z0-9]+)\.csv$", base)
    cond = isnothing(m2) ? missing : m2.captures[1]
    return yr, cond
end

# Load vectors only (rows = samples, cols = features). Ignore all other columns if present.
# Also return a tiny meta DataFrame with a single column :dataset (string label).
function load_vectors_only(path::String, dataset_label::String)
    df = CSV.read(path, DataFrame)
    name_list = String.(names(df))
    vector_cols = [c for c in name_list if startswith(c, "A")]
    if isempty(vector_cols)
        error("No vector columns (starting with 'A') found in $path")
    end
    X = Matrix(coalesce.(df[:, vector_cols], 0.0))   # n×d
    meta = DataFrame(dataset = fill(dataset_label, size(X,1)))
    return X, meta, vector_cols
end

# Centroids (mean along rows) by group string
function centroids(scores::AbstractMatrix, groups::Vector{<:AbstractString})
    @assert size(scores, 1) == length(groups)
    G = unique(groups)
    bygroup = Dict{String, Vector{Float64}}()
    for g in G
        idx = findall(==(g), groups)
        bygroup[String(g)] = vec(mean(scores[idx, :], dims=1))
    end
    return bygroup
end

# Cosine similarity via Distances.jl
cos_sim(u::AbstractVector, v::AbstractVector) = 1 - (CosineDist())(u, v)

# 95% covariance ellipse coords for 2D
function ellipse95(μ::Vector{<:Real}, Σ::AbstractMatrix; npts=200)
    r2 = quantile(Chisq(2), 0.95)
    evals, evecs = eigen(Symmetric(Σ))
    evals = max.(evals, 0)
    R = evecs * Diagonal(sqrt.(evals))
    θ = range(0, 2π; length=npts)
    circ = [cos.(θ)'; sin.(θ)']
    E = (sqrt(r2) .* (R * circ)) .+ μ
    return E[1, :], E[2, :]
end

# -----------------------------------------------------------------
# 3) PER-FILE ANALYSIS
# -----------------------------------------------------------------
"""
    analyze_file(input_csv, dataset_label; nPC=30, k_percentile=0.98)

Per-file analysis that ignores any metadata columns in the CSV:
- Cosine similarity heatmap on vectors (A* columns)
- PCA -> KDTree -> k-distance curve (saved)
- DBSCAN on top PCs with auto-ε from k-distance percentile
- t-SNE on raw vectors, colored by DBSCAN cluster
"""
function analyze_file(input_csv::String, dataset_label::String; nPC::Int=30, k_percentile::Float64=0.98)
    println("-------------------------------------------------")
    println("🚀 Starting Analysis for: $input_csv")
    println("   Saving outputs with prefix: $dataset_label")
    println("-------------------------------------------------")

    # Load vectors only
    X, _meta, _vcols = load_vectors_only(input_csv, dataset_label)
    n, d = size(X)
    println("   ... Data separated ($n samples, $d features).")

    # Cosine similarity heatmap
    println("   ... Generating Heatmap")
    cosine_dist_matrix = pairwise(CosineDist(), X, dims=1)
    similarity_matrix = 1.0 .- cosine_dist_matrix
    hm = heatmap(similarity_matrix; title="Cosine Similarity: $dataset_label", c=:viridis, xticks=false, yticks=false)
    savefig("results/$(dataset_label)_cosine_heatmap.png")
    println("   ✅ Saved 'results/$(dataset_label)_cosine_heatmap.png'")

    # PCA to reduce for DBSCAN
    println("   ... PCA for DBSCAN")
    Xt = transpose(X)                                        # d×n
    outdim = min(nPC, min(d, n))
    pca_model = fit(PCA, Xt; maxoutdim=outdim)               # center=true default
    Z = transpose(MultivariateStats.transform(pca_model, Xt))# n×outdim
    data_t = permutedims(Z)                                  # outdim × n (features × samples)

    # k-distance curve
    println("   ... Building KDTree & k-distance (k=10)")
    tree = KDTree(data_t)
    min_neighbors = 10
    knn_k = min_neighbors + 1
    idxs, dists = knn(tree, data_t, knn_k, true)             # older API: positional sortres
    k_dists = dists isa AbstractMatrix ? vec(dists[knn_k, :]) : [di[knn_k] for di in dists]
    kd_sorted = sort(k_dists)
    plt_k = plot(1:length(kd_sorted), kd_sorted;
                 xlabel="Points (sorted)", ylabel="k-distance (k = $min_neighbors)",
                 title="k-distance curve: $dataset_label")
    savefig(plt_k, "results/$(dataset_label)_kdist_curve.png")
    println("   ✅ Saved 'results/$(dataset_label)_kdist_curve.png'")

    eps = quantile(k_dists, k_percentile)
    println("   ... Auto-ε @ p=$(round(k_percentile*100,digits=1))% → $(round(eps,digits=5))")

    # DBSCAN on reduced space
    cluster_labels = ones(Int, n)  # predeclare for safety
    num_clusters = 1
    println("   ... Running DBSCAN on top PCs")
    try
        db = dbscan(data_t, eps; min_neighbors=min_neighbors)
        cluster_labels = db.assignments
        num_clusters = length(unique(string.(cluster_labels)))
        println("   ... DBSCAN complete. Found $num_clusters clusters (eps=$(round(eps,digits=5))).")
    catch e
        @warn "DBSCAN failed ($e). Falling back to eps=1.0."
        db = dbscan(data_t, 1.0; min_neighbors=min_neighbors)
        cluster_labels = db.assignments
        num_clusters = length(unique(string.(cluster_labels)))
        println("   ... DBSCAN fallback complete. Found $num_clusters clusters (eps=1.0).")
    end

    # t-SNE on raw space, colored by DBSCAN cluster
    println("   ... Running t-SNE (this may take a minute...)")
    perplexity_val = clamp(30, 5, max(5, n - 1))
    tsne_results = tsne(X, 2, perplexity_val, 500)   # (n, 2)
    tsne_x = tsne_results[:, 1]; tsne_y = tsne_results[:, 2]
    tsne_plot = scatter(tsne_x, tsne_y; group=string.(cluster_labels),
                        title="t-SNE ($dataset_label) - Colored by DBSCAN Cluster",
                        xlabel="t-SNE Component 1", ylabel="t-SNE Component 2",
                        markersize=2, markeralpha=0.5, markerstrokewidth=0, legend=:outertopright)
    savefig("results/$(dataset_label)_tsne_visualization_colored.png")
    println("   ✅ Saved 'results/$(dataset_label)_tsne_visualization_colored.png'")

    println("🏁 Finished analysis for $dataset_label.")
    return nothing
end

# -----------------------------------------------------------------
# 4) CROSS-DATASET COMPARISON (JOINT PCA; year/condition from label or filename)
# -----------------------------------------------------------------
"""
    compare_datasets(file_list; k=10, standardize=false)

Joint PCA across all files. Ignores any metadata columns in the CSVs.
Parses (year, condition) from the provided dataset label (e.g., "2020_native")
with a fallback that parses the actual filename.

Outputs:
- results/cross_pca_scatter_pc1_pc2.png
- results/cross_pca_hexbin_pc1_pc2.png   (or 2D hist fallback)
- results/cross_pca_centroids_pc1_pc2.png
- results/cross_pca_centroids_ellipses.png
- results/cross_pca_trajectories_pc1_pc2.png
- results/cross_pca_explained_variance.png
- results/cross_centroid_cosine_topK.csv
- results/cross_pca_pc1_centroids.csv
- results/cross_pca_pc1_evolution.png
- results/cross_pca_centroid_shifts.csv
- results/prominent_vectors_summary.csv
- results/prominent_vectors_<dataset>.csv
- results/pc_loadings_full.csv
- results/pc1_top_features.csv
- results/pc2_top_features.csv
"""
function compare_datasets(file_list::Vector{Tuple{String,String}}; k::Int=10, standardize::Bool=false)
    println("\n-------------------------------------------------")
    println("🔗 Cross-dataset analysis (PCA)")
    println("   (year/condition parsed from label like '2020_native', fallback to filename)")
    println("-------------------------------------------------")

    # Load all vectors; dataset label per row; ignore metadata cols
    matrices   = Matrix{Float64}[]
    metas      = DataFrame[]
    vcols_all  = Vector{String}[]
    for (path, label) in file_list
        try
            X, meta, vcols = load_vectors_only(path, label)
            push!(matrices, X)
            push!(metas, meta)         # only :dataset column
            push!(vcols_all, vcols)
            println("   ... loaded $(path)  →  $(size(X,1)) samples, $(size(X,2)) features")
        catch e
            @warn "Skipping $path due to error: $e"
        end
    end
    isempty(matrices) && (println("❌ No datasets loaded."); return)

    # Feature names reference
    feature_names = vcols_all[1]

    X_all    = vcat(matrices...)         # n×d
    meta_all = vcat(metas...)            # n×1 (:dataset)
    n, d = size(X_all)
    println("   ... combined matrix: $n samples, $d features")

    # Labels from label strings
    dataset_labels = String.(meta_all.dataset)

    # Parse year/condition using label, fallback to filename for the rows of that file
    file_year  = Vector{Union{Missing,Int}}(undef, n)
    condition  = Vector{Union{Missing,String}}(undef, n)

    row_start = 1
    for (idx_file, (path, label)) in enumerate(file_list)
        Xrows = size(matrices[idx_file], 1)
        row_end = row_start + Xrows - 1

        y1, c1 = parse_year_condition(label)
        y2, c2 = parse_year_condition_from_path(path)

        y = ismissing(y1) ? y2 : y1
        c = ismissing(c1) ? c2 : c1

        file_year[row_start:row_end] .= y
        condition[row_start:row_end] .= ismissing(c) ? missing : String(c)

        row_start = row_end + 1
    end
    n_valid = sum(.!(ismissing.(file_year)) .& .!(ismissing.(condition)))
    println("   ... parsed (condition,year) for $n_valid / $n rows")

    # Optional standardization across all rows
    X_for_pca = if standardize
        Xμ = mean(X_all, dims=1)
        Xσ = std(X_all, dims=1; corrected=true); Xσ[Xσ .== 0] .= 1.0
        (X_all .- Xμ) ./ Xσ
    else
        X_all
    end

    # PCA (observations as columns)
    Xt = transpose(X_for_pca)                       # d×n
    maxout = min(k, min(d, n))
    pca_model = fit(PCA, Xt; maxoutdim=maxout)
    scores = transpose(MultivariateStats.transform(pca_model, Xt))  # n×p
    p = size(scores, 2)
    prinvars = pca_model.prinvars
    evr = prinvars ./ sum(prinvars)

    # -------- Prominent vectors (PC1 extremes & representative) per dataset label --------
    try
        pc1 = scores[:, 1]
        Krep = min(5, p)
        repspace = scores[:, 1:Krep]

        summ = DataFrame(dataset=String[],
                         prominent_abs_pc1_index=Int[], prominent_abs_pc1_value=Float64[],
                         prominent_pos_pc1_index=Int[], prominent_pos_pc1_value=Float64[],
                         prominent_neg_pc1_index=Int[], prominent_neg_pc1_value=Float64[],
                         representative_index=Int[], representative_dist=Float64[],
                         year_abs=Union{Missing,Int}[], year_pos=Union{Missing,Int}[],
                         year_neg=Union{Missing,Int}[], year_rep=Union{Missing,Int}[],
                         condition_abs=Union{Missing,String}[], condition_pos=Union{Missing,String}[],
                         condition_neg=Union{Missing,String}[], condition_rep=Union{Missing,String}[])

        for ds in sort(unique(dataset_labels))
            idxs = findall(==(ds), dataset_labels)

            # Leaders on PC1
            absvals = abs.(pc1[idxs])
            i_abs_local = argmax(absvals); i_abs = idxs[i_abs_local]; v_abs = pc1[i_abs]
            i_pos = idxs[argmax(pc1[idxs])]; v_pos = pc1[i_pos]
            i_neg = idxs[argmin(pc1[idxs])]; v_neg = pc1[i_neg]

            # Representative (medoid) in first Krep PCs
            μ = vec(mean(repspace[idxs, :], dims=1))
            d2 = [sum((repspace[j, :] .- μ).^2) for j in idxs]
            i_rep_local = argmin(d2); i_rep = idxs[i_rep_local]; rep_dist = sqrt(d2[i_rep_local])

            push!(summ, (ds, i_abs, v_abs, i_pos, v_pos, i_neg, v_neg, i_rep, rep_dist,
                         file_year[i_abs], file_year[i_pos], file_year[i_neg], file_year[i_rep],
                         condition[i_abs], condition[i_pos], condition[i_neg], condition[i_rep]))

            # Per-dataset CSV (indices & file-derived year/condition)
            ds_out = DataFrame(
                role         = ["abs_pc1_leader", "pos_pc1_leader", "neg_pc1_leader", "representative_medoid"],
                global_index = [i_abs, i_pos, i_neg, i_rep],
                year_file    = [file_year[i_abs], file_year[i_pos], file_year[i_neg], file_year[i_rep]],
                condition    = [condition[i_abs], condition[i_pos], condition[i_neg], condition[i_rep]],
                PC1_score    = [v_abs, v_pos, v_neg, pc1[i_rep]]
            )
            CSV.write("results/prominent_vectors_$(ds).csv", ds_out)
        end

        CSV.write("results/prominent_vectors_summary.csv", summ)
        println("   ✅ Saved results/prominent_vectors_summary.csv and per-dataset CSVs")
    catch e
        @warn "Failed to compute/save prominent vectors: $e"
    end

    # -------- Feature loadings & top features --------
    try
        W = MultivariateStats.projection(pca_model)  # d×p
        feats = feature_names
        dfW = DataFrame(feature = feats)
        for j in 1:size(W,2)
            dfW[!, Symbol("PC$(j)")] = W[:, j]
        end
        CSV.write("results/pc_loadings_full.csv", dfW)
        println("   ✅ Saved results/pc_loadings_full.csv")

        function top_features(W::AbstractMatrix, pc::Int; N::Int=30)
            vals = W[:, pc]
            idx = sortperm(abs.(vals); rev=true)[1:min(N, length(vals))]
            DataFrame(rank = 1:length(idx), feature = feats[idx], loading = vals[idx], abs_loading = abs.(vals[idx]))
        end
        CSV.write("results/pc1_top_features.csv", top_features(W, 1; N=30))
        println("   ✅ Saved results/pc1_top_features.csv")
        if size(W,2) >= 2
            CSV.write("results/pc2_top_features.csv", top_features(W, 2; N=30))
            println("   ✅ Saved results/pc2_top_features.csv")
        end
    catch e
        @warn "Failed to compute/save loadings: $e"
    end

    # -------- Visuals --------
    try
        plt1 = scatter(scores[:,1], scores[:,2];
                       group=dataset_labels,
                       title="Cross-dataset PCA: PC1 vs PC2",
                       xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                       ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)",
                       markersize=1.5, markeralpha=0.2, markerstrokewidth=0,
                       legend=:outertopright)
        savefig(plt1, "results/cross_pca_scatter_pc1_pc2.png")
        println("   ✅ Saved results/cross_pca_scatter_pc1_pc2.png")
    catch e
        @warn "Failed to save scatter: $e"
    end

    # Hexbin if supported; otherwise 2D histogram heatmap fallback
    try
        plt_hex = nothing
        try
            plt_hex = hexbin(scores[:,1], scores[:,2];
                             nbins=60,
                             title="PC1–PC2 density (hexbin)",
                             xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                             ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)")
        catch
            x = scores[:,1]; y = scores[:,2]
            nb = 120
            xmin, xmax = extrema(x); ymin, ymax = extrema(y)
            if xmin == xmax; xmin -= 1e-6; xmax += 1e-6; end
            if ymin == ymax; ymin -= 1e-6; ymax += 1e-6; end
            xb = range(xmin, xmax; length=nb+1)
            yb = range(ymin, ymax; length=nb+1)
            H = zeros(nb, nb)
            for i in eachindex(x)
                ix = clamp(searchsortedlast(xb, x[i]), 1, nb)
                iy = clamp(searchsortedlast(yb, y[i]), 1, nb)
                H[iy, ix] += 1
            end
            plt_hex = heatmap(
                collect(xb[1:end-1]), collect(yb[1:end-1]), H';
                xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)",
                title="PC1–PC2 density (2D histogram)"
            )
        end
        savefig(plt_hex, "results/cross_pca_hexbin_pc1_pc2.png")
        println("   ✅ Saved results/cross_pca_hexbin_pc1_pc2.png")
    catch e
        @warn "Failed to save density plot altogether: $e"
    end

    # Centroids by dataset label
    try
        cent_by_ds = centroids(scores[:,1:2], dataset_labels)
        ds_names = sort(collect(keys(cent_by_ds)))
        cx = [cent_by_ds[ds][1] for ds in ds_names]
        cy = [cent_by_ds[ds][2] for ds in ds_names]
        plt2 = scatter(cx, cy; seriestype=:scatter, markershape=:circle,
                       title="Centroids by dataset (PC1–PC2)",
                       xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                       ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)",
                       legend=false, markersize=8)
        for (i, name) in pairs(ds_names)
            annotate!(cx[i], cy[i], text(name, 9, :left))
        end
        savefig(plt2, "results/cross_pca_centroids_pc1_pc2.png")
        println("   ✅ Saved results/cross_pca_centroids_pc1_pc2.png")
    catch e
        @warn "Failed to save centroids: $e"
    end

    # 95% ellipses by dataset label
    try
        plt_ell = plot(title="Centroids with 95% ellipses (PC1–PC2)",
                       xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                       ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)")
        for ds in sort(unique(dataset_labels))
            idx = findall(==(ds), dataset_labels)
            X2 = scores[idx, 1:2]
            μ = vec(mean(X2, dims=1)); Σ = cov(X2; corrected=true)
            ex, ey = ellipse95(μ, Σ)
            plot!(ex, ey; label=false)
            scatter!([μ[1]], [μ[2]]; label=ds, markersize=7)
        end
        savefig(plt_ell, "results/cross_pca_centroids_ellipses.png")
        println("   ✅ Saved results/cross_pca_centroids_ellipses.png")
    catch e
        @warn "Failed to save ellipses: $e"
    end

    # Condition-wise trajectories by file year (PC1–PC2)
    try
        plt_tr = plot(title="Centroid trajectories by condition (PC1–PC2)",
                      xlabel="PC1 ($(round(evr[1]*100, digits=1))% var)",
                      ylabel="PC2 ($(round(evr[2]*100, digits=1))% var)")
        have_any = false
        for cond in sort(unique(skipmissing(condition)))
            idxs = findall(==(cond), condition)
            ys = file_year[idxs]
            valid = .!(ismissing.(ys))
            isempty(findall(valid)) && continue

            sub = DataFrame(pc1 = scores[idxs[valid], 1],
                            pc2 = scores[idxs[valid], 2],
                            year = Int.(ys[valid]))
            years = sort(unique(sub.year))
            xs = Float64[]; ys2 = Float64[]
            for y in years
                s = sub[sub.year .== y, :]
                μ = vec(mean(Matrix(s[:, [:pc1, :pc2]]), dims=1))
                push!(xs, μ[1]); push!(ys2, μ[2])
            end
            plot!(xs, ys2; label=cond, markershape=:circle, markersize=6)
            if length(xs) > 1
                dx = diff(xs); dy = diff(ys2)
                quiver!(xs[1:end-1], ys2[1:end-1]; quiver=(dx, dy), arrow=true, label=false)
            end
            for (i, y) in enumerate(years)
                annotate!(xs[i], ys2[i], text(string(y), 9, :left))
            end
            have_any = true
        end
        if have_any
            savefig(plt_tr, "results/cross_pca_trajectories_pc1_pc2.png")
            println("   ✅ Saved results/cross_pca_trajectories_pc1_pc2.png")
        else
            @warn "No valid (condition,year) combo for trajectories."
        end
    catch e
        @warn "Failed to save trajectories: $e"
    end

    # Explained variance
    try
        m = min(10, length(evr))
        plt_var = bar(1:m, evr[1:m];
                      xlabel="Principal component",
                      ylabel="Explained variance ratio",
                      title="Explained variance (top $m PCs)")
        savefig(plt_var, "results/cross_pca_explained_variance.png")
        println("   ✅ Saved results/cross_pca_explained_variance.png")
    catch e
        @warn "Failed to save explained variance: $e"
    end

    # Cosine similarity between dataset centroids (top K PCs)
    try
        topK = min(5, size(scores, 2))
        centK = centroids(scores[:, 1:topK], dataset_labels)
        ds = sort(collect(keys(centK)))
        S = Matrix{Float64}(undef, length(ds), length(ds))
        for i in eachindex(ds), j in eachindex(ds)
            S[i, j] = cos_sim(centK[ds[i]], centK[ds[j]])
        end
        out = DataFrame([:dataset => ds])
        for (j, name) in enumerate(ds)
            out[!, Symbol(name)] = S[:, j]
        end
        CSV.write("results/cross_centroid_cosine_topK.csv", out)
        println("   ✅ Saved results/cross_centroid_cosine_topK.csv")
    catch e
        @warn "Failed to save centroid cosine table: $e"
    end

    # PC1 centroid per dataset label + rank (dominant component)
    try
        pc1_scores = scores[:, 1]
        cent_pc1 = centroids(reshape(pc1_scores, :, 1), dataset_labels)
        ds = sort(collect(keys(cent_pc1)))
        pc1vals = [cent_pc1[d][1] for d in ds]
        order = sortperm(pc1vals; rev=true)
        inv = invperm(order)
        report = DataFrame(dataset = ds, PC1_centroid = pc1vals, PC1_rank = inv)
        CSV.write("results/cross_pca_pc1_centroids.csv", report)
        println("   ✅ Saved results/cross_pca_pc1_centroids.csv")
    catch e
        @warn "Failed to save PC1 centroid report: $e"
    end

    # PC1 evolution by condition (file-year)
    try
        valid = .!(ismissing.(file_year)) .& .!(ismissing.(condition))
        if any(valid)
            df_evo = DataFrame(cond = String.(condition[valid]),
                               year = Int.(file_year[valid]),
                               pc1  = scores[valid, 1])
            gk = groupby(df_evo, [:cond, :year])
            evo = combine(gk, :pc1 => mean => :pc1_mean)
            conds = sort(unique(evo.cond))
            plt_evo = plot(title="PC1 centroid over years by condition",
                           xlabel="Year", ylabel="PC1 centroid")
            for c in conds
                sub = evo[evo.cond .== c, :]
                ys = sort(unique(sub.year))
                μ = [sub[sub.year .== y, :pc1_mean][1] for y in ys]
                plot!(ys, μ; label=c, markershape=:circle, markersize=6)
            end
            savefig(plt_evo, "results/cross_pca_pc1_evolution.png")
            println("   ✅ Saved results/cross_pca_pc1_evolution.png")
        else
            @warn "No valid (condition,year) for PC1 evolution."
        end
    catch e
        @warn "Failed to save PC1 evolution: $e"
    end

    # Condition-wise centroid shift (earliest → latest year) in (PC1, PC2)
    try
        valid = .!(ismissing.(file_year)) .& .!(ismissing.(condition))
        if any(valid)
            df = DataFrame(cond = String!(copy(condition[valid])),
                           year = Int.(file_year[valid]),
                           pc1  = scores[valid, 1],
                           pc2  = scores[valid, 2])
            g = groupby(df, [:cond, :year])
            cent = combine(g, [:pc1, :pc2] .=> mean .=> [:pc1_mean, :pc2_mean])
            out = DataFrame(cond=String[], year_start=Int[], year_end=Int[],
                            d_pc1=Float64[], d_pc2=Float64[], shift_norm=Float64[], angle_deg=Float64[])
            for c in unique(cent.cond)
                sub = cent[cent.cond .== c, :]
                yrs = sort(unique(sub.year))
                if length(yrs) >= 2
                    a = sub[sub.year .== first(yrs), [:pc1_mean, :pc2_mean]]
                    b = sub[sub.year .== last(yrs),  [:pc1_mean, :pc2_mean]]
                    d1 = b.pc1_mean[1] - a.pc1_mean[1]
                    d2 = b.pc2_mean[1] - a.pc2_mean[1]
                    push!(out, (c, first(yrs), last(yrs), d1, d2, hypot(d1,d2), atan(d2, d1)*180/pi))
                end
            end
            CSV.write("results/cross_pca_centroid_shifts.csv", out)
            println("   ✅ Saved results/cross_pca_centroid_shifts.csv")
        end
    catch e
        @warn "Failed to save centroid shift summary: $e"
    end

    println("🏁 Finished cross-dataset PCA (filename-derived year/condition; metadata ignored).")
    return nothing
end

# -----------------------------------------------------------------
# 5) MAIN
# -----------------------------------------------------------------
# IMPORTANT: The label passed as the second tuple element should be "YYYY_condition"
# to parse year/condition correctly (e.g., "2020_native", "2020_test").
files_to_analyze = [
    ("data/Acacia_saligna/2019_embeddings_acacia_saligna_native.csv", "2019_native"),
    ("data/Acacia_saligna/2019_embeddings_africa.csv",   "2019_test"),
    ("data/Acacia_saligna/2019_embeddings_eu.csv",   "2019_eu"),
    ("data/Acacia_saligna/2020_embeddings_acacia_saligna_native.csv", "2020_native"),
    ("data/Acacia_saligna/2020_embeddings_africa.csv",   "2020_test"),
    ("data/Acacia_saligna/2020_embeddings_eu.csv",   "2020_eu"),
    ("data/Acacia_saligna/2021_embeddings_acacia_saligna_native.csv", "2021_native"),
    ("data/Acacia_saligna/2021_embeddings_africa.csv",   "2021_test"),
    ("data/Acacia_saligna/2021_embeddings_eu.csv",   "2021_eu"),
    ("data/Acacia_saligna/2022_embeddings_acacia_saligna_native.csv", "2022_native"),
    ("data/Acacia_saligna/2022_embeddings_africa.csv",   "2022_test"),
    ("data/Acacia_saligna/2022_embeddings_eu.csv",   "2022_eu"),
    ("data/Acacia_saligna/2023_embeddings_acacia_saligna_native.csv", "2023_native"),
    ("data/Acacia_saligna/2023_embeddings_africa.csv",   "2023_test"),
    ("data/Acacia_saligna/2023_embeddings_eu.csv",   "2023_eu"),
    #("data/2024_embeddings_lantana_native.csv", "2024_native"),
    #("data/2024_embeddings_lantana_test.csv",   "2024_test"),
]

# Run per-file analyses
for (path, label) in files_to_analyze
    Base.invokelatest(analyze_file, path, label; nPC=30, k_percentile=0.98)
end

# Joint comparison across all files
compare_datasets(files_to_analyze; k=10, standardize=false)

println("\n🎉 All analyses complete.")

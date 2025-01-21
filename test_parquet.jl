import GeoParquet as GP
using DataFrames
using GeoMakie
using ColorSchemes
using LibGEOS  # For WKB decoding
using GeometryBasics  # For geometry handling
using ArchGDAL

data_path = "/home/maarten/Documents/GIT/interesting_scripts/data/data_ZA.parquet";

# Read GeoParquet data
gdf = GP.read(data_path)

# Filter rows based on specieskey
filtered_gdf = gdf[.!ismissing.(gdf.specieskey) .&& (gdf.specieskey .== 2435350.0), :]

# Function to convert LibGEOS geometries to GeometryBasics geometries
function convert_to_geometrybasics(geom::LibGEOS.AbstractGeometry)
    if geom isa LibGEOS.Polygon
        # Convert a single polygon
        rings = [GeometryBasics.Point2.(ring.coords[:, 1], ring.coords[:, 2]) for ring in geom.rings]
        return GeometryBasics.Polygon(rings...)
    elseif geom isa LibGEOS.MultiPolygon
        # Convert a multipolygon by iterating over its components
        n_geometries = LibGEOS.getngeometry(geom)  # Get number of geometries
        sub_geometries = [convert_to_geometrybasics(LibGEOS.getgeometry(geom, i - 1)) for i in 1:n_geometries]
        return GeometryBasics.MultiPolygon(sub_geometries)
    else
        error("Unsupported geometry type: $(typeof(geom))")
    end
end

# Convert WKB geometries to GeometryBasics geometries
geometries = [
    convert_to_geometrybasics(LibGEOS.readgeom(geom.val))
    for geom in filtered_gdf.geometry
]

# Extract the 'count' column for coloring
counts = filtered_gdf.occurrences

# Normalize the color scale
normalized_counts = (counts .- minimum(counts)) ./ (maximum(counts) - minimum(counts))

# Create a map with a geographic projection
fig = Figure()
ax = geoaxis(fig[1, 1], projection = :mercator)

# Plot filtered geometries with colors based on 'count'
poly!(ax, geometries, 
    color = normalized_counts,  # Color by normalized count values
    colormap = :viridis,        # Choose a colormap (e.g., :viridis, :plasma)
    transparency = false
)

fig

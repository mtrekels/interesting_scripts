import rasterio
import numpy as np
import folium
from rasterio.warp import transform_bounds
from matplotlib import colormaps
from matplotlib.colors import Normalize
import base64
import io
import matplotlib.pyplot as plt

# --- Parameters ---
raster_path = "species_occurrences_total.tif"
target_species_key = 5406695.0  # NEW speciesKey

# --- Open and extract target band ---
with rasterio.open(raster_path) as src:
    band_index = None
    for i, desc in enumerate(src.descriptions):
        if desc == str(target_species_key):
            band_index = i + 1
            break
    if band_index is None:
        raise ValueError(f"SpeciesKey {target_species_key} not found.")
    
    band_data = src.read(band_index)
    bounds_proj = src.bounds
    bounds_latlon = transform_bounds(src.crs, "EPSG:4326", *bounds_proj)

# --- Normalize and apply colormap ---
valid_data = band_data[band_data > 0]
if valid_data.size == 0:
    raise ValueError("No data values greater than 0 to display.")
    
norm = Normalize(vmin=valid_data.min(), vmax=valid_data.max())
cmap = colormaps["viridis"]
rgba = (cmap(norm(band_data)) * 255).astype(np.uint8)  # RGBA image

# --- Create Folium map with OpenStreetMap tiles ---
center = [(bounds_latlon[1] + bounds_latlon[3]) / 2,
          (bounds_latlon[0] + bounds_latlon[2]) / 2]

m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")

# --- Add raster overlay ---
img_overlay = folium.raster_layers.ImageOverlay(
    image=rgba,
    bounds=[[bounds_latlon[1], bounds_latlon[0]],
            [bounds_latlon[3], bounds_latlon[2]]],
    opacity=0.7,
    name=f"SpeciesKey {target_species_key}",
    interactive=True
)
img_overlay.add_to(m)

# --- Add colorbar legend ---
def colormap_legend(norm, cmap, title):
    fig, ax = plt.subplots(figsize=(4, 0.5))
    fig.subplots_adjust(bottom=0.5)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.set_label(title)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    html = f'<img src="data:image/png;base64,{b64}">'
    return html

legend_html = colormap_legend(norm, cmap, "Occurrence Count")
legend = folium.Element(legend_html)
legend_container = folium.map.LayerControl(position='bottomright')
m.get_root().html.add_child(folium.Element(f"""
    <div style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        background-color: white;
        padding: 8px;
        border: 2px solid gray;
        border-radius: 5px;
    ">{legend_html}</div>
"""))

# --- Add layer control and save ---
folium.LayerControl().add_to(m)
m.save("species_5406695_map.html")
m

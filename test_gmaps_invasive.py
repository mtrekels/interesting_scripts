import os, re, time, math, requests, pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, linemerge

API_KEY = os.environ["GMAPS_API"]

def street_to_linestring(place_name: str, street_name: str) -> LineString:
    G = ox.graph_from_place(place_name, network_type="all")  # 'all' catches service/footways
    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # normalize 'name' safely
    def norm(v):
        if isinstance(v, (list, tuple, set)): return ", ".join(map(str, v))
        return "" if v is None else str(v)

    names = gdf["name"].apply(norm) if "name" in gdf else pd.Series("", index=gdf.index)
    pat = rf"\b{re.escape(street_name)}\b"
    mask = names.str.contains(pat, case=False, regex=True, na=False)
    if not mask.any():
        mask = names.str.contains(re.escape(street_name), case=False, regex=True, na=False)

    street_edges = gdf.loc[mask]
    if street_edges.empty:
        raise ValueError(f"Street '{street_name}' not found in {place_name}")

    lines = []
    for geom in street_edges.geometry:
        if isinstance(geom, MultiLineString): lines.extend(geom.geoms)
        elif geom is not None: lines.append(geom)
    merged = linemerge(unary_union(lines))
    if isinstance(merged, MultiLineString):
        merged = max(merged.geoms, key=lambda g: g.length)
    return merged

def project_to_metric(line_ll: LineString) -> tuple[LineString, gpd.GeoSeries]:
    g = gpd.GeoSeries([line_ll], crs="EPSG:4326").to_crs(3857)  # Web Mercator meters
    return g.iloc[0], g

def densify_by_meters(line_ll: LineString, step_m: int = 20) -> list[Point]:
    line_m, _ = project_to_metric(line_ll)
    pts_m = [line_m.interpolate(d) for d in range(0, int(line_m.length), step_m)]
    pts_m.append(line_m.interpolate(line_m.length))
    # back to WGS84
    g_m = gpd.GeoSeries(pts_m, crs="EPSG:3857").to_crs(4326)
    return [Point(p.x, p.y) for p in g_m]  # shapely Points in lon/lat order

def bearing(p1: Point, p2: Point):
    dlon = math.radians(p2.x - p1.x)
    y = math.sin(dlon) * math.cos(math.radians(p2.y))
    x = (math.cos(math.radians(p1.y))*math.sin(math.radians(p2.y)) -
         math.sin(math.radians(p1.y))*math.cos(math.radians(p2.y))*math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def streetview_metadata(lat, lng):
    r = requests.get(
        "https://maps.googleapis.com/maps/api/streetview/metadata",
        params={"location": f"{lat},{lng}", "key": API_KEY}
    )
    r.raise_for_status()
    return r.json()

def streetview_image(pano_id, heading, size="640x640"):
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "pano": pano_id,
        "size": size,
        "heading": heading,
        "key": API_KEY,
        "return_error_code": "true",
    }
    return url, params

# --- Your case ---
line = street_to_linestring("Rotselaar, Belgium", "Varentstraat")
pts = densify_by_meters(line, step_m=20)

rows = []
for i, pt in enumerate(pts[:-1]):
    lat, lng = pt.y, pt.x  # shapely Point is (x=lon, y=lat)
    meta = streetview_metadata(lat, lng)
    if meta.get("status") == "OK" and meta.get("pano_id"):
        rows.append({
            "lat": lat, "lng": lng,
            "pano_id": meta.get("pano_id"),
            "date": meta.get("date"),
            "heading": bearing(pt, pts[i+1])
        })
    time.sleep(0.05)

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError(
        "Still no panos. Likely no coverage on this street segment, or key/billing issue. "
        "Try step_m=10..30, network_type='all', or probe a nearby main road."
    )

df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
df = (df.sort_values("date", na_position="last")
        .drop_duplicates("pano_id", keep="first")
        .reset_index(drop=True))

print(df.head())

# Example: download the first 10 images
for _, r in df.head(10).iterrows():
    url, params = streetview_image(r.pano_id, r.heading)
    img = requests.get(url, params=params)
    if img.status_code == 200:
        with open(f"streetview_{r.pano_id}_{r.date}.jpg", "wb") as f:
            f.write(img.content)
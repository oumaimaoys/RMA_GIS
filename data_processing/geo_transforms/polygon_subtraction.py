import json
from shapely.geometry import shape, mapping
from pathlib import Path

# === Step 1: Load GeoJSON file ===
input_path = "/home/ouyassine/Documents/projects/RMA_SIG_app/data_ingestion/raw_data/dakhla.geojson"   # Replace with your actual file name
output_path = "a_minus_b.geojson"

with open(input_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# === Step 2: Extract the two polygons ===
features = geojson_data.get("features", [])
if len(features) < 2:
    raise ValueError("The GeoJSON file must contain at least two polygon features.")

polygon_a = shape(features[0]["geometry"])
polygon_b = shape(features[1]["geometry"])


print("Polygon A intersects B:", polygon_a.intersects(polygon_b))


# === Step 3: Compute A - B ===
difference = polygon_a.difference(polygon_b)

# === Step 4: Convert to GeoJSON format ===
result_feature = {
    "type": "Feature",
    "properties": {"operation": "A - B"},
    "geometry": mapping(difference)
}

result_geojson = {
    "type": "FeatureCollection",
    "features": [result_feature]
}

# === Step 5: Save result ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_geojson, f, indent=2)

print(f"Polygon difference saved to: {output_path}")

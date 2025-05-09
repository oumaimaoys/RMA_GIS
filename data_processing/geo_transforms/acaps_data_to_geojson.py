import json

def convert_to_geojson(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    features = []

    for entry in data:
        try:
            lat = float(entry["latitude"])
            lon = float(entry["longitude"])

            # Create a GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {k: v for k, v in entry.items() if k not in ["latitude", "longitude"]}
            }

            features.append(feature)

        except (ValueError, KeyError):
            print(f"Skipping entry due to invalid coordinates: {entry}")

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(geojson, outfile, ensure_ascii=False, indent=2)

    print(f"GeoJSON saved to {output_path}")

convert_to_geojson("/home/ouyassine/Documents/projects/RMA_SIG_app/data_ingestion/raw_data/acaps_data.json", "acaps_data.geojson")
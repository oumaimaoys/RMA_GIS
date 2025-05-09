import csv
import json

def csv_to_geojson(input_file, output_file):
    # Initialize a GeoJSON object
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Read the CSV file
    with open(input_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # Loop through each row in the CSV file
        for row in reader:
            # Create a GeoJSON feature for each row
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",  # Since we have latitude and longitude for each commune
                    "coordinates": [float(row["Longitude"]), float(row["Latitude"])]  # [longitude, latitude]
                },
                "properties": {
                    "commune": row["commune"],
                    "population": row["Population"],
                    "region": row["region"]
                }
            }
            # Append the feature to the features list
            geojson["features"].append(feature)

    # Write the GeoJSON data to a file
    with open(output_file, mode='w', encoding='utf-8') as outfile:
        json.dump(geojson, outfile, indent=4)

    print(f"GeoJSON file has been saved as {output_file}.")

# Example usage:
csv_to_geojson('/home/ouyassine/Documents/projects/RMA_SIG_app/data_ingestion/raw_data/cities_cleaned.csv', 'output_file.geojson')

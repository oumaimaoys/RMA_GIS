import json
import pandas as pd

# Load the GeoJSON file
with open('/home/ouyassine/Documents/projects/RMA_SIG_project/data_ingestion/raw_data/jsons/banks.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract required data
rows = []
for feature in data['features']:
    name = feature['properties'].get('name', 'Unknown')
    lon, lat = feature['geometry']['coordinates']
    rows.append({'bank': name, 'longitude': lon, 'latitude': lat})

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to Excel
df.to_excel('banks.xlsx', index=False)

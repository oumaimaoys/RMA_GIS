import pandas as pd
import json
import unicodedata

def normalize(name: str) -> str:
    """
    Remove accents, lowercase, strip whitespace.
    """
    nfkd = unicodedata.normalize('NFKD', name)
    only_ascii = nfkd.encode('ASCII', 'ignore').decode('utf-8')
    return only_ascii.strip().lower()

# 1. Load the CSV
csv_path = '/home/ouyassine/Documents/projects/RMA_SIG_project/data_ingestion/raw_data/csvs/cities_cleaned.csv'           # your CSV file
df = pd.read_csv(csv_path)

# Create a lookup from normalized commune name → population value
df['_norm'] = df['commune'].apply(normalize)
pop_lookup = df.set_index('_norm')['Population'].to_dict()

# 2. Load the GeoJSON
geojson_path = '/home/ouyassine/Documents/projects/RMA_SIG_project/RMA_SIG/frontend/static/geojson/communes.geojson'   # your GeoJSON file
with open(geojson_path, 'r', encoding='utf-8') as f:
    geo = json.load(f)

# 3. Update each feature’s Population
for feat in geo.get('features', []):
    props = feat.get('properties', {})

    # adjust this key if your GeoJSON uses a different property name
    raw_name = props.get('Nom_Commun') or props.get('nom_commu') or props.get('COMMUNE')  
    if raw_name:
        key = normalize(raw_name)
        if key in pop_lookup:
            props['Population'] = float(pop_lookup[key])
        else:
            # optional: warn if no match found
            print(f"⚠️ No CSV match for GeoJSON feature '{raw_name}'")

# 4. Write out updated GeoJSON
out_path = 'communes_updated.geojson'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(geo, f, ensure_ascii=False, indent=2)

print(f"Updated GeoJSON saved to {out_path}")

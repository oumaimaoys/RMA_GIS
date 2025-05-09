import json
import csv
import pandas as pd

# Paths
json_path = "/home/ouyassine/Documents/projects/RMA_SIG_app/data_ingestion/raw_data/acaps_data.json"
csv_path = "output.csv"
excel_path = "output.xlsx"

# === Step 1: Load JSON ===
with open(json_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

# === Step 2: Write to CSV ===
with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

# === Step 3: Convert CSV to Excel ===
df = pd.read_csv(csv_path)
df.to_excel(excel_path, index=False)

print(f"Successfully saved to: {excel_path}")

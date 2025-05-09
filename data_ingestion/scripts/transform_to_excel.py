import json
import pandas as pd

# Path to your JSON file
json_path = "/home/ouyassine/Documents/projects/RMA_SIG_app/data_ingestion/raw_data/banks.json"  # change if needed
excel_output_path = "banks.xlsx"

# Load the JSON data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel(excel_output_path, index=False)

print(f"✅ Excel file saved to: {excel_output_path}")

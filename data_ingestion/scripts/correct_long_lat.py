#!/usr/bin/env python3
"""
update_coords.py

Hard-coded paths version: set FIRST_PATH, SECOND_PATH, OUTPUT_PATH below.
"""

import pandas as pd

# ─── Edit these ────────────────────────────────────────────────────────────────
FIRST_PATH  = "/home/ouyassine/Documents/projects/RMA_SIG_project/agents.xlsx"
SECOND_PATH = "/home/ouyassine/Documents/projects/RMA_SIG_project/acaps_data.xlsx"
OUTPUT_PATH = "/home/ouyassine/Documents/projects/RMA_SIG_project/agents_updated.xlsx"
# ───────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load both files
    df_main   = pd.read_excel(FIRST_PATH)
    df_lookup = pd.read_excel(SECOND_PATH)

    # 2) Normalize column names
    df_main.columns   = df_main.columns.str.strip()
    df_lookup.columns = df_lookup.columns.str.strip()

    # 3) Rename lookup columns to match main
    df_lookup = df_lookup.rename(columns={
        "code_acaps": "Code ACAPS",
        "longitude":  "Longitude",
        "latitude":   "Latitude",
    })

    # 4) Preserve originals for fallback
    orig_lat = df_main["Latitude"].copy()
    orig_lon = df_main["Longitude"].copy()

    # 5) Build deduped lookup Series
    lut_lat = df_lookup.groupby("Code ACAPS")["Latitude"].first()
    lut_lon = df_lookup.groupby("Code ACAPS")["Longitude"].first()

    # 6) Map & replace, falling back where there’s no match
    df_main["Latitude"]  = df_main["Code ACAPS"].map(lut_lat).fillna(orig_lat)
    df_main["Longitude"] = df_main["Code ACAPS"].map(lut_lon).fillna(orig_lon)

    # 7) Write out updated sheet
    df_main.to_excel(OUTPUT_PATH, index=False)
    print(f"✅ Updated file written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

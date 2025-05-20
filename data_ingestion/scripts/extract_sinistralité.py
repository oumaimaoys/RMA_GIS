import pandas as pd

# ─── Edit this path to point to your Excel file ────────────────────────────────
FILE_PATH = '/mnt/data/SP 2022-2024 par PDV - réseau exclusif (003) (1).xlsx'
# ────────────────────────────────────────────────────────────────────────────────

# 1) Read the workbook, assuming the real header is on the second row (skip the first)
df = pd.read_excel(FILE_PATH, header=1)

# 2) Rename columns for clarity
#    We expect the columns: ['Réseau','Code Inter','Nom Inter','Usage',
#                             'Prime','S/P','Prime','S/P','Prime','S/P','Prime','S/P']
#    After header=1, pandas will auto-rename duplicates: 'Prime', 'S/P.1', 'Prime.1', etc.
df = df.rename(columns={
    'Prime':         'Prime_2022',
    'S/P':           'SP_2022',
    'Prime.1':       'Prime_2023',
    'S/P.1':         'SP_2023',
    'Prime.2':       'Prime_2024',
    'S/P.2':         'SP_2024',
    'Prime.3':       'Prime_Total',
    'S/P.3':         'SP_Total',
})

# 3) Filter rows where Usage == "Total" (the summary line per Nom Inter)
df_totals = df[df['Usage'].str.strip().eq('Total')]

# 4) Select only Nom Inter and the 2024 totals
result = df_totals[['Nom Inter', 'Prime_2024', 'SP_2024']].copy()

# 5) (Optional) Clean up percentage column: if SP_2024 is a string with '%' remove it and convert to float
if result['SP_2024'].dtype == object:
    result['SP_2024'] = result['SP_2024'].str.replace('%', '', regex=False).astype(float) / 100

# 6) Write out to a new Excel
OUTPUT_PATH = 'nom_inter_totals_2024.xlsx'
result.to_excel(OUTPUT_PATH, index=False)

print(f"Extracted totals for 2024 saved to {OUTPUT_PATH}:")
print(result)

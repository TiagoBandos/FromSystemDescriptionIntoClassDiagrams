import pandas as pd
from pathlib import Path

folder = Path(__file__).parent
excel_files = list(folder.glob("*.xlsx"))

# Dictionary to hold combined DataFrames for each sheet
combined_sheets = {}

for file in excel_files:
    sheets = pd.read_excel(file, sheet_name=None)  # read all sheets
    
    for sheet_name, df in sheets.items():
        if sheet_name not in combined_sheets:
            combined_sheets[sheet_name] = []
        combined_sheets[sheet_name].append(df)

# Write all combined sheets into a single Excel file
with pd.ExcelWriter("all_results_combined_by_sheet.xlsx", engine='openpyxl') as writer:
    for sheet_name, list_of_dfs in combined_sheets.items():
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

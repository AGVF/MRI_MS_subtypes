import pandas as pd
import numpy as np
import os

# Directories
input_csv_path = "../../data/original_data/demoall.csv"
output_folder = "../../data/generated_data/filtered_data"
output_filename = "filtered_data.csv"
na_threshold = 70
mode = '3TP'  # Options: '3TP' or 'TP1'

# Load the data
df = pd.read_csv(input_csv_path)

# Clean and Merge MSType and MStype columns
def clean_string(val):
    return val.strip() if isinstance(val, str) else val

if 'MSType' in df.columns and 'MStype' in df.columns:
    df['MStype_clean'] = df['MStype'].apply(clean_string)
    df['MSType'] = df['MSType'].combine_first(df['MStype_clean'])
    df.drop(columns=['MStype', 'MStype_clean'], inplace=True)
    df['MSType'].replace("[nan]", np.nan, inplace=True)

# Filtering by timepoints and cohorts
required_columns = {'TP', 'ID', 'Cohort'}
assert required_columns.issubset(df.columns), "Missing required columns: TP, ID, or Cohort"

if 'Group' in df.columns:
    df = df[df['Group'] != 'HC']

if mode == '3TP':
    tp_counts = df.groupby(['Cohort', 'ID'])['TP'].nunique().reset_index()
    valid_ids = tp_counts[tp_counts['TP'] == 3]
    df = df.merge(valid_ids[['Cohort', 'ID']], on=['Cohort', 'ID'], how='inner')
elif mode == 'TP1':
    df = df[df['TP'] == 1]
else:
    raise ValueError("Invalid mode. Use '3TP' or 'TP1'.")

# Selecting columns based on prefixes and clinical data
clinical_columns = [
    'EDSS', 'Cohort', 'TP', 'Age', 'Sex', 'MSType', 'Time_since_diagnosis',
    'SRT-L', 'SRT-C', 'SRT-D', 'Spatial RT', 'Delayed Spatial RT',
    'SDMT', 'PASAT', 'WLG', 'SSST', 'SixMWT', 'VOL_Lesion', 'ID'
]

selected_columns = [col for col in df.columns if (
    col.startswith(('VOL_', 'FA_', 'MD_')) or col in clinical_columns
)]
df = df[selected_columns]

# Save NaN counts before filtering
initial_nas = df.isna().sum()
os.makedirs(output_folder, exist_ok=True)

initial_nas_path = os.path.join(output_folder, f"nans_per_column_{mode}.txt")
with open(initial_nas_path, "w") as f:
    for col, na in initial_nas.items():
        f.write(f"{col}: {na} NaNs\n")
print(f"NaN summary saved: {initial_nas_path}")

# Drop columns with ≥ NA threshold
columns_to_keep = initial_nas[initial_nas < na_threshold].index
df = df[columns_to_keep]

# Save NaNs after column filter
post_filter_nas = df.isna().sum()
post_filter_path = os.path.join(output_folder, f"nans_after_filter_{mode}.txt")
with open(post_filter_path, "w") as f:
    for col, na in post_filter_nas.items():
        f.write(f"{col}: {na} NaNs\n")
print(f"Post-filter NaN summary saved: {post_filter_path}")

# Drop rows with remaining NaNs
df.drop(columns=['EDSS', 'SDMT', 'SSST', 'SixMWT'], inplace=True, errors='ignore')

# Manual NaN check on specific columns
specific_na_columns = [
    'MD_globus_pallidus', 'MD_putamen', 'MD_caudate', 'MD_hippocampus', 'MD_thalamus',
    'MD_Anterior_thalamic_radiation', 'MD_Corticospinal_tract', 'MD_Cingulate_gyrus',
    'MD_Forceps_major', 'MD_Forceps_minor', 'MD_fronto_occipital',
    'MD_inferior_longitudinal_fasciculus', 'MD_superior_longitudinal_fasciculus',
    'MD_uncinate_fasciculus', 'FA_globus_pallidus', 'FA_putamen', 'FA_caudate',
    'FA_hippocampus', 'FA_thalamus', 'FA_Anterior_thalamic_radiation',
    'FA_Corticospinal_tract', 'FA_Cingulate_gyrus', 'FA_Forceps_major',
    'FA_Forceps_minor', 'FA_fronto_occipital', 'FA_inferior_longitudinal_fasciculus',
    'FA_superior_longitudinal_fasciculus', 'FA_uncinate_fasciculus'
]

missing_rows = df[df[specific_na_columns].isna().any(axis=1)]
if not missing_rows.empty:
    print("Rows with specific missing data:")
    print(missing_rows[['ID', 'Cohort'] + specific_na_columns])

df = df.dropna()

# Save
final_csv_path = os.path.join(output_folder, f"{mode.lower()}_{output_filename}")
df.to_csv(final_csv_path, index=False)
print(f"Final CSV saved without NaNs: {final_csv_path}")
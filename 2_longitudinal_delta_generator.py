import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel

# Paths and information
input_path = "../data/updated_data_all_tp/df_transformado.csv"
output_csv_path = "../data/updated_data_all_tp/delta_sig_changes/delta_values_no_change_clinical.csv"
output_folder = os.path.dirname(output_csv_path)
alpha = 0.05  # significance threshold

# Load the data
df = pd.read_csv(input_path)

# Filter subjects with TP1 and TP3
subject_tp_count = df.groupby(["ID", "Cohort"])["TP"].nunique().reset_index()
valid_subjects = subject_tp_count[subject_tp_count["TP"] >= 3]
df_valid = df.merge(valid_subjects[["ID", "Cohort"]], on=["ID", "Cohort"])
df_valid = df_valid[df_valid["TP"].isin([1, 3])]

# Ensure exactly TP1 and TP3 per subject
tp_counts = df_valid.groupby(["ID", "Cohort"])["TP"].nunique()
subjects_ok = tp_counts[tp_counts == 2].index
df_valid = df_valid.set_index(["ID", "Cohort"]).loc[subjects_ok].reset_index()
df_valid = df_valid.sort_values(by=["ID", "Cohort", "TP"])

# Define variables
clinical_vars = [
    'EDSS', 'SRT-L', 'SRT-C', 'SRT-D', 'Spatial RT', 'Delayed Spatial RT',
    'SDMT', 'PASAT', 'WLG', 'SSST', 'SixMWT', 'VOL_Lesion'
]

static_vars = ['Cohort', 'TP', 'Age', 'Sex', 'MSType', 'Time_since_diagnosis', 'ID', 'ID_STORMDB']

bio_vars = [col for col in df.columns if col.startswith(("VOL_", "FA_", "MD_"))]

# Split by TP (TP1 AND TP3)
df_tp1 = df_valid[df_valid["TP"] == 1].set_index(["ID", "Cohort"])
df_tp3 = df_valid[df_valid["TP"] == 3].set_index(["ID", "Cohort"])

# Significance testing (is there any difference between TP1 and TP3?)
clinical_results = []
clinical_tp3 = pd.DataFrame()
clinical_changed = []
clinical_unchanged = []

for var in clinical_vars:
    if var not in df_tp1.columns or var not in df_tp3.columns:
        continue

    v1 = df_tp1[var]
    v3 = df_tp3[var]
    stat, pval = ttest_rel(v1, v3, nan_policy='omit')
    significant = pval < alpha

    clinical_results.append({"variable": var, "p_value": pval, "significant_change": significant})
    clinical_tp3[var] = v3

    if significant:
        clinical_changed.append(var)
    else:
        clinical_unchanged.append(var)

# Significance testing for biological variables
bio_results = []
bio_changed = []
bio_unchanged = []

for var in bio_vars:
    if var not in df_tp1.columns or var not in df_tp3.columns:
        continue

    v1 = df_tp1[var]
    v3 = df_tp3[var]
    stat, pval = ttest_rel(v1, v3, nan_policy='omit')
    significant = pval < alpha

    bio_results.append({"variable": var, "p_value": pval, "significant_change": significant})

    if significant:
        bio_changed.append(var)
    else:
        bio_unchanged.append(var)

# Calculate delta values for biological variables
df_delta_bio = df_valid.groupby(["ID", "Cohort"])[bio_vars].diff().dropna()

# Reconstruct the final DataFrame
df_base = df_tp3[["Age"] + [v for v in static_vars if v in df_tp3.columns and v != "TP"]].reset_index()
df_delta_bio = df_delta_bio.reset_index(drop=True)
clinical_tp3 = clinical_tp3.reset_index(drop=True)

df_final = pd.concat([df_base, df_delta_bio, clinical_tp3], axis=1)

# Generate output folder and save the final DataFrame
os.makedirs(output_folder, exist_ok=True)
df_final.to_csv(output_csv_path, index=False)

# Save
df_clinical_tests = pd.DataFrame(clinical_results)
df_bio_tests = pd.DataFrame(bio_results)

df_clinical_tests.to_csv(os.path.join(output_folder, "clinical_change_tests.csv"), index=False)
df_bio_tests.to_csv(os.path.join(output_folder, "biological_change_tests.csv"), index=False)

# Clinical variables summary
summary_clinical_txt = os.path.join(output_folder, "summary_clinical_changes.txt")
with open(summary_clinical_txt, "w") as f:
    f.write("📊 Clinical Variable Change Summary\n")
    f.write(f"Total subjects included: {df_final.shape[0]}\n")
    f.write(f"Total clinical variables tested: {len(clinical_results)}\n\n")
    f.write("Variables with significant change (TP3 used in output):\n")
    for var in clinical_changed:
        f.write(f"  - {var}\n")
    f.write("\nVariables without significant change (TP3 used in output):\n")
    for var in clinical_unchanged:
        f.write(f"  - {var}\n")

# Biological variables summary txt
summary_bio_txt = os.path.join(output_folder, "summary_biological_changes.txt")
with open(summary_bio_txt, "w") as f:
    f.write("🧬 Biological Variable Change Summary\n")
    f.write(f"Total biological variables tested: {len(bio_results)}\n\n")
    f.write("Variables with significant change (delta computed):\n")
    for var in bio_changed:
        f.write(f"  - {var}\n")
    f.write("\nVariables without significant change (delta still computed):\n")
    for var in bio_unchanged:
        f.write(f"  - {var}\n")

# Debbuging output
print(f"✅ Final dataset saved to: {output_csv_path}")
print(f"📄 Clinical t-tests saved to: clinical_change_tests.csv")
print(f"📄 Biological t-tests saved to: biological_change_tests.csv")
print(f"📝 Clinical summary saved to: summary_clinical_changes.txt")
print(f"📝 Biological summary saved to: summary_biological_changes.txt")

# Save by modality

# Identify FA_, MD_, VOL_ variables present in the final dataset
fa_md_vars = [col for col in df_final.columns if col.startswith("FA_") or col.startswith("MD_")]
vol_vars = [col for col in df_final.columns if col.startswith("VOL_")]

# Include clinical variables (from earlier defined lists)
clinical_vars_present = [var for var in clinical_vars if var in df_final.columns]
static_vars_present = [var for var in static_vars if var in df_final.columns]
clinical_columns = clinical_vars_present + static_vars_present

# Prepare filtered DataFrames
df_diffusion = df_final[clinical_columns + fa_md_vars]
df_volume = df_final[clinical_columns + vol_vars]

# Define filenames based on base name
base_name = os.path.splitext(os.path.basename(output_csv_path))[0]
diffusion_filename = base_name + "_diff.csv"
volume_filename = base_name + "_vol.csv"

# Full paths
diffusion_path = os.path.join(output_folder, diffusion_filename)
volume_path = os.path.join(output_folder, volume_filename)

# Save files
df_diffusion.to_csv(diffusion_path, index=False)
df_volume.to_csv(volume_path, index=False)

print(f"📁 Diffusion + clinical dataset saved to: {diffusion_path}")
print(f"📁 Volume + clinical dataset saved to: {volume_path}")

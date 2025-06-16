import pandas as pd
import os

def filter_by_cohort(input_path, output_dir, cohorts_to_keep=["EXBRAIN", "POTOMS"]):
    """
    Filters rows where 'Cohort' is in cohorts_to_keep, saves the full filtered dataset,
    and also saves versions with only diffusion (FA_, MD_) and volume (VOL_) variables plus clinical variables.
    """
    # Load data
    df = pd.read_csv(input_path)

    # Filter by cohort
    df_filtered = df[df["Cohort"].isin(cohorts_to_keep)]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define base filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    filtered_name = f"{base_name}_filtered_cohorts"
    output_full = os.path.join(output_dir, f"{filtered_name}.csv")

    # Save full filtered dataset
    df_filtered.to_csv(output_full, index=False)
    print(f"✅ Full filtered dataset saved to: {output_full}")

    # Identify variable groups
    fa_md_vars = [col for col in df_filtered.columns if col.startswith("FA_") or col.startswith("MD_")]
    vol_vars = [col for col in df_filtered.columns if col.startswith("VOL_")]

    # Clinical columns to retain (auto-detected)
    known_clinical = [
        'ID', 'Cohort', 'Age', 'Sex', 'MSType', 'EDSS',
        'Time_since_diagnosis', 'SDMT', 'SixMWT', 'SSST',
        'SRT-L', 'SRT-C', 'SRT-D', 'Spatial RT', 'Delayed Spatial RT',
        'PASAT', 'WLG', 'VOL_Lesion', 'ID_STORMDB'
    ]
    clinical_vars = [col for col in known_clinical if col in df_filtered.columns]

    # Create and save diffusion-only dataset
    df_diff = df_filtered[clinical_vars + fa_md_vars]
    output_diff = os.path.join(output_dir, f"{filtered_name}_diff.csv")
    df_diff.to_csv(output_diff, index=False)
    print(f"📁 Diffusion + clinical dataset saved to: {output_diff}")

    # Create and save volume-only dataset
    df_vol = df_filtered[clinical_vars + vol_vars]
    output_vol = os.path.join(output_dir, f"{filtered_name}_vol.csv")
    df_vol.to_csv(output_vol, index=False)
    print(f"📁 Volume + clinical dataset saved to: {output_vol}")

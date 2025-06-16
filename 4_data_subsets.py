# Subgroup Generator Script
import pandas as pd
import os
import numpy as np

# Load data
data = pd.read_csv('../data/updated_data/demoall.csv')
data = data[data['TP'] == 1]  # Keep only TP = 1
print(f"Data shape after filtering: {data.shape}")

# Define type of variables
clinical_vars = [
    'EDSS', 'Age', 'Sex', 'MSType', 'Time_since_diagnosis',
    'SRT-L', 'SRT-C', 'SRT-D', 'Spatial RT', 'Delayed Spatial RT',
    'SDMT', 'PASAT', 'WLG', 'SSST', 'SixMWT', 'VOL_Lesion', 'ID'
]

prefixes = ['VOL_', 'FA_', 'MD_']

columns_to_remove = ['ID_STORMDB', 'ID', 'initialer', 'group_exercise1', 'Dropout', 'index', 'scale', 'VOL_scale']

#Define data subsets generator functin
def generate_subgroup(data, clinical_vars, prefixes, remove_cols=None, group_type='all'):
    """
    Generates a filtered version of the dataset by selecting specific types of variables.

    group_type options:
    - 'all': clinical + FA + MD + VOL
    - 'clinical': only clinical
    - 'vol': only volumetric
    - 'diff': only diffusion (FA & MD)
    - 'clinical_vol': clinical + volumetric
    - 'clinical_diff': clinical + diffusion
    - 'clinical_diff_vol': clinical + diffusion + volumetric
    - 'no_clinical': only metrics (removes clinical)
    """
    data = data.copy()

    if remove_cols:
        valid_cols = [col for col in remove_cols if col in data.columns]
        data = data.drop(columns=valid_cols)

    available_clinical = [col for col in clinical_vars if col in data.columns]

    if group_type == 'all':
        cols = available_clinical + [col for col in data.columns if any(col.startswith(p) for p in prefixes)]
    elif group_type == 'clinical':
        cols = available_clinical
    elif group_type == 'vol':
        cols = [col for col in data.columns if col.startswith('VOL_')]
    elif group_type == 'diff':
        cols = [col for col in data.columns if col.startswith('FA_') or col.startswith('MD_')]
    elif group_type == 'clinical_diff':
        cols = available_clinical + [col for col in data.columns if col.startswith('FA_') or col.startswith('MD_')]
    elif group_type == 'clinical_vol':
        cols = available_clinical + [col for col in data.columns if col.startswith('VOL_')]
    elif group_type == 'clinical_diff_vol':
        cols = available_clinical + [col for col in data.columns if any(col.startswith(p) for p in prefixes)]
    elif group_type == 'no_clinical':
        cols = [col for col in data.columns if col not in available_clinical]
    else:
        raise ValueError(f"Unknown group_type: {group_type}")

    # Ensure 'Cohort' and 'TP' are retained
    for col in ['Cohort', 'TP']:
        if col in data.columns and col not in cols:
            cols.append(col)

    return data[cols]

# Execute
if __name__ == "__main__":
    clinical_df = generate_subgroup(data, clinical_vars, prefixes, columns_to_remove, group_type='clinical')
    clin_vol_df = generate_subgroup(data, clinical_vars, prefixes, columns_to_remove, group_type='clinical_vol')
    clin_diff_df = generate_subgroup(data, clinical_vars, prefixes, columns_to_remove, group_type='clinical_diff')

    print("✅ Subgroups generated:")
    print(f"Clinical only: {clinical_df.shape}")
    print(f"Clinical + Volumes: {clin_vol_df.shape}")
    print(f"Clinical + Diffusion: {clin_diff_df.shape}")

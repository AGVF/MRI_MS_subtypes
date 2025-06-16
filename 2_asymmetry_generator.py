# hem data calculation
import pandas as pd

# Define volume pairs (Right, Left)
volume_pairs = [
    ("VOL_parietal_right_gm", "VOL_parietal_left_gm"),
    ("VOL_lateral_ventricle_right", "VOL_lateral_ventricle_left"),
    ("VOL_occipital_right_gm", "VOL_occipital_left_gm"),
    ("VOL_globus_pallidus_right", "VOL_globus_pallidus_left"),
    ("VOL_putamen_right", "VOL_putamen_left"),
    ("VOL_frontal_right_wm", "VOL_frontal_left_wm"),
    ("VOL_subthalamic_nucleus_right", "VOL_subthalamic_nucleus_left"),
    ("VOL_fornix_right", "VOL_fornix_left"),
    ("VOL_caudate_right", "VOL_caudate_left"),
    ("VOL_occipital_right_wm", "VOL_occipital_left_wm"),
    ("VOL_parietal_right_wm", "VOL_parietal_left_wm"),
    ("VOL_temporal_right_wm", "VOL_temporal_left_wm"),
    ("VOL_cerebellum_right", "VOL_cerebellum_left"),
    ("VOL_thalamus_right", "VOL_thalamus_left"),
    ("VOL_frontal_right_gm", "VOL_frontal_left_gm"),
    ("VOL_temporal_right_gm", "VOL_temporal_left_gm"),
]

# Define clinical variables to retain
clinical_vars = [
    'ID', 'Cohort', 'Age', 'Sex', 'MSType', 'EDSS',
    'Time_since_diagnosis', 'SDMT', 'SixMWT', 'SSST'
]

def compute_asymmetry_indices(df, volume_pairs, clinical_vars):
    """
    Computes asymmetry index (AI) for each volume pair and returns a DataFrame
    containing clinical variables and AI values.
    """
    df_ai = df[clinical_vars].copy()

    for right_var, left_var in volume_pairs:
        if right_var in df.columns and left_var in df.columns:
            L = df[left_var]
            R = df[right_var]
            ai = (L - R) / (L + R)
            region_name = left_var.replace("VOL_", "").replace("_left", "").replace("_right", "")
            ai_column = f"AI_{region_name}"
            df_ai[ai_column] = ai
        else:
            print(f"Warning: Missing pair -> {left_var}, {right_var}")

    return df_ai



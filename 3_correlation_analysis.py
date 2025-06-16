# Removing redundancies in the dataset
import pandas as pd

df = pd.read_csv("../../data/generated_data/transformed_data/df_transformed.csv")

# 1. Remove bilateral structures that also exist in patch-based segmentation
cols_patch_based = [
    'VOL_putamen_left', 'VOL_putamen_right',
    'VOL_thalamus_left', 'VOL_thalamus_right',
    'VOL_caudate_left', 'VOL_caudate_right',
    'VOL_globus_pallidus_left', 'VOL_globus_pallidus_right',
]
df.drop(columns=[col for col in cols_patch_based if col in df.columns], inplace=True)

# 2. Remove individual components already represented in total GM/WM/CSF volumes
cols_contained_in_totals = [
    'VOL_frontal_left_gm', 'VOL_frontal_right_gm',
    'VOL_temporal_left_gm', 'VOL_temporal_right_gm',
    'VOL_parietal_left_gm', 'VOL_parietal_right_gm',
    'VOL_frontal_left_wm', 'VOL_frontal_right_wm',
    'VOL_temporal_left_wm', 'VOL_temporal_right_wm',
    'VOL_parietal_left_wm', 'VOL_parietal_right_wm',
    'VOL_extracerebral_CSF'
]
df.drop(columns=[col for col in cols_contained_in_totals if col in df.columns], inplace=True)

# 3. Sum bilateral structures that don't have an existing total variable (only a few examples)
def sum_bilateral(df, left_col, right_col, new_col):
    if left_col in df.columns and right_col in df.columns:
        df[new_col] = df[left_col] + df[right_col]
        df.drop(columns=[left_col, right_col], inplace=True)

sum_bilateral(df, 'VOL_lateral_ventricle_left', 'VOL_lateral_ventricle_right', 'VOL_lateral_ventricle_total')
sum_bilateral(df, 'VOL_cerebellum_left', 'VOL_cerebellum_right', 'VOL_cerebellum_total')
sum_bilateral(df, 'VOL_occipital_left_gm', 'VOL_occipital_right_gm', 'VOL_occipital_total_gm')
sum_bilateral(df, 'VOL_occipital_left_wm', 'VOL_occipital_right_wm', 'VOL_occipital_total_wm')
sum_bilateral(df, 'VOL_fornix_left', 'VOL_fornix_right', 'VOL_fornix_total')
sum_bilateral(df, 'VOL_subthalamic_nucleus_left', 'VOL_subthalamic_nucleus_right', 'VOL_subthalamic_nucleus_total')

# 4. Final confirmation of cleaned columns
print("Final columns:", df.columns.tolist())

# Optionally save the result
# df = df.drop(columns=['VOL_Lesion.1'])
print(df.shape)
df.to_csv("../../data/generated_data/transformed_preprocessed_data/baseline_data.csv", index=False)

# feature transformation:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, skew, kurtosis
from sklearn.preprocessing import StandardScaler

#Define functions
def bimodality_coefficient(x):
    """Calculates the bimodality coefficient of a variable"""
    x = x.dropna()
    s = skew(x)
    k = kurtosis(x, fisher=False)
    n = len(x)
    bc = (s**2 + 1) / (k + (3 * ((n - 1)**2)) / ((n - 2) * (n - 3))) if n > 3 else 0
    return bc

def transform_skewed_variables(df, output_dir, skew_threshold=1.0, excluded_vars=None):
    df = df.copy()
    if excluded_vars is None:
        excluded_vars = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_features = {}
    transformed_vars = []
    still_skewed = {}
    bimodal_vars = {}

    os.makedirs(output_dir, exist_ok=True)
    hist_dir = os.path.join(output_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)

    for col in numeric_cols:
        if col in excluded_vars:
            print(f"⏭️  {col}: excluded from transformation.")
            continue

        data_original = df[col].dropna()
        skew_before = skew(data_original)
        bc_before = bimodality_coefficient(data_original)

        if abs(skew_before) > skew_threshold:
            plt.figure(figsize=(6, 4))
            sns.histplot(data_original, kde=True, bins=30)
            plt.title(f"{col} - Original (Skew: {skew_before:.2f})")
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"{col}_before.png"))
            plt.close()

            if (df[col] <= 0).any():
                df[col] = np.log1p(df[col] - df[col].min() + 1)
                method = "log1p"
            else:
                try:
                    transformed, _ = boxcox(df[col])
                    df[col] = transformed
                    method = "boxcox"
                except ValueError:
                    df[col] = np.log1p(df[col])
                    method = "log1p (fallback)"

            skew_after = skew(df[col].dropna())
            bc_after = bimodality_coefficient(df[col])

            # Save results
            skewed_features[col] = {
                "skew_before": skew_before,
                "skew_after": skew_after,
                "method": method,
                "bimodality_before": bc_before,
                "bimodality_after": bc_after
            }

            transformed_vars.append(col)

            if abs(skew_after) > skew_threshold:
                still_skewed[col] = skew_after
            if bc_after > 0.555:
                bimodal_vars[col] = bc_after

            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, bins=30, color="green")
            plt.title(f"{col} - Transformed ({method}, Skew: {skew_after:.2f})")
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"{col}_after.png"))
            plt.close()

    # ========== NORMALIZATION (z-score) ==========
    vars_to_normalize = [col for col in numeric_cols if col not in excluded_vars]
    scaler = StandardScaler()
    df[vars_to_normalize] = scaler.fit_transform(df[vars_to_normalize])

    print(f"\n📏 Normalization applied to: {vars_to_normalize}")

    # Save files
    df.to_csv(os.path.join(output_dir, "df_transformed.csv"), index=False)
    pd.DataFrame.from_dict(skewed_features, orient='index').to_csv(os.path.join(output_dir, "skew_summary.csv"))
    pd.Series(still_skewed, name="Skew_after").to_csv(os.path.join(output_dir, "still_skewed.csv"))
    pd.Series(bimodal_vars, name="Bimodality_after").to_csv(os.path.join(output_dir, "bimodal_vars.csv"))

    print("\n✅ Transformed variables:")
    for var in transformed_vars:
        info = skewed_features[var]
        print(f"- {var}: {info['method']} | skew before: {info['skew_before']:.2f}, after: {info['skew_after']:.2f}")

    print(f"\nStill skewed variables after transformation: {list(still_skewed.keys())}")
    print(f"Bimodal variables: {list(bimodal_vars.keys())}")
    print(f"\n Files saved in: {output_dir}")
    return df, transformed_vars, skewed_features, still_skewed, bimodal_vars

# Variables that should NOT be transformed
vars_to_exclude = [
    'EDSS', 'SRT-L', 'SRT-C', 'SRT-D', 'Spatial RT', 'Delayed Spatial RT',
    'SDMT', 'PASAT', 'WLG', 'SSST', 'SixMWT', 'VOL_Lesion', 'Cohort', 'TP', 'Age', 'Sex', 'MSType', 'Time_since_diagnosis', 'ID',
]

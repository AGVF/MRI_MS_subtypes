import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def simplified_pca_clinical_analysis(
    df,
    output_dir,
    clinical_vars=None,
    dataset_name="dataset",
    n_components=5,
    method="spearman"
):
    """
    Perform PCA on numeric features and correlate components with clinical variables.
    Saves explained variance, top loadings, correlation CSVs, and heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    if clinical_vars is None:
        clinical_vars = []

    clinical = df[clinical_vars].copy() if clinical_vars else pd.DataFrame(index=df.index)

    # Encode categorical clinical variables
    for col in clinical.columns:
        if clinical[col].dtype == 'object':
            clinical[col] = clinical[col].astype('category').cat.codes

    # Keep numeric features only (excluding ID/Cohort)
    features = df.select_dtypes(include='number').drop(columns=['ID', 'Cohort'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=df.index)

    # Run PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    pcs = pca.fit_transform(X_scaled)
    pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=X_scaled.index)

    # Save explained variance
    pd.DataFrame({
        'PC': pcs_df.columns,
        'ExplainedVariance': pca.explained_variance_ratio_
    }).to_csv(os.path.join(output_dir, f'{dataset_name}_explained_variance.csv'), index=False)

    # Save top loadings per PC
    loadings = pd.DataFrame(pca.components_.T, index=X_scaled.columns, columns=pcs_df.columns)
    top_loads = {}
    for pc in loadings.columns:
        top_vars = loadings[pc].abs().sort_values(ascending=False).head(5)
        top_loads[pc] = top_vars.index.tolist()

    with open(os.path.join(output_dir, f'{dataset_name}_top_loadings.txt'), 'w') as f:
        for pc, vars in top_loads.items():
            f.write(f'{pc}: {", ".join(vars)}\n')

    # Correlate PCs with clinical vars
    if not clinical.empty:
        corr_data = []
        for pc in pcs_df.columns:
            for var in clinical.columns:
                rho, pval = spearmanr(pcs_df[pc], clinical[var], nan_policy='omit')
                corr_data.append({'Component': pc, 'ClinicalVar': var, 'Spearman_rho': rho, 'p_value': pval})

        corr_df = pd.DataFrame(corr_data)
        raw_pvals = corr_df['p_value'].values
        rejected, pvals_corrected, _, _ = multipletests(raw_pvals, method='fdr_bh')
        corr_df['p_value_FDR'] = pvals_corrected
        corr_df['Significant_FDR'] = rejected
        corr_df.to_csv(os.path.join(output_dir, f'{dataset_name}_correlation_pcs_vs_clinical.csv'), index=False)

        # Heatmap
        pivot = corr_df.pivot(index='Component', columns='ClinicalVar', values='Spearman_rho')
        plt.figure(figsize=(min(1.2 * len(pivot.columns), 15), min(0.4 * len(pivot), 20)))
        sns.heatmap(pivot, annot=True, cmap='vlag', center=0, fmt=".2f")
        plt.title(f'{dataset_name} — PCs vs Clinical Measures')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_heatmap_pc_vs_clinical.png'))
        plt.close()

    print(f"\n✅ Analysis complete: {dataset_name}\n- Output folder: {output_dir}\n")

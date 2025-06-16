# clustermap_batch_effect.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

#Preprocessing function
def preprocess_data(df, cohort_col='Cohort', id_col='ID', tp_col='TP'):
    df = df.copy()
    df['ID_cohort_TP'] = df[id_col].astype(str) + "_" + df[cohort_col] + "_" + df[tp_col].astype(str)

    annotations = df[[cohort_col, id_col, tp_col, 'ID_cohort_TP']]

    # Select only relevant numerical columns
    metric_prefixes = ['MD_', 'FA_', 'VOL_']
    feature_cols = [col for col in df.columns if any(col.startswith(p) for p in metric_prefixes)]
    features = df[feature_cols].copy()
    features.index = df['ID_cohort_TP']

    return annotations, features

# Define clustermap function
def plot_clustermap_with_cohorts(df, save_path=None):
    annotations, features = preprocess_data(df)

    # Hierarchical clustering
    row_linkage = linkage(pdist(features, metric='euclidean'), method='average')
    col_linkage = linkage(pdist(features.T, metric='euclidean'), method='average')

    # Colors by cohort (for visualization only)
    palette = sns.color_palette("Set2", annotations['Cohort'].nunique())
    cohort_colors = dict(zip(annotations['Cohort'].unique(), palette))
    row_colors = annotations.set_index('ID_cohort_TP')['Cohort'].map(cohort_colors)

    # Clustermap
    g = sns.clustermap(features,
                       row_linkage=row_linkage,
                       col_linkage=col_linkage,
                       row_colors=row_colors,
                       cmap="Blues",
                       figsize=(14, 10),
                       cbar_pos=(0.02, 0.8, 0.03, 0.18))

    # ARI (Adjusted Rand Index)
    reordered_ids = features.index[g.dendrogram_row.reordered_ind]
    true_labels = annotations.set_index('ID_cohort_TP').loc[reordered_ids]['Cohort']
    cluster_labels = fcluster(row_linkage, t=true_labels.nunique(), criterion='maxclust')

    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"🔍 Adjusted Rand Index (ARI): {ari:.3f} — closer to 1 indicates better alignment with true cohorts.")

    # Save figure
    if save_path:
        g.savefig(save_path, dpi=300)
        print(f"📁 Heatmap saved to {save_path}")

    plt.show()


# Execute
if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv("../../data/generated_data/transformed_data/df_transformed.csv")
    data['TP'] = 1
    data = data.drop(columns=[
        'EDSS', 'Age', 'Sex', 'MSType', 'SDMT', 'SSST', 'SixMWT', 
        'Time_since_diagnosis', 'EDSS', 'VOL_Lesion.1'
    ])

    # Run clustermap
    plot_clustermap_with_cohorts(data, save_path="../../results/exploratory_analysis/clustermap.png")

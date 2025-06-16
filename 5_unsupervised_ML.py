# PCA analysis and tSNE 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap 



def generate_combined_pca_plot(data_list, output_dir, label_column=None, prefix="pca_combined"):
    """
    Applies PCA to a list of datasets, generates a combined figure with vertical subplots,
    and also saves each individual PCA plot.

    Parameters:
        data_list (list): List of pandas DataFrames with numeric data.
        output_dir (str): Folder where the plots will be saved.
        label_column (str): Column with class labels to color the points (optional).
        prefix (str): Prefix for the output file names.
    """

    num_datasets = len(data_list)
    if num_datasets < 1:
        raise ValueError("You must provide at least one dataset.")

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=num_datasets, ncols=1, figsize=(10, 5 * num_datasets))
    if num_datasets == 1:
        axes = [axes]  # to ensure it's iterable

    fontsize_labels = 14
    fontsize_ticks = 12
    fontsize_title = 16
    fontsize_legend = 14

    # Color palette (if labels are available)
    all_labels = []
    color_map = {}
    if label_column and label_column in data_list[0].columns:
        all_labels = sorted(data_list[0][label_column].dropna().unique())
        palette = sns.color_palette("tab10", len(all_labels))
        color_map = {label: palette[i] for i, label in enumerate(all_labels)}

    for i, df in enumerate(data_list):
        ax = axes[i]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError(f"Dataset {i+1} does not have enough numeric columns.")

        data_numeric = df[numeric_cols]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_numeric)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)
        explained_var = pca.explained_variance_ratio_ * 100

        # Colors by label (if provided)
        if label_column and label_column in df.columns:
            labels = df[label_column].values
            point_colors = [color_map[label] for label in labels]
        else:
            labels = None
            point_colors = "royalblue"

        # --- Combined plot ---
        ax.scatter(pca_result[:, 0], pca_result[:, 1], c=point_colors, alpha=0.6)
        ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}%)", fontsize=fontsize_labels)
        ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}%)", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.set_title(f"PCA Plot {i+1}", fontsize=fontsize_title)

        # --- Individual plot ---
        fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
        ax_ind.scatter(pca_result[:, 0], pca_result[:, 1], c=point_colors, alpha=0.6)
        ax_ind.set_xlabel(f"PC1 ({explained_var[0]:.2f}%)", fontsize=fontsize_labels)
        ax_ind.set_ylabel(f"PC2 ({explained_var[1]:.2f}%)", fontsize=fontsize_labels)
        ax_ind.set_title(f"PCA Plot {i+1}", fontsize=fontsize_title)
        if label_column and all_labels:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_map[label], markersize=8)
                for label in all_labels
            ]
            ax_ind.legend(handles, all_labels, title=label_column, fontsize=fontsize_legend)

        output_ind_path = os.path.join(output_dir, f"{prefix}_plot{i+1}.png")
        fig_ind.savefig(output_ind_path, dpi=300)
        plt.close(fig_ind)

    # Add panel labels (a), (b), ...
    for i in range(num_datasets):
        fig.text(0.02, 1 - (i + 0.5) / num_datasets, f"({chr(97 + i)})", fontsize=fontsize_title, fontweight='bold')

    # Side legend (only once)
    if label_column and all_labels:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map[label], markersize=8)
            for label in all_labels
        ]
        fig.legend(
            handles,
            all_labels,
            title=label_column,
            loc="center right",
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend,
            frameon=False,
            borderaxespad=1
        )

    plt.tight_layout(rect=[0.05, 0, 0.85, 1])
    output_combined = os.path.join(output_dir, f"{prefix}.png")
    fig.savefig(output_combined, dpi=300)
    plt.close(fig)
    print(f"[INFO] Combined image saved to: {output_combined}")

def create_groupwise_pca_plots(
    data,
    category_col,
    output_dir,
    prefix="groupwise_pca_",
    n_components=2,
    color_column=None,
    color_data=None  # new argument
):
    """
    Applies PCA to subsets of data defined by a categorical column, generates one plot per group,
    and saves PCA loadings for each.

    Parameters:
        data (DataFrame): The input dataset.
        category_col (str): Column used to split the dataset into groups.
        output_dir (str): Directory where the plots and loadings will be saved.
        prefix (str): Prefix for output file names.
        n_components (int): Number of principal components.
        color_column (str, optional): Column used for coloring the scatter points.
        color_data (DataFrame, optional): Alternative DataFrame from which to get color labels.
    """

    os.makedirs(output_dir, exist_ok=True)

    if category_col not in data.columns:
        raise ValueError(f"The column '{category_col}' is not present in the PCA DataFrame.")

    if color_column:
        if color_data is None:
            color_data = data
        if color_column not in color_data.columns:
            raise ValueError(f"The color column '{color_column}' is not present in the provided color DataFrame.")

    unique_groups = sorted(data[category_col].dropna().unique())
    all_groups = unique_groups + ['All']

    fig, axes = plt.subplots(1, len(all_groups), figsize=(7 * len(all_groups), 6))
    if len(all_groups) == 1:
        axes = [axes]

    loadings_dict = {}

    for idx, group in enumerate(all_groups):
        if group == 'All':
            df_group = data.copy()
            color_group = color_data.copy() if color_column else None
            title = "All Data"
        else:
            df_group = data[data[category_col] == group]
            color_group = color_data[data[category_col] == group] if color_column else None
            title = f"{category_col} = {group}"

        numeric_cols = df_group.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise ValueError(f"At least two numeric columns are required for PCA in group '{group}'.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_group[numeric_cols])

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(scaled_data)
        explained = pca.explained_variance_ratio_ * 100

        ax = axes[idx]

        if color_column:
            labels = color_group[color_column].astype(str)
            unique_labels = sorted(labels.unique())
            palette = sns.color_palette("tab10", len(unique_labels))
            color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
            point_colors = labels.map(color_map)
            scatter = ax.scatter(components[:, 0], components[:, 1], c=point_colors, alpha=0.6)
        else:
            scatter = ax.scatter(components[:, 0], components[:, 1], alpha=0.6, color="royalblue")

        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({explained[0]:.2f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.2f}%)")

        if color_column:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=label,
                           markerfacecolor=color_map[label], markersize=8)
                for label in unique_labels
            ]
            ax.legend(handles=handles, title=color_column, loc='best')

        loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)],
                                index=numeric_cols)
        loadings_dict[group] = loadings

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{prefix}{category_col}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    for group, load in loadings_dict.items():
        load.to_csv(os.path.join(output_dir, f"{prefix}{category_col}_{group}_loadings.csv"))

    return loadings_dict

def generate_combined_tsne_plot(data_list, output_dir, label_column=None,
                                 prefix="tsne_combined", perplexity=30, n_iter=1000, random_state=42):
    """
    Applies t-SNE to 3 datasets, generates a combined figure with custom titles and side legend.
    """

    assert len(data_list) == 3, "You must provide exactly 3 datasets."

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    fontsize_labels = 14
    fontsize_ticks = 12
    fontsize_title = 16
    fontsize_legend = 14

    custom_titles = [
        "Volume and Diffusion Metrics",
        "Volume Metrics",
        "Diffusion Features"
    ]

    all_labels = []
    color_map = {}

    # Detect unique labels (assuming all have the same)
    if label_column and label_column in data_list[0].columns:
        all_labels = sorted(data_list[0][label_column].dropna().unique())
        palette = sns.color_palette("tab10", len(all_labels))
        color_map = {label: palette[i] for i, label in enumerate(all_labels)}

    for i, df in enumerate(data_list):
        ax = axes[i]

        # Filter numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError(f"Dataset {i+1} does not have enough numeric columns.")

        data_numeric = df[numeric_cols]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_numeric)

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        tsne_result = tsne.fit_transform(data_scaled)

        # Save t-SNE parameters
        with open(os.path.join(output_dir, f"{prefix}_dataset{i+1}_params.txt"), "w") as f:
            f.write(f"Perplexity: {perplexity}\n")
            f.write(f"Iterations: {n_iter}\n")
            f.write(f"Random State: {random_state}\n")
            f.write(f"Input features: {len(numeric_cols)}\n")

        # Colors
        if label_column and label_column in df.columns:
            labels = df[label_column].values
            point_colors = [color_map[label] for label in labels]
        else:
            labels = None
            point_colors = "royalblue"

        ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=point_colors, alpha=0.6)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=fontsize_labels)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.set_title(custom_titles[i], fontsize=fontsize_title)

    # Labels (a), (b), (c) on the left
    for i in range(3):
        fig.text(0.02, 0.91 - i * 0.31, f"({chr(97 + i)})", fontsize=fontsize_title, fontweight='bold')

    # Side legend
    if label_column and all_labels:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map[label], markersize=8)
            for label in all_labels
        ]
        fig.legend(
            handles,
            all_labels,
            title=label_column,
            loc="center right",
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend,
            frameon=False,
            borderaxespad=1
        )

    # Adjust space for the legend
    plt.tight_layout(rect=[0.05, 0, 0.85, 1])
    output_path = os.path.join(output_dir, f"{prefix}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Combined t-SNE image saved to: {output_path}")

def generate_combined_umap_plot(data_list, output_dir, label_column=None,
                                 prefix="umap_combined", n_neighbors=15, min_dist=0.1,
                                 metric='euclidean', random_state=42):
    """
    Applies UMAP to 3 datasets and generates a combined figure with scientific style.
    """

    assert len(data_list) == 3, "You must provide exactly 3 datasets."

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    fontsize_labels = 14
    fontsize_ticks = 12
    fontsize_title = 16
    fontsize_legend = 14

    custom_titles = [
        "Volume and Diffusion Metrics",
        "Volume Metrics",
        "Diffusion Features"
    ]

    all_labels = []
    color_map = {}

    # Create color map if labels exist
    if label_column and label_column in data_list[0].columns:
        all_labels = sorted(data_list[0][label_column].dropna().unique())
        palette = sns.color_palette("tab10", len(all_labels))
        color_map = {label: palette[i] for i, label in enumerate(all_labels)}

    for i, df in enumerate(data_list):
        ax = axes[i]

        # Validation and normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError(f"Dataset {i+1} does not have enough numeric columns.")

        data_numeric = df[numeric_cols]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_numeric)

        # UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            metric=metric, random_state=random_state)
        umap_result = reducer.fit_transform(data_scaled)

        # Save UMAP parameters
        with open(os.path.join(output_dir, f"{prefix}_dataset{i+1}_params.txt"), "w") as f:
            f.write(f"n_neighbors: {n_neighbors}\n")
            f.write(f"min_dist: {min_dist}\n")
            f.write(f"metric: {metric}\n")
            f.write(f"random_state: {random_state}\n")
            f.write(f"Input features: {len(numeric_cols)}\n")

        # Colors
        if label_column and label_column in df.columns:
            labels = df[label_column].values
            point_colors = [color_map[label] for label in labels]
        else:
            labels = None
            point_colors = "royalblue"

        ax.scatter(umap_result[:, 0], umap_result[:, 1], c=point_colors, alpha=0.6)
        ax.set_xlabel("UMAP Dimension 1", fontsize=fontsize_labels)
        ax.set_ylabel("UMAP Dimension 2", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.set_title(custom_titles[i], fontsize=fontsize_title)

    # External labels (a), (b), (c)
    for i in range(3):
        fig.text(0.02, 0.91 - i * 0.31, f"({chr(97 + i)})", fontsize=fontsize_title, fontweight='bold')

    # Side legend
    if label_column and all_labels:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map[label], markersize=8)
            for label in all_labels
        ]
        fig.legend(
            handles,
            all_labels,
            title=label_column,
            loc="center right",
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend,
            frameon=False,
            borderaxespad=1
        )

    plt.tight_layout(rect=[0.05, 0, 0.85, 1])
    output_path = os.path.join(output_dir, f"{prefix}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Combined UMAP image saved to: {output_path}")

def generate_dimred_grid(data_list, output_dir, label_column=None,
                         prefix="dimred_grid",
                         dataset_titles=None,
                         perplexity=30, n_iter=1000,
                         umap_neighbors=15, umap_min_dist=0.1,
                         umap_metric='euclidean',
                         random_state=42):
    """
    Applies PCA, t-SNE, and UMAP to a list of datasets and generates a combined figure with:
    - Rows = datasets
    - Columns = techniques
    - Titles as row separators (left-aligned)
    """

    if not isinstance(data_list, list):
        data_list = [data_list]

    techniques = ["PCA", "t-SNE", "UMAP"]
    n_datasets = len(data_list)
    n_rows = n_datasets
    n_cols = len(techniques)

    if dataset_titles is None:
        dataset_titles = [f"Dataset {i+1}" for i in range(n_datasets)]

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(6.0 * n_cols, 5.0 * n_rows),
        squeeze=False,
        gridspec_kw={'hspace': 0.3, 'wspace': 0.3}
    )

    fontsize_labels = 16
    fontsize_ticks = 14
    fontsize_title = 18
    fontsize_legend = 16

    panel_labels = [f"({chr(97 + i)})" for i in range(n_datasets * n_cols)]

    color_map = {}
    all_labels = []
    if label_column and label_column in data_list[0].columns:
        all_labels = sorted(data_list[0][label_column].dropna().unique())
        palette = sns.color_palette("tab10", len(all_labels))
        color_map = {label: palette[i] for i, label in enumerate(all_labels)}

    label_index = 0
    for row, df in enumerate(data_list):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError(f"Dataset {row+1} does not have enough numeric columns.")

        data_numeric = df[numeric_cols]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_numeric)

        if label_column and label_column in df.columns:
            labels = df[label_column].values
            point_colors = [color_map[label] for label in labels]
        else:
            labels = None
            point_colors = "royalblue"

        # --- PCA ---
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)
        exp_var = pca.explained_variance_ratio_ * 100
        ax = axes[row, 0]
        ax.scatter(pca_result[:, 0], pca_result[:, 1], c=point_colors, alpha=0.6)
        ax.set_xlabel(f"PC1 ({exp_var[0]:.2f}%)", fontsize=fontsize_labels)
        ax.set_ylabel(f"PC2 ({exp_var[1]:.2f}%)", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.text(-0.15, 1.05, panel_labels[label_index], transform=ax.transAxes,
                fontsize=fontsize_title, fontweight='bold', va='top')
        label_index += 1

        # --- t-SNE ---
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        tsne_result = tsne.fit_transform(data_scaled)
        ax = axes[row, 1]
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=point_colors, alpha=0.6)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=fontsize_labels)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.text(-0.15, 1.05, panel_labels[label_index], transform=ax.transAxes,
                fontsize=fontsize_title, fontweight='bold', va='top')
        label_index += 1

        # --- UMAP ---
        reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist,
                            metric=umap_metric, random_state=random_state)
        umap_result = reducer.fit_transform(data_scaled)
        ax = axes[row, 2]
        ax.scatter(umap_result[:, 0], umap_result[:, 1], c=point_colors, alpha=0.6)
        ax.set_xlabel("UMAP Dimension 1", fontsize=fontsize_labels)
        ax.set_ylabel("UMAP Dimension 2", fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.text(-0.15, 1.05, panel_labels[label_index], transform=ax.transAxes,
                fontsize=fontsize_title, fontweight='bold', va='top')
        label_index += 1

        # --- Title to separate ---
        if dataset_titles:
            axes[row, 0].annotate(dataset_titles[row],
                                  xy=(-0.55, 0.5), xycoords='axes fraction',
                                  fontsize=fontsize_title + 2,
                                  fontweight='bold', va='center', ha='right')

    # Titles by technique (columns)
    for col, technique in enumerate(techniques):
        axes[0, col].set_title(technique, fontsize=fontsize_title + 2, fontweight='bold', pad=20)

    # Legend
    legend = None
    if label_column and all_labels:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map[label], markersize=8)
            for label in all_labels
        ]
        legend = fig.legend(
            handles,
            all_labels,
            title=label_column,
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend,
            frameon=False
        )

    fig.subplots_adjust(left=0.15, right=0.88, top=0.93, bottom=0.07, hspace=0.4, wspace=0.3)

    output_path = os.path.join(output_dir, f"{prefix}.png")
    if legend:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
    else:
        plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Combined figure saved to: {output_path}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_multi_plot(
    df,
    plot_type,
    variables,
    group_by,
    ncols=2,
    figsize=(16, 10),
    save_path=None,
    palette='Set2',
    custom_bins=None
):
    sns.set(style="whitegrid")
    sns.set_context("talk", font_scale=1.2)

    n = len(variables)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    common_legend = None

    for i, var in enumerate(variables):
        ax = axes[i]
        ax.set_title(f"({chr(97+i)})", loc='left', fontsize=16, weight='bold')

        if plot_type == "box":
            plot = sns.boxplot(data=df, x=group_by, y=var, hue=group_by, palette=palette, ax=ax)
            # Add a legend to boxplots
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels and not common_legend:
                common_legend = (handles, labels)

        elif plot_type == "hist":
            if custom_bins and var in custom_bins:
                bin_info = custom_bins[var]
                bin_col = f"_bin_{var}"
                df[bin_col] = pd.cut(df[var], bins=bin_info["bins"], labels=bin_info["labels"], include_lowest=True)
                plot_data = df.groupby([bin_col, group_by], observed=False).size().reset_index(name="count")
                plot = sns.barplot(data=plot_data, x=bin_col, y="count", hue=group_by, palette=palette, ax=ax)
                ax.set_xlabel(var)
                df.drop(columns=[bin_col], inplace=True)
            else:
                plot = sns.histplot(data=df, x=var, hue=group_by, multiple='dodge', palette=palette, ax=ax)
                ax.set_ylabel("Count")

        elif plot_type == "bar":
            plot = sns.barplot(data=df, x=group_by, y=var, hue=group_by, palette=palette, ax=ax)

        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

        # Make sure there is a legend
        if not common_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                common_legend = (handles, labels)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Delete empty axes if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # Common leyend
    if common_legend:
        fig.legend(*common_legend, loc="lower center", ncol=len(common_legend[1]), fontsize=12)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"✅ Guardado en: {save_path}")
    else:
        plt.show()

# execute
# load data
df = pd.read_csv("../data/generated_data/filtered_data/tp1_filtered_data_TP1.csv")
vars_boxplot = ['Age','SDMT','SSST','SixMWT','VOL_Lesion']
vars_hist = ['EDSS','Sex','MSType', 'Time_since_diagnosis']


custom_bins = {
    'Time_since_diagnosis': {
        'bins': [0, 1, 5, 10, 20, 100],
        'labels': ['<1', '1-5', '5-10', '10-20', '20+']
    }
}
# Boxplots
generate_multi_plot(
    df=df,
    plot_type='box',
    variables=vars_boxplot,
    group_by='Cohort',
    ncols=3,
    figsize=(20, 12),
    save_path='../../results/exploratory_analysis/boxplots.png'
)

# Histograms
generate_multi_plot(
    df=df,
    plot_type="hist",
    variables=['EDSS','Sex','MSType', 'Time_since_diagnosis'],
    group_by="Cohort",
    custom_bins=custom_bins,
    save_path='../../results/exploratory_analysis/combined_plots.png'
)
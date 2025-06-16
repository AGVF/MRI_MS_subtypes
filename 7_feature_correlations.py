# --- REQUIRED LIBRARIES ---
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Define clinical variables
clinical_variables = ['SDMT', 'EDSS', 'SSST', 'SixMWT', 'VOL_Lesion']

# Region classification
region_type = {
    'globus_pallidus': 'subcortical', 'putamen': 'subcortical', 'caudate': 'subcortical',
    'hippocampus': 'subcortical', 'thalamus': 'subcortical', 'subthalamic_nucleus_total': 'subcortical',
    'Anterior_thalamic_radiation': 'white_matter_tract', 'Corticospinal_tract': 'white_matter_tract',
    'Cingulate_gyrus': 'white_matter_tract', 'Forceps_major': 'white_matter_tract',
    'Forceps_minor': 'white_matter_tract', 'fronto_occipital': 'white_matter_tract',
    'inferior_longitudinal_fasciculus': 'white_matter_tract', 'superior_longitudinal_fasciculus': 'white_matter_tract',
    'uncinate_fasciculus': 'white_matter_tract', 'fornix_total': 'white_matter_tract',
    'ICC_vol': 'global_volume', 'CSF_vol': 'global_volume', 'GM_vol': 'global_volume',
    'WM_vol': 'global_volume', 'brainstem': 'global_volume', 'lateral_ventricle_total': 'ventricle',
    '3rd_ventricle': 'ventricle', '4th_ventricle': 'ventricle', 'cerebellum_total': 'cerebellar',
    'occipital_total_gm': 'cortical', 'occipital_total_wm': 'cortical'
}

# Function to extract metrics and regions from DataFrame columns
def extract_metrics_and_regions(df):
    columns = []
    for col in df.columns:
        if any(m in col for m in ['FA', 'MD', 'Vol', 'VOL']) and '_' in col:
            parts = col.split('_')
            metric = parts[0]
            region = '_'.join(parts[1:])
            columns.append((col, metric, region, region_type.get(region, 'other')))
    return columns

# Compue correlations between clinical variables and metrics before and after a treatment
def compute_correlations(df_before, df_after):
    columns_info = extract_metrics_and_regions(df_before)
    results = []

    for col, metric, region, rtype in columns_info:
        for var in clinical_variables:
            if var in df_before.columns:
                rho_b, p_b = spearmanr(df_before[col], df_before[var], nan_policy='omit')
                rho_a, p_a = spearmanr(df_after[col], df_after[var], nan_policy='omit')
                results.append({
                    'RegionType': rtype, 'Metric': metric, 'Region': region,
                    'Variable': var, 'Before': f"r={rho_b:.2f}; p={p_b:.3f}", 'After': f"r={rho_a:.2f}; p={p_a:.3f}",
                    'pval_before': p_b, 'pval_after': p_a
                })
    return pd.DataFrame(results)

# Function to generate PDF tables
def generate_table_pdf(df, title, pdf):
    fig, ax = plt.subplots(figsize=(11, max(3, len(df) * 0.4)))
    ax.axis('tight')
    ax.axis('off')

    table = pd.DataFrame(df)
    cells = []
    for i, row in table.iterrows():
        c = []
        for col in ['Region', 'Variable', 'Before', 'After']:
            text = row[col]
            color = 'yellow' if (col == 'Before' and row['pval_before'] < 0.05) or (col == 'After' and row['pval_after'] < 0.05) else 'white'
            c.append({'text': text, 'facecolor': color})
        cells.append(c)

    table_np = [[c['text'] for c in row] for row in cells]
    cell_colors = [[c['facecolor'] for c in row] for row in cells]

    ax.table(cellText=table_np, cellColours=cell_colors,
             colLabels=['Region', 'Clinical Var', 'Before', 'After'],
             loc='center', cellLoc='center')
    ax.set_title(title, fontsize=14, weight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# Main function to generate the report
def generate_report(df_before, df_after, pdf_name):
    results = compute_correlations(df_before, df_after)

    with PdfPages(pdf_name) as pdf:
        for rtype in results['RegionType'].unique():
            for metric in results[results['RegionType'] == rtype]['Metric'].unique():
                subset_all = results[(results['RegionType'] == rtype) & (results['Metric'] == metric)]
                generate_table_pdf(subset_all, f"{rtype.capitalize()} – {metric} (All Data)", pdf)

                if 'Cohort' in df_before.columns:
                    for cohort in df_before['Cohort'].dropna().unique():
                        df_b = df_before[df_before['Cohort'] == cohort]
                        df_a = df_after[df_after['Cohort'] == cohort]
                        sub_res = compute_correlations(df_b, df_a)
                        sub_filt = sub_res[(sub_res['RegionType'] == rtype) & (sub_res['Metric'] == metric)]
                        if not sub_filt.empty:
                            generate_table_pdf(sub_filt, f"{rtype.capitalize()} – {metric} (Cohort {cohort})", pdf)

    print(f"Report saved as {pdf_name}")


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmaps_from_datasets(datasets, output_dir):
    """
    Genera heatmaps de medias por Cohort para variables de tipo diffusion, volumen y clínicas.
    Acepta uno o varios DataFrames (pueden tener variables distintas).

    Parameters:
        datasets (DataFrame o lista de DataFrames): Datos de entrada
        output_dir (str): Ruta donde se guardarán las figuras
    """
    # === Asegurar directorio de salida ===
    os.makedirs(output_dir, exist_ok=True)

    # === Si es un solo DataFrame, convertirlo en lista ===
    if not isinstance(datasets, list):
        datasets = [datasets]

    # === Variables clínicas relevantes ===
    clinical_data = ['EDSS', "Cohort", "TP", 'Age', 'Sex', 'MSType', 'Time_since_diagnosis',
                     'SRT-L', 'SRT-C', 'SRT-D', 'Spatial RT', 'Delayed Spatial RT',
                     'SDMT', 'PASAT', 'WLG', 'SSST', 'SixMWT', 'VOL_Lesion', 'ID']

    # === Inicializar contenedor de variables por tipo ===
    all_vars = {'clinical': [], 'volume': [], 'diffusion': []}
    merged_df = None

    # === Unir datasets y clasificar columnas ===
    for df in datasets:
        if 'Cohort' not in df.columns:
            raise ValueError("Cada DataFrame debe contener la columna 'Cohort'.")
        if merged_df is None:
            merged_df = df.copy()
        else:
            merge_keys = [k for k in ['ID', 'Cohort'] if k in df.columns and k in merged_df.columns]
            merged_df = pd.merge(merged_df, df, on=merge_keys, how='outer')

        for col in df.columns:
            if col.startswith("FA_") or col.startswith("MD_"):
                all_vars['diffusion'].append(col)
            elif col.startswith("VOL_"):
                all_vars['volume'].append(col)
            elif col in clinical_data and col != "Cohort":
                all_vars['clinical'].append(col)

    # Eliminar duplicados
    for key in all_vars:
        all_vars[key] = sorted(set(all_vars[key]))

    # Filtrar variables realmente presentes en el merged_df
    valid_vars = [col for col in sum(all_vars.values(), []) if col in merged_df.columns]

    if not valid_vars:
        raise ValueError("No se encontraron variables válidas en los datasets.")

    # Calcular medias por Cohort
    mean_df = merged_df.groupby("Cohort")[valid_vars].mean().T

    # === Función auxiliar para graficar y guardar ===
    def plot_block(block, title, filename):
        if block.empty:
            return
        plt.figure(figsize=(10, max(1, 0.4 * len(block))))
        ax = sns.heatmap(block, cmap='coolwarm', annot=True, fmt=".2f",
                         cbar=True, cbar_kws={'label': 'Mean Value'})
        ax.set_title(title, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    # === Crear gráficos si hay variables ===
    if all_vars['diffusion']:
        plot_block(mean_df.loc[[v for v in all_vars['diffusion'] if v in mean_df.index]],
                   "(a) Diffusion Only Dataset", "mean_diffusion.png")
    if all_vars['volume']:
        plot_block(mean_df.loc[[v for v in all_vars['volume'] if v in mean_df.index]],
                   "(b) Volume Only Dataset", "mean_volume.png")
    if all_vars['clinical']:
        plot_block(mean_df.loc[[v for v in all_vars['clinical'] if v in mean_df.index]],
                   "(c) Clinical Features", "mean_clinical.png")

    print(f"Heatmaps guardados en: {output_dir}")
    
df_diff = pd.read_csv("../data/updated_data_sk_corrected/data_trans_norm_diff.csv")
df_vol = pd.read_csv("../data/updated_data_sk_corrected/datos_preprocesados_trans_norm_vol.csv")

# También puede usarse con solo un dataset
generate_heatmaps_from_datasets([df_diff, df_vol], "../initial_exploring/new/mean_heatmaps")
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def evaluate_top_predictors(df, clinical_vars, predictor_type='vol', top_n=5, extra_vars=None):
    """
    Performs simple linear regressions between each clinical variable and selected predictors,
    including multiple comparisons correction using FDR (Benjamini-Hochberg).
    
    Parameters:
        df (DataFrame): Input dataset.
        clinical_vars (list): List of clinical variable names.
        predictor_type (str): One of 'vol', 'diff', or 'vol_diff' to select predictor types.
        top_n (int): Number of top predictors to retain per clinical variable.
        extra_vars (list): Additional variables to include as predictors (optional).
        
    Returns:
        summary_df (DataFrame): DataFrame with R², p-values, and FDR-adjusted significance for top predictors.
    """
    df = df.drop(columns=['cohort', 'TP'], errors='ignore')

    all_columns = df.columns.tolist()
    clinical_set = set(clinical_vars)

    # Select predictors based on type
    if predictor_type == 'vol':
        predictors = [v for v in all_columns if 'VOL' in v and v not in clinical_set]
    elif predictor_type == 'diff':
        predictors = [v for v in all_columns if ('FA' in v or 'MD' in v) and v not in clinical_set]
    elif predictor_type == 'vol_diff':
        predictors = [v for v in all_columns if any(x in v for x in ['VOL', 'FA', 'MD']) and v not in clinical_set]
    else:
        raise ValueError("predictor_type must be 'vol', 'diff', or 'vol_diff'.")

    # Optionally exclude specific variables
    if predictor_type in ['vol', 'vol_diff']:
        predictors = [p for p in predictors if not p.startswith('VOL_Lesion')]

    # Add extra variables if provided
    if extra_vars:
        extra_valid = [v for v in extra_vars if isinstance(v, str) and v not in clinical_set]
        predictors = list(set(predictors + extra_valid))

    all_results = []

    for clinical_var in clinical_vars:
        results = []
        for predictor in predictors:
            try:
                formula = f'Q("{clinical_var}") ~ Q("{predictor}")'
                model = smf.ols(formula=formula, data=df).fit()

                r2 = model.rsquared
                p_val = model.pvalues[1] if len(model.pvalues) > 1 else None

                results.append({
                    'Clinical Variable': clinical_var,
                    'Predictor': predictor,
                    'R²': round(r2, 4),
                    'p-value': f'{p_val:.6g}' if p_val is not None else None
                })

            except Exception:
                continue

        top_results = sorted(results, key=lambda x: x['R²'], reverse=True)[:top_n]
        all_results.extend(top_results)

    # FDR correction
    if all_results:
        raw_pvals = [float(r['p-value']) if r['p-value'] is not None else 1.0 for r in all_results]
        rejected, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

        for i, res in enumerate(all_results):
            res['p-value (FDR)'] = f'{corrected_pvals[i]:.6g}'
            res['Significant (FDR < 0.05)'] = 'Yes' if rejected[i] else 'No'

    summary_df = pd.DataFrame(all_results)
    return summary_df

# List of clinical variables to analyze
clinical_cols = ['EDSS', 'Age', 'Time_since_diagnosis', 'SDMT',
                 'SSST', 'SixMWT', 'VOL_Lesion', 'ID', 'Cohort']
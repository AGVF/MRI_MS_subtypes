# Batch Effect correction script
# Load packages
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy, f_oneway
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder 
from statsmodels.multivariate.manova import MANOVA


# Create the functions
# Functions to correct for a batch effect
# Auxiliary function to estimate a batch effect
def estimate_threshold_batch(df, batch_col, factors, num_permutaciones=100):
    """
    Estimate batch effect based on column prefixes FA_, MD_ o VOL_.
    """

    # Seleccionar variables relevantes
    diffusion_vars = [col for col in df.columns if col.startswith("FA_") or col.startswith("MD_") or col.startswith("VOL_")]
    data = df[diffusion_vars]
    batches = df[batch_col].unique()

    permuted_results = {
        "AveDist": [],
        "ANOVA Mean P-value": [],
        "MANOVA P-value": []
    }

    for i in range(num_permutaciones):
        shuffled_batch = np.random.permutation(df[batch_col].values)
        df_shuffled = df.copy()
        df_shuffled[batch_col] = shuffled_batch

        # 1. AveDist
        batch_means = [data[df_shuffled[batch_col] == batch].mean() for batch in batches]
        batch_distances = cdist(batch_means, batch_means, 'euclidean')
        ave_dist = np.mean(batch_distances[np.tril_indices(len(batch_means), k=-1)])
        permuted_results["AveDist"].append(ave_dist)

        # 2. ANOVA
        anova_pvalues = [
            f_oneway(*[data[df_shuffled[batch_col] == batch][col] for batch in batches])[1]
            for col in data.columns
        ]
        permuted_results["ANOVA Mean P-value"].append(np.mean(anova_pvalues))

        # 3. MANOVA
        try:
            formula = f"{' + '.join(data.columns)} ~ {batch_col}"
            df_temp = df_shuffled[[batch_col] + list(data.columns)]
            maov = MANOVA.from_formula(formula, data=df_temp)
            result = maov.mv_test()
            wilks_result = result.results[batch_col]['stat']
            manova_p = wilks_result.loc["Wilks' lambda", 'Pr > F']
        except Exception as e:
            print(f" Error in MANOVA permutation {i+1}: {e}")
            print(f"Formula: {formula}")
            print(f"Columns: {list(data.columns)}")
            print(f"Shape df_temp: {df_temp.shape}")
            manova_p = np.nan

        permuted_results["MANOVA P-value"].append(manova_p)

    # Filter NaNs before calculating percentiles
    valid_manovas = [p for p in permuted_results["MANOVA P-value"] if not np.isnan(p)]
    if len(valid_manovas) == 0:
        raise RuntimeError("No MANOVA p-values could be calculated in any permutation.")

    thresholds = {
        "AveDist": np.percentile(permuted_results["AveDist"], 95),
        "ANOVA Mean P-value": np.percentile(permuted_results["ANOVA Mean P-value"], 5),
        "MANOVA P-value": np.percentile(valid_manovas, 5)
    }

    return thresholds

# Function to check for a batch effect
def check_batch(df, batch_col, factors, thresholds=None, output_path=None):
    """
    Batch effect diagnosis using only FA_ and MD_ variables.
    """

    # Filter only relevant diffusion variables
    diffusion_vars = [col for col in df.columns if col.startswith("FA_") or col.startswith("MD_") or col.startswith("VOL_")]
    data = df[diffusion_vars]
    batches = df[batch_col].unique()

    # 1. AveDist
    batch_means = [data[df[batch_col] == batch].mean().values for batch in batches]
    batch_distances = cdist(batch_means, batch_means, 'euclidean')
    avedist = np.mean(batch_distances[np.tril_indices(len(batch_means), k=-1)])

    # 2. ANOVA
    anova_pvalues = [
        f_oneway(*[data[df[batch_col] == batch][col] for batch in batches])[1]
        for col in data.columns
    ]
    anova_mean_p = np.mean(anova_pvalues)

    # 3. MANOVA
    try:
        formula = f"{' + '.join(data.columns)} ~ {batch_col}"
        df_temp = df[[batch_col] + list(data.columns)]
        maov = MANOVA.from_formula(formula, data=df_temp)
        result = maov.mv_test()
        print(result)
        p_line = str(result).split('\n')
        p_value_line = [line for line in p_line if 'Wilks' in line]
        manova_p = float(p_value_line[0].split()[-1]) if p_value_line else np.nan
    except Exception as e:
        manova_p = np.nan
    
    # Defaults if no thresholds are provided
    if thresholds is None:
        thresholds = {
            "AveDist": 1.0,
            "ANOVA Mean P-value": 0.05,
            "MANOVA P-value": 0.05
        }

    report = []
    report.append("----- Batch Effect Evaluation -----\n")
    report.append(f"AveDist: {avedist:.4f} | Threshold: {thresholds['AveDist']:.4f}")
    report.append(f"ANOVA Mean P-value: {anova_mean_p:.4e} | Threshold: {thresholds['ANOVA Mean P-value']:.4e}")
    report.append(f"MANOVA P-value: {manova_p:.4e} | Threshold: {thresholds['MANOVA P-value']:.4e}\n")

    if avedist > thresholds["AveDist"]:
        report.append("[⚠️] AveDist above threshold → potential batch effect.")
    else:
        report.append("[✅] AveDist within acceptable range.")

    if anova_mean_p < thresholds["ANOVA Mean P-value"]:
        report.append("[⚠️] ANOVA p-value suggests group differences.")
    else:
        report.append("[✅] ANOVA p-value above threshold.")

    if manova_p < thresholds["MANOVA P-value"]:
        report.append("[⚠️] MANOVA indicates significant multivariate group differences.")
    else:
        report.append("[✅] MANOVA does not detect multivariate group differences.")

    # Conclusion based on thresholds
    if (avedist > thresholds["AveDist"] or
        anova_mean_p < thresholds["ANOVA Mean P-value"] or
        manova_p < thresholds["MANOVA P-value"]):
        report.append("\n🔴 Conclusion: Batch effect likely present.")
    else:
        report.append("\n🟢 Conclusion: No evidence of batch effect.")

    if output_path:
        with open(output_path, 'w') as f:
            f.write("\n".join(report))
    else:
        return {
            "AveDist": avedist,
            "ANOVA Mean P-value": anova_mean_p,
            "MANOVA P-value": manova_p,
            "thresholds": thresholds,
            "evaluation_messages": report
        }

# Function to compute residuals using random forest
def correct_batch_effects(df, factors, exclude=[]):
    """
    Corrects batch effects in all variables of the dataset using Random Forest regression.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        factors (list): List of columns representing confounding variables (e.g., age, cohort, sex).
        exclude (list): List of columns that should NOT be modified.

    Returns:
        pd.DataFrame with corrected values, plus the excluded columns and original confounding factors.
    """
    df_corrected = pd.DataFrame(index=df.index)  # Start with an empty DataFrame with the same index

    # Identify variables to correct
    variables = [col for col in df.columns if col not in factors and col not in exclude]

    # One-Hot Encode categorical confounding variables
    cat_vars = [var for var in factors if df[var].dtype == 'object']
    enc = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cats = pd.DataFrame(enc.fit_transform(df[cat_vars]),
                                 columns=enc.get_feature_names_out(cat_vars),
                                 index=df.index)

    # Build the covariate matrix (numeric + encoded categorical variables)
    num_vars = [var for var in factors if var not in cat_vars]
    X = pd.concat([df[num_vars], encoded_cats], axis=1)

    # Correct each target variable
    for var in variables:
        y = df[var]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        df_corrected[var] = y - y_pred  # Residuals as corrected values

    # Add back the excluded columns and original confounding factors
    df_corrected = pd.concat([
        df_corrected,
        df[exclude],
        df[factors]
    ], axis=1)

    return df_corrected

# Function to apply batch correction using age as the sole factor
def age_and_batch_correction(df, batch_col='Batch', feature_prefixes=['FA', 'MD']):
    """
    Corrects the effect of age and batch in specific columns using linear regression.

    Parameters:
        df (pd.DataFrame): Original data.
        batch_col (str): Column with batch identifiers.
        feature_prefixes (list): List of column prefixes to apply the correction to.

    Returns:
        pd.DataFrame with corrected columns and untouched columns preserved.
    """
    df_corrected = pd.DataFrame(index=df.index)  # Initialize an empty DataFrame

    # One-hot encode the batch column
    batch_dummies = pd.get_dummies(df[batch_col], drop_first=True).astype(float)

    for col in df.columns:
        if any(col.startswith(prefix) for prefix in feature_prefixes):
            y = df[col].values
            X = pd.concat([df[['Age']], batch_dummies], axis=1)
            X = sm.add_constant(X)
            X = X.apply(pd.to_numeric, errors='raise')

            model = sm.OLS(y, X).fit()
            beta_age = model.params['Age']

            # Adjust for age (correct only the age component)
            corrected = y - (df['Age'] - df['Age'].mean()) * beta_age
            df_corrected[col] = corrected
        else:
            # Copy columns that are not being corrected
            df_corrected[col] = df[col]

    return df_corrected

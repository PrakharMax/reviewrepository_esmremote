# phase_6_environmental_moderation_lmm.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys
from config import DATA_DIR
import warnings
warnings.filterwarnings("ignore")

# Path configuration to the processed evening survey dataset
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"


def calculate_pseudo_r2(model_result, model_name="Model"):
    """
    Calculates Marginal and Conditional R-squared values for the linear mixed-effects
    model based on the Nakagawa & Schielzeth (2013) variance decomposition framework.
    """
    try:
        # Variance calculation of the fixed effect predictions
        fixed_effects_var = np.var(model_result.predict())

        # Extraction of random effect variance (between-person tau_00)
        random_effects_var = float(model_result.cov_re.iloc[0, 0])

        # Extraction of residual variance (within-person sigma^2 noise)
        residual_var = model_result.scale

        # Total variance summation across fixed, random, and residual components
        total_var = fixed_effects_var + random_effects_var + residual_var

        # Variance proportions derivation
        r2_marginal = fixed_effects_var / total_var
        r2_conditional = (fixed_effects_var + random_effects_var) / total_var

        print(f"\nINFO: {model_name} Pseudo R-Squared Diagnostics")
        print(f"Marginal R2 (Fixed Effects Proportion):     {r2_marginal:.3f}")
        print(f"Conditional R2 (Total Explained Variance):   {r2_conditional:.3f}")

        return r2_marginal, r2_conditional

    except Exception as e:
        print(f"ERROR: Could not compute pseudo R2 variance components: {e}")
        return np.nan, np.nan


def run_moderation_analysis():
    """
    Executes a linear mixed-effects moderation analysis evaluating the additive
    and interactive properties of cumulative carbon dioxide and localized lighting flags.
    """
    print("INFO: Loading processed evening dataset for environmental moderation modeling...")
    try:
        df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
        df_eve.columns = df_eve.columns.str.strip()
    except Exception as e:
        print(f"ERROR: Failed to load specified data asset: {e}")
        sys.exit(1)

    # Statistical array definition for targeted environmental variables
    target_cols = [
        'Productivity_Retention_Gap',
        'Within_Daily_CO2',
        'Light_Impact_Flag'
    ]

    # Enforcing continuous numeric float formatting across target arrays
    for col in target_cols:
        df_eve[col] = pd.to_numeric(df_eve[col], errors='coerce')

    df_eve['ID'] = df_eve['ID'].astype(str)

    # Complete observation filtration isolating intersecting predictor rows
    df_mod = df_eve.dropna(subset=target_cols + ['ID']).copy()
    print(f"INFO: Running linear mixed moderation models on {len(df_mod)} complete observation rows.")

    # ---------------------------------------------------------
    # Model 1: Additive Baseline Model
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: INDEPENDENT ADDITIVE EFFECTS (CO2 + LIGHT)")
    print("=" * 60)
    try:
        mdl_add = smf.mixedlm(
            formula="Productivity_Retention_Gap ~ Within_Daily_CO2 + Light_Impact_Flag",
            data=df_mod,
            groups=df_mod["ID"]
        ).fit(method='lbfgs')

        print(mdl_add.summary().tables[1])
        calculate_pseudo_r2(mdl_add, "Model 1")
    except Exception as e:
        print(f"ERROR: Model 1 estimation failed: {e}")

    # ---------------------------------------------------------
    # Model 2: Interaction/Moderation Model
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: INTERACTIVE BUFFERING EFFECTS (CO2 * LIGHT)")
    print("=" * 60)
    try:
        mdl_int = smf.mixedlm(
            formula="Productivity_Retention_Gap ~ Within_Daily_CO2 * Light_Impact_Flag",
            data=df_mod,
            groups=df_mod["ID"]
        ).fit(method='lbfgs')

        print(mdl_int.summary().tables[1])
        calculate_pseudo_r2(mdl_int, "Model 2")

        print("\nSTATISTICAL INFERENCE: EXACT VARIABLE P-VALUES")
        pvals = mdl_int.pvalues

        # Statsmodels internally reformats interaction columns using a colon notation string ("VarA:VarB")
        interaction_term = "Within_Daily_CO2:Light_Impact_Flag"

        print(f"Intercept:                        {pvals['Intercept']:.10f}")
        print(f"Within Daily CO2:                 {pvals['Within_Daily_CO2']:.10f}")
        print(f"Light Impact Flag:                {pvals['Light_Impact_Flag']:.10f}")
        if interaction_term in pvals:
            print(f"CO2 * Light Interaction:          {pvals[interaction_term]:.10f}")
        else:
            print("WARNING: Interaction parameter key mismatch during automated dictionary parsing.")

    except Exception as e:
        print(f"ERROR: Model 2 estimation failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_moderation_analysis()
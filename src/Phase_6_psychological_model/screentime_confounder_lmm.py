# phase_6_screentime_confounder_lmm.py
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


def run_screentime_analysis():
    """
    Executes an exploratory analysis tracking digital fatigue covariates alongside
    carbon dioxide metrics against evening productivity gaps.

    Note: This module serves as an exploratory verification block for lifestyle
    confounder analysis and was not included in the primary publication results.
    """
    print("INFO: Loading processed evening dataset for confounder modeling...")
    try:
        df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
        df_eve.columns = df_eve.columns.str.strip()
    except Exception as e:
        print(f"ERROR: Failed to load specified data asset: {e}")
        sys.exit(1)

    # Statistical array definition for targeted validation parameters
    target_cols = [
        'Productivity_Retention_Gap',
        'Within_Daily_CO2',
        'Workday_Screen_Time_Added'
    ]

    # Enforcing continuous numeric float formatting across target arrays
    for col in target_cols:
        df_eve[col] = pd.to_numeric(df_eve[col], errors='coerce')

    df_eve['ID'] = df_eve['ID'].astype(str)

    # Data Quality Assurance: Correction of out-of-bounds negative behavioral metrics
    neg_count = (df_eve['Workday_Screen_Time_Added'] < 0).sum()
    print(f"INFO: Data QA check identified {neg_count} out-of-bounds negative screen-time records.")

    if neg_count > 0:
        print("INFO: Re-indexing negative behavioral entries to null values for model safety.")
        df_eve.loc[df_eve['Workday_Screen_Time_Added'] < 0, 'Workday_Screen_Time_Added'] = np.nan

    # Complete observation filtration across specified variables
    df_mask = df_eve.dropna(subset=target_cols + ['ID']).copy()
    print(f"INFO: Running linear mixed models on {len(df_mask)} complete observation rows.")

    # ---------------------------------------------------------
    # Model 1: Screen Time Univariate Assessment
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: COGNITIVE OVERHEAD VS SCREEN TIME")
    print("=" * 60)
    try:
        mdl_screen = smf.mixedlm(
            formula="Productivity_Retention_Gap ~ Workday_Screen_Time_Added",
            data=df_mask,
            groups=df_mask["ID"]
        ).fit(method='lbfgs')
        print(mdl_screen.summary().tables[1])
        calculate_pseudo_r2(mdl_screen, "Model 1")
    except Exception as e:
        print(f"ERROR: Model 1 execution failed: {e}")

    # ---------------------------------------------------------
    # Model 2: Multivariate Screen Time and Environmental Assessment
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: MULTIVARIATE CONFOUNDER ANALYSIS")
    print("=" * 60)
    try:
        mdl_duel = smf.mixedlm(
            formula="Productivity_Retention_Gap ~ Workday_Screen_Time_Added + Within_Daily_CO2",
            data=df_mask,
            groups=df_mask["ID"]
        ).fit(method='lbfgs')
        print(mdl_duel.summary().tables[1])
        calculate_pseudo_r2(mdl_duel, "Model 2")

        print("\nSTATISTICAL INFERENCE: EXACT VARIABLE P-VALUES")
        pvals = mdl_duel.pvalues
        zvals = mdl_duel.tvalues

        print(f"Intercept:                        {pvals['Intercept']:.10f}")
        print(f"Workday_Screen_Time_Added:        {pvals['Workday_Screen_Time_Added']:.10f}")
        print(f"Within_Daily_CO2:                 {pvals['Within_Daily_CO2']:.10f}")
        print(f"\nZ-score (Workday_Screen_Time):    {zvals['Workday_Screen_Time_Added']:.4f}")
    except Exception as e:
        print(f"ERROR: Model 2 execution failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_screentime_analysis()
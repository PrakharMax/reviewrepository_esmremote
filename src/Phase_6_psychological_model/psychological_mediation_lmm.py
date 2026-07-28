# phase_6_psychological_mediation_lmm.py
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


def run_mediation_analysis():
    """
    Executes a multi-stage linear mixed-effects regression pipeline to evaluate
    the statistical mediation pathway connecting CO2 exposure, symptoms, and productivity.
    """
    print("INFO: Loading processed evening dataset for mediation pathway tracking...")
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
        'Evening_Symptom_Count'
    ]

    # Enforcing continuous numeric float formatting across target arrays
    for col in target_cols:
        df_eve[col] = pd.to_numeric(df_eve[col], errors='coerce')

    df_eve['ID'] = df_eve['ID'].astype(str)

    # Complete observation filtration isolating valid longitudinal rows
    df_med = df_eve.dropna(subset=target_cols + ['ID']).copy()
    print(f"INFO: Complete data rows identified for mediation pipeline: {len(df_med)}")

    if len(df_med) == 0:
        print("ERROR: Zero observations remaining after dropping missing rows. Execution halted.")
        sys.exit(1)

    # ---------------------------------------------------------
    # Model 1: Path A Evaluation (CO2 -> Symptoms)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: MEDIATION PATH A (CO2 -> SYMPTOMS)")
    print("=" * 60)
    try:
        mdl_a = smf.mixedlm(
            formula="Evening_Symptom_Count ~ Within_Daily_CO2",
            data=df_med,
            groups=df_med["ID"]
        ).fit(method='lbfgs')

        print(mdl_a.summary().tables[1])
        calculate_pseudo_r2(mdl_a, "Model 1: CO2 -> Symptoms")

        print("\nSTATISTICAL INFERENCE p-VALUES")
        pvals_a = mdl_a.pvalues
        print(f"Within_Daily_CO2 parameter coefficient: {pvals_a['Within_Daily_CO2']:.10f}")

    except Exception as e:
        print(f"ERROR: Model 1 parameter estimation failed: {e}")

    print("\n" + "=" * 60 + "\n")

    # ---------------------------------------------------------
    # Model 2: Path B Evaluation (Symptoms -> Productivity)
    # ---------------------------------------------------------
    print("=" * 60)
    print("STATISTICAL LMM SUMMARY: MEDIATION PATH B (SYMPTOMS -> PRODUCTIVITY)")
    print("=" * 60)
    try:
        mdl_b = smf.mixedlm(
            formula="Productivity_Retention_Gap ~ Evening_Symptom_Count + Within_Daily_CO2",
            data=df_med,
            groups=df_med["ID"]
        ).fit(method='lbfgs')

        print(mdl_b.summary().tables[1])
        calculate_pseudo_r2(mdl_b, "Model 2: Symptoms -> Productivity")

        print("\nSTATISTICAL INFERENCE p-VALUES")
        pvals_b = mdl_b.pvalues
        print(f"Evening_Symptom_Count descriptor factor: {pvals_b['Evening_Symptom_Count']:.10f}")
        print(f"Within_Daily_CO2 direct effect factor:   {pvals_b['Within_Daily_CO2']:.10f}")

    except Exception as e:
        print(f"ERROR: Model 2 parameter estimation failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_mediation_analysis()
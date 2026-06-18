# phase_6_morning_cognitive_hangover_lmm.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys
from config import DATA_DIR
import warnings
warnings.filterwarnings("ignore")

# Path configuration to the processed morning survey dataset
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"


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


def extract_rating(val):
    """
    Safely extracts leading string indicators from Webropol rating scales
    and normalizes them to floating-point values.
    """
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).split('=')[0].strip())
    except Exception:
        return np.nan


def binarize_symptom(val):
    """
    Enforces strict binary classification (0 or 1) across self-reported
    building-related health symptoms to standardize cohort counts.
    """
    if pd.isna(val) or str(val).strip() in ['0', '0.0', 'None', 'nan']:
        return 0
    return 1


def run_morning_analysis_pipeline():
    """
    Executes the morning linear mixed-effects modeling loop evaluating
    overnight carbon dioxide exposures against building symptom burden and focus parameters.
    """
    print("INFO: Loading processed morning dataset for cognitive hangover modeling...")
    try:
        df_morn = pd.read_csv(MORNING_FILE, low_memory=False)
        df_morn.columns = df_morn.columns.str.strip()
    except Exception as e:
        print(f"ERROR: Failed to load specified data asset: {e}")
        sys.exit(1)

    # Automated structural verification for localized focus tracking headers
    try:
        focus_col = [c for c in df_morn.columns if 'Rate your focus level' in c][0]
    except IndexError:
        print("ERROR: Specified subjective cognitive tracking parameters missing from file header columns.")
        sys.exit(1)

    co2_col = 'Within_CO2'
    sleep_col = 'Control_Good_Sleep'
    symptom_keywords = ['Dry throat', 'Dry eyes', 'Headache', 'Stuffy nose', 'Cough']
    symptom_cols = [c for c in df_morn.columns if any(k in c for k in symptom_keywords)]

    print("INFO: Executing data normalization and binary symptom array mapping...")
    for col in symptom_cols:
        df_morn[col] = df_morn[col].apply(binarize_symptom)

    # Feature engineering of outcome counts and continuous variable structures
    df_morn['Symptom_Count'] = df_morn[symptom_cols].sum(axis=1)
    df_morn['Focus_Num'] = df_morn[focus_col].apply(extract_rating)
    df_morn['CO2_Num'] = pd.to_numeric(df_morn[co2_col], errors='coerce')
    df_morn['Sleep_Num'] = pd.to_numeric(df_morn[sleep_col], errors='coerce')
    df_morn['ID'] = df_morn['ID'].astype(str)

    # Complete observation filtration isolating valid predictor rows
    df_clean = df_morn.dropna(subset=['Focus_Num', 'CO2_Num', 'ID']).copy()
    print(f"INFO: Data cleaning complete. Total rows ready for linear mixed modeling: {len(df_clean)}")

    # ---------------------------------------------------------
    # Model 1: The Morning exposure effect (CO2 -> Symptoms)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: Morning exposure effect MODEL (CO2 -> SYMPTOMS)")
    print("=" * 60)
    try:
        mdl1 = smf.mixedlm(
            formula="Symptom_Count ~ CO2_Num + Sleep_Num",
            data=df_clean,
            groups=df_clean["ID"]
        ).fit(method='lbfgs')

        print(mdl1.summary().tables[1])
        calculate_pseudo_r2(mdl1, "Morning exposure effect")

        print("\nSTATISTICAL COEFFICIENT TRANSFORMATION")
        exact_co2_coef = mdl1.params['CO2_Num']
        jump_1000 = exact_co2_coef * 1000
        print(f"Unrounded CO2 continuous multiplier coefficient:  {exact_co2_coef:.7f}")
        print(f"Scaled exposure vector impact (+1000 ppm CO2 change): +{jump_1000:.3f} morning symptoms")

        print("\nSTATISTICAL INFERENCE p-VALUES")
        pvals1 = mdl1.pvalues
        print(f"CO2_Num (Overnight Exposure Variant): {pvals1['CO2_Num']:.5f}")

    except Exception as e:
        print(f"ERROR: Model 1 execution matrix optimization failed: {e}")

    print("\n" + "=" * 60 + "\n")

    # ---------------------------------------------------------
    # Model 2: The Focus Blockade (Symptoms -> Focus)
    # ---------------------------------------------------------
    print("=" * 60)
    print("STATISTICAL LMM SUMMARY: FOCUS BLOCKADE MODEL (SYMPTOMS -> FOCUS)")
    print("=" * 60)
    try:
        mdl2 = smf.mixedlm(
            formula="Focus_Num ~ Symptom_Count + CO2_Num + Sleep_Num",
            data=df_clean,
            groups=df_clean["ID"]
        ).fit(method='lbfgs')

        print(mdl2.summary().tables[1])
        calculate_pseudo_r2(mdl2, "Focus Blockade")

        print("\nSTATISTICAL INFERENCE p-VALUES")
        pvals2 = mdl2.pvalues
        print(f"Symptom_Count descriptor factor: {pvals2['Symptom_Count']:.5f}")
        print(f"CO2_Num parameter coefficient:   {pvals2['CO2_Num']:.5f}")
        print(f"Sleep_Num control variance:      {pvals2['Sleep_Num']:.5f}")

    except Exception as e:
        print(f"ERROR: Model 2 execution matrix optimization failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_morning_analysis_pipeline()
# phase_6_social_presence_confounder_lmm.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys
from config import DATA_DIR
import warnings
warnings.filterwarnings("ignore")

# Path configurations targeting processed survey and wearable datasets
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"
COZIE_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def calculate_pseudo_r2(model_result, model_name="Model"):
    """
    Calculates Marginal and Conditional R-squared values for the linear mixed-effects
    model based on the Nakagawa & Schielzeth (2013) variance decomposition framework.
    """
    try:
        # Variance calculation of the fixed effect predictions
        fixed_effects_var = np.var(model_result.predict())

        # Random effects variance components tracking
        random_effects_var = float(model_result.cov_re.iloc[0, 0])
        residual_var = model_result.scale
        total_var = fixed_effects_var + random_effects_var + residual_var

        # Proportional calculation mappings
        r2_marginal = fixed_effects_var / total_var
        r2_conditional = (fixed_effects_var + random_effects_var) / total_var

        print(f"\nINFO: {model_name} Pseudo R-Squared Diagnostics")
        print(f"Marginal R2 (Fixed Effects Proportion):     {r2_marginal:.3f}")
        print(f"Conditional R2 (Total Explained Variance):   {r2_conditional:.3f}")

        return r2_marginal, r2_conditional

    except Exception as e:
        print(f"ERROR: Could not compute pseudo R2 variance components: {e}")
        return np.nan, np.nan


def check_social(x):
    """
    Evaluates categorical categorical indicators to generate binary classifications
    for co-located visitor or personal interruption fields.
    """
    return 1 if str(x) in ["Housemate / Family", "Guest", "Pet"] else 0


def run_social_presence_analysis():
    """
    Aggregates longitudinal contextual markers to extract daily social presence indicators,
    merging them with evening metrics to execute control validation mixed regressions.
    """
    print("INFO: Loading processed evening tracking data...")
    try:
        df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
        df_eve.columns = df_eve.columns.str.strip()

        df_eve['ID'] = df_eve['ID'].astype(str)
        df_eve['Date'] = pd.to_datetime(df_eve['Helsinki Time'], utc=True, errors='coerce').dt.date
    except Exception as e:
        print(f"ERROR: Failed to process evening data asset: {e}")
        sys.exit(1)

    print("INFO: Processing longitudinal context files for daily social flag feature extraction...")
    try:
        df_coz = pd.read_csv(COZIE_FILE, usecols=['ID', 'ws_timestamp_start', 'Interruption_Social', 'q6_visitor'],
                             low_memory=False)
        df_coz['ID'] = df_coz['ID'].astype(str)
        df_coz['Date'] = pd.to_datetime(df_coz['ws_timestamp_start'], utc=True, errors='coerce').dt.date

        # Row-wise logical evaluation for co-presence vectors
        df_coz['social_flag_q6'] = df_coz['q6_visitor'].apply(check_social)
        df_coz['social_flag_interrupt'] = df_coz['Interruption_Social'].apply(check_social)
        df_coz['is_social'] = ((df_coz['social_flag_q6'] == 1) | (df_coz['social_flag_interrupt'] == 1)).astype(int)

        # Generating localized maximum value groupings to extract daily participant-level flags
        daily_social = df_coz.groupby(['ID', 'Date'])['is_social'].max().reset_index()
        daily_social.rename(columns={'is_social': 'Social_Presence_Flag'}, inplace=True)
        print(
            f"INFO: Data logic layer identified baseline social presence on {int(daily_social['Social_Presence_Flag'].sum())} tracking days.")
    except Exception as e:
        print(f"ERROR: Failed to extract parameters from environmental tracking frames: {e}")
        sys.exit(1)

    # Database concatenation and downstream alignment
    df_final = pd.merge(df_eve, daily_social, on=['ID', 'Date'], how='left')
    df_final['Social_Presence_Flag'] = df_final['Social_Presence_Flag'].fillna(0)

    target_cols = ['Productivity_Retention_Gap', 'Within_Daily_CO2', 'Social_Presence_Flag']
    for col in target_cols:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    df_model = df_final.dropna(subset=target_cols + ['ID']).copy()
    print(f"INFO: Database pairing complete. Total clean observations for analysis: {len(df_model)}")

    # ---------------------------------------------------------
    # Model 1: Additive Social Control Baseline
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: INDEPENDENT ADDITIVE CONTEXT CONTROLS (CO2 + SOCIAL)")
    print("=" * 60)
    try:
        mdl_add = smf.mixedlm(
            formula="Productivity_Retention_Gap ~ Within_Daily_CO2 + Social_Presence_Flag",
            data=df_model,
            groups=df_model["ID"]
        ).fit(method='lbfgs')

        print(mdl_add.summary().tables[1])
        calculate_pseudo_r2(mdl_add, "Model 1")
    except Exception as e:
        print(f"ERROR: Model 1 parameter estimation failed: {e}")

    # ---------------------------------------------------------
    # Model 2: Interactive Social Control Model
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTICAL LMM SUMMARY: INTERACTIVE CONTEXT CONTROLS (CO2 * SOCIAL)")
    print("=" * 60)
    try:
        mdl_int = smf.mixedlm(
            formula="Productivity_Retention_Gap ~ Within_Daily_CO2 * Social_Presence_Flag",
            data=df_model,
            groups=df_model["ID"]
        ).fit(method='lbfgs')

        print(mdl_int.summary().tables[1])
        calculate_pseudo_r2(mdl_int, "Model 2")

        print("\nSTATISTICAL INFERENCE: EXACT VARIABLE P-VALUES")
        pvals = mdl_int.pvalues
        interaction_term = "Within_Daily_CO2:Social_Presence_Flag"

        print(f"Intercept:                        {pvals['Intercept']:.10f}")
        print(f"Within Daily CO2:                 {pvals['Within_Daily_CO2']:.10f}")
        print(f"Social Presence Flag:             {pvals['Social_Presence_Flag']:.10f}")
        if interaction_term in pvals:
            print(f"CO2 * Social Interaction:         {pvals[interaction_term]:.10f}")
        else:
            print("WARNING: Interaction parameter key mismatch during automated dictionary parsing.")
    except Exception as e:
        print(f"ERROR: Model 2 parameter estimation failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_social_presence_analysis()
# phase_4_physiological_lmm.py
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import warnings
from config import DATA_DIR

warnings.filterwarnings("ignore")

# Path configuration to the master clean dataset
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"

def calculate_pseudo_r2(model_result):
    """
    Calculates Marginal and Conditional R-squared values for the linear mixed-effects
    model based on the Nakagawa & Schielzeth (2013) disaggregation framework.
    """
    # Variance calculation of the fixed effect predictions
    fixed_effects_var = np.var(model_result.predict())

    # Extraction of random effect variance (between-person tau_00)
    random_effects_var = model_result.cov_re.iloc[0, 0]

    # Extraction of residual variance (within-person sigma^2 noise)
    residual_var = model_result.scale

    # Total variance summation across fixed and random components
    total_var = fixed_effects_var + random_effects_var + residual_var

    # Variance proportions derivation
    r2_marginal = fixed_effects_var / total_var
    r2_conditional = (fixed_effects_var + random_effects_var) / total_var

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY: NAKAGAWA PSEUDO R-SQUARED DIAGNOSTICS")
    print("=" * 60)
    print(f"Marginal R2 (Fixed Effects Proportion):     {r2_marginal:.3f}")
    print(f"Conditional R2 (Total Explained Variance):   {r2_conditional:.3f}")
    print("=" * 60)

    return r2_marginal, r2_conditional

def run_phase4_lmm():
    """
    Loads centered longitudinal data, parses and synchronizes UTC/local timestamps,
    applies participant filtering thresholds, and fits the physiological mixed-effects model.
    """
    print("INFO: Loading centered physiological and environmental dataset...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    try:
        # ISO timestamp parsing and chronological standardization
        df['Datetime'] = pd.to_datetime(df['index'])

        # Time-zone transformation to Helsinki local time to preserve alignment with diurnal working hours
        if df['Datetime'].dt.tz is None:
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Helsinki')
        else:
            df['Datetime'] = df['Datetime'].dt.tz_convert('Europe/Helsinki')

        # Fractional day-hour index calculation for time-of-day circadian baseline tracking
        df['Hour_of_Day'] = df['Datetime'].dt.hour + (df['Datetime'].dt.minute / 60.0)
        print("INFO: Chronological parsing complete. Local timezone applied.")
    except Exception as e:
        print(f"ERROR: Timestamp synchronization failure: {e}")
        return

    # List definition of continuous predictors, target variables, and group identifiers
    model_vars = ['HR_Pre_10m_Avg', 'Within_CO2', 'Steps_Pre_10m_Sum', 'Within_Temperature', 'Hour_of_Day', 'ID']
    df_clean = df.dropna(subset=model_vars)

    # Compliance threshold filter: Retention of active participants with 50 or more tracking observations
    valid_counts = df_clean.groupby('ID').size()
    valid_ids = valid_counts[valid_counts >= 50].index
    df_model = df_clean[df_clean['ID'].isin(valid_ids)]

    print(f"INFO: Fitting conditional linear mixed-effects model on {len(df_model)} observations across {len(valid_ids)} participants...")

    # Specification of the Level 1 longitudinal regression formula with concurrent confounders
    formula = "HR_Pre_10m_Avg ~ Within_CO2 + Steps_Pre_10m_Sum + Within_Temperature + Hour_of_Day"

    # Execution and fit parameter estimation of the random-intercept LMM configuration
    model = smf.mixedlm(formula, df_model, groups=df_model["ID"])
    result = model.fit()

    print("\n" + "=" * 60)
    print("STATISTICAL OLS/LMM RECOEFFICIENT SUMMARY")
    print("=" * 60)
    print(result.summary())

    print("\n" + "=" * 60)
    print("STATISTICAL INFERENCE: EXACT VARIABLE P-VALUES")
    print("=" * 60)

    # Systematic iteration and print configuration for floating-point p-values in scientific notation
    exact_p_values = result.pvalues
    for term, p_val in exact_p_values.items():
        print(f"{term:25}: {p_val:.10e}")

    # Execution of the Nakagawa effect-size evaluation algorithm
    calculate_pseudo_r2(result)

if __name__ == "__main__":
    run_phase4_lmm()
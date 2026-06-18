# phase_6_behavioral_interaction_lmm.py
import pandas as pd
import statsmodels.formula.api as smf
import sys
from config import DATA_DIR
import warnings
warnings.filterwarnings("ignore")
# Path configuration to the processed evening survey dataset
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"

def run_behavioral_interaction_models():
    """
    Exploratory Phase 6 Behavioral Pipeline.
    Note: These models evaluate exploratory interaction trends between daily cumulative
    air quality exposures and adaptive actions, serving as secondary validation checks.
    """
    """
    Fits linear mixed-effects models evaluating the impact of cumulative workday CO2 exposure
    and reported ventilation mitigation actions on subjective mood and productivity gaps.
    """
    print("INFO: Loading processed evening dataset for behavioral modeling...")
    try:
        df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
    except Exception as e:
        print(f"ERROR: Failed to load specified data asset: {e}")
        sys.exit(1)

    # Statistical array definition for targeted structural variables
    target_cols = [
        'Mood_Delta',
        'Productivity_Retention_Gap',
        'Within_Daily_CO2',
        'Evening_Reported_Ventilation_Actions'
    ]

    # Enforcing strict continuous numeric float formatting across target arrays
    for col in target_cols:
        df_eve[col] = pd.to_numeric(df_eve[col], errors='coerce')

    # Complete observation filtration for the mood delta LMM pipeline
    df_mood = df_eve.dropna(
        subset=[
            'Mood_Delta',
            'Within_Daily_CO2',
            'Evening_Reported_Ventilation_Actions',
            'ID'
        ]
    )

    # Complete observation filtration for the productivity retention gap LMM pipeline
    df_prod = df_eve.dropna(
        subset=[
            'Productivity_Retention_Gap',
            'Within_Daily_CO2',
            'Evening_Reported_Ventilation_Actions',
            'ID'
        ]
    )

    print(f"INFO: Complete data rows identified for Mood Model: {len(df_mood)} rows across tracking timeline.")
    print(f"INFO: Complete data rows identified for Productivity Model: {len(df_prod)} rows across tracking timeline.")

    # ---------------------------------------------------------
    # Model 1: Mood Delta Evaluation
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STATISTICAL LMM SUMMARY: MOOD DELTA BEHAVIORAL EXPOSURE RESPONSE")
    print("="*60)
    try:
        mdl_mood = smf.mixedlm(
            "Mood_Delta ~ Evening_Reported_Ventilation_Actions + Within_Daily_CO2",
            data=df_mood,
            groups=df_mood["ID"]
        ).fit()
        print(mdl_mood.summary().tables[1])
    except Exception as e:
        print(f"ERROR: Matrix estimation failure on Mood Model: {e}")

    # ---------------------------------------------------------
    # Model 2: Productivity Retention Gap Evaluation
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STATISTICAL LMM SUMMARY: PRODUCTIVITY GAP BEHAVIORAL EXPOSURE RESPONSE")
    print("="*60)
    try:
        mdl_prod = smf.mixedlm(
            "Productivity_Retention_Gap ~ Evening_Reported_Ventilation_Actions + Within_Daily_CO2",
            data=df_prod,
            groups=df_prod["ID"]
        ).fit()
        print(mdl_prod.summary().tables[1])
    except Exception as e:
        print(f"ERROR: Matrix estimation failure on Productivity Model: {e}")
    print("="*60)

if __name__ == "__main__":
    run_behavioral_interaction_models()
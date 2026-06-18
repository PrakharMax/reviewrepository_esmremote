# phase_6_subjective_wellbeing_lmm.py
import pandas as pd
import statsmodels.formula.api as smf
from config import DATA_DIR
import warnings
warnings.filterwarnings("ignore")

# Path configuration to the processed evening survey dataset
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"


def run_phase6_models():
    """
    Executes the definitive Phase 6 linear mixed-effects regressions testing the multi-day
    impact of cumulative CO2 exposure on productivity retention gaps and stress deltas.
    """
    print("INFO: Loading processed evening dataset for RQ  modeling...")
    try:
        df = pd.read_csv(EVENING_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: Specified data asset not found at tracking path: {EVENING_FILE}")
        return

    df['ID'] = df['ID'].astype(str)

    # ---------------------------------------------------------
    #  Productivity Retention Gap Evaluation
    # ---------------------------------------------------------
    model1_vars = ['Productivity_Retention_Gap', 'Within_Daily_CO2', 'Within_Daily_Temperature', 'ID']
    df_m1 = df.dropna(subset=model1_vars)

    print(f"INFO: Fitting linear mixed model for Productivity Gap across {len(df_m1)} observation rows...")
    formula_m1 = "Productivity_Retention_Gap ~ Within_Daily_CO2 + Within_Daily_Temperature"

    try:
        # Enforcing lbfgs optimization to prevent mathematical convergence warnings
        model1 = smf.mixedlm(formula_m1, df_m1, groups=df_m1["ID"])
        result1 = model1.fit(method='lbfgs')
        print("\n" + "=" * 60)
        print("STATISTICAL LMM SUMMARY: (PRODUCTIVITY GAP INTERCEPT)")
        print("=" * 60)
        print(result1.summary())
    except Exception as e:
        print(f"ERROR: Model 1 matrix optimization failed to converge: {e}")

    # ---------------------------------------------------------
    #  Stress Delta Evaluation
    # ---------------------------------------------------------
    model2_vars = ['Stress_Delta', 'Within_Daily_CO2', 'Within_Daily_Temperature', 'ID']
    df_m2 = df.dropna(subset=model2_vars)

    print(f"\nINFO: Fitting linear mixed model for Stress Delta across {len(df_m2)} observation rows...")
    formula_m2 = "Stress_Delta ~ Within_Daily_CO2 + Within_Daily_Temperature"

    try:
        # Enforcing lbfgs optimization to prevent mathematical convergence warnings
        model2 = smf.mixedlm(formula_m2, df_m2, groups=df_m2["ID"])
        result2 = model2.fit(method='lbfgs')
        print("\n" + "=" * 60)
        print("STATISTICAL LMM SUMMARY: (STRESS DELTA INTERCEPT)")
        print("=" * 60)
        print(result2.summary())
    except Exception as e:
        print(f"ERROR: Model 2 matrix optimization failed to converge: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_phase6_models()
# phase_6_morning_wellbeing_lmm.py
import pandas as pd
import statsmodels.formula.api as smf
from config import DATA_DIR
import warnings
warnings.filterwarnings("ignore")

# Path configuration to the processed morning survey dataset
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"

# String configurations representing raw survey column headers
RAW_FOCUS = 'Rate your focus level right now (1–5) Please select the option that best describes your current focus level:'
RAW_STRESS = ' How relaxed or calm do you feel right now? (1–5) Please select the option that best describes your current stress level:'


def clean_likert(series):
    """
    Safely extracts leading numeric digits from raw survey string entries
    and normalizes them to float values for statistical modeling.
    """
    return series.astype(str).str.extract(r'(\d)')[0].astype(float)


def run_morning_models():
    """
    Executes linear mixed-effects regressions testing the multi-day impact of overnight
    environmental parameters on morning cognitive focus and subjective relaxation levels.
    """
    print("INFO: Loading processed morning dataset for RQ validation...")
    try:
        df = pd.read_csv(MORNING_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: Specified data asset not found at tracking path: {MORNING_FILE}")
        return

    # Normalization and variable alignment across outcome and control arrays
    df['Morning_Focus'] = clean_likert(df[RAW_FOCUS])
    df['Morning_Relaxation'] = clean_likert(df[RAW_STRESS])
    df['Good_Sleep'] = df['Control_Good_Sleep']
    df['ID'] = df['ID'].astype(str)

    # ---------------------------------------------------------
    #  Morning Focus Level Evaluation
    # ---------------------------------------------------------
    model3_vars = ['Morning_Focus', 'Within_CO2', 'Good_Sleep', 'Within_Temperature', 'ID']
    df_m3 = df.dropna(subset=model3_vars)

    print(f"INFO: Fitting linear mixed model for Morning Focus across {len(df_m3)} observation rows...")
    formula_m3 = "Morning_Focus ~ Within_CO2 + Good_Sleep + Within_Temperature"

    try:
        # Enforcing lbfgs optimization to guarantee matrix convergence stability
        model3 = smf.mixedlm(formula_m3, df_m3, groups=df_m3["ID"])
        result3 = model3.fit(method='lbfgs')
        print("\n" + "=" * 60)
        print("STATISTICAL LMM SUMMARY: (MORNING FOCUS INTERCEPT)")
        print("=" * 60)
        print(result3.summary())
    except Exception as e:
        print(f"ERROR: Model 3 matrix optimization failed to converge: {e}")

    # ---------------------------------------------------------
    # Morning Relaxation Level Evaluation
    # ---------------------------------------------------------
    model4_vars = ['Morning_Relaxation', 'Within_CO2', 'Good_Sleep', 'Within_Temperature', 'ID']
    df_m4 = df.dropna(subset=model4_vars)

    print(f"\nINFO: Fitting linear mixed model for Morning Relaxation across {len(df_m4)} observation rows...")
    formula_m4 = "Morning_Relaxation ~ Within_CO2 + Good_Sleep + Within_Temperature"

    try:
        # Enforcing lbfgs optimization to guarantee matrix convergence stability
        model4 = smf.mixedlm(formula_m4, df_m4, groups=df_m4["ID"])
        result4 = model4.fit(method='lbfgs')
        print("\n" + "=" * 60)
        print("STATISTICAL LMM SUMMARY: (MORNING RELAXATION INTERCEPT)")
        print("=" * 60)
        print(result4.summary())
    except Exception as e:
        print(f"ERROR: Model 4 matrix optimization failed to converge: {e}")
    print("=" * 60)


if __name__ == "__main__":
    run_morning_models()
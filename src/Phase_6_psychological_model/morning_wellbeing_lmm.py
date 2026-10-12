import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")  # Suppress convergence warnings
from config import DATA_DIR

# --- CONFIGURATION ---
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"

# The exact Webropol column names
RAW_FOCUS = 'Rate your focus level right now (1–5) Please select the option that best describes your current focus level:'
RAW_STRESS = ' How relaxed or calm do you feel right now? (1–5) Please select the option that best describes your current stress level:'


def clean_likert(series):
    """Extracts the leading number from the text string"""
    return series.astype(str).str.extract(r'(\d)')[0].astype(float)


def run_morning_models():
    print("🌅 Loading Morning Survey Data for Phase 6 Models...")
    try:
        df = pd.read_csv(MORNING_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"❌ File not found at {MORNING_FILE}")
        return

    # 1. Clean and rename the target variables for statsmodels
    df['Morning_Focus'] = clean_likert(df[RAW_FOCUS])
    df['Morning_Relaxation'] = clean_likert(df[RAW_STRESS])  # Higher score = More relaxed
    df['Good_Sleep'] = df['Control_Good_Sleep']

    # Ensure ID is treated as a categorical string
    df['ID'] = df['ID'].astype(str)

    # ==========================================
    # 🧠 MODEL 3: HYPOTHESIS 4 (MORNING FOCUS)
    # ==========================================
    model3_vars = ['Morning_Focus', 'Within_CO2', 'Good_Sleep', 'Within_Temperature', 'ID']
    df_m3 = df.dropna(subset=model3_vars)

    print(f"\n📊 Running LMM for Morning Focus on {len(df_m3)} mornings...")
    formula_m3 = "Morning_Focus ~ Within_CO2 + Good_Sleep + Within_Temperature"

    try:
        model3 = smf.mixedlm(formula_m3, df_m3, groups=df_m3["ID"])
        result3 = model3.fit()
        print("\n" + "=" * 60)
        print("🏆 PHASE 6: HYPOTHESIS 4 (MORNING FOCUS)")
        print("=" * 60)
        print(result3.summary())
    except Exception as e:
        print(f"Model 3 failed to converge: {e}")

    # ==========================================
    # 🧘 MODEL 4: HYPOTHESIS 3 (MORNING RELAXATION)
    # ==========================================
    model4_vars = ['Morning_Relaxation', 'Within_CO2', 'Good_Sleep', 'Within_Temperature', 'ID']
    df_m4 = df.dropna(subset=model4_vars)

    print(f"\n📊 Running LMM for Morning Relaxation on {len(df_m4)} mornings...")
    formula_m4 = "Morning_Relaxation ~ Within_CO2 + Good_Sleep + Within_Temperature"

    try:
        model4 = smf.mixedlm(formula_m4, df_m4, groups=df_m4["ID"])
        result4 = model4.fit()
        print("\n" + "=" * 60)
        print("🏆 PHASE 6: HYPOTHESIS 3 (MORNING RELAXATION)")
        print("=" * 60)
        print(result4.summary())
    except Exception as e:
        print(f"Model 4 failed to converge: {e}")


if __name__ == "__main__":
    run_morning_models()
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
from config import DATA_DIR

# --- CONFIGURATION ---
EVENING_FILE =  DATA_DIR / "Master_Evening_Centered.csv"
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"


def run_phase6_models():
    print("🚀 Loading Survey Data for Phase 6 Models (Complete-Case Analysis)...")
    try:
        df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
        df_morn = pd.read_csv(MORNING_FILE, low_memory=False)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return

    df_eve['ID'] = df_eve['ID'].astype(str)
    df_morn['ID'] = df_morn['ID'].astype(str)

    # --- COMPLETE-CASE FILTER (Reviewer 1, Point 6) ---
    time_col_eve = 'Helsinki Time' if 'Helsinki Time' in df_eve.columns else 'Timestamp'
    time_col_morn = 'Helsinki Time' if 'Helsinki Time' in df_morn.columns else 'Timestamp'

    df_eve['Date'] = pd.to_datetime(df_eve[time_col_eve], errors='coerce', utc=True).dt.tz_convert(
        "Europe/Helsinki").dt.date
    df_morn['Date'] = pd.to_datetime(df_morn[time_col_morn], errors='coerce', utc=True).dt.tz_convert(
        "Europe/Helsinki").dt.date

    valid_morning_days = df_morn[['ID', 'Date']].dropna().drop_duplicates()
    df_complete = pd.merge(df_eve, valid_morning_days, on=['ID', 'Date'], how='inner')

    print(f"🧹 Complete-Case Filter applied: Reduced from {len(df_eve)} to {len(df_complete)} strictly matched days.")

    # ==========================================
    # 🧠 MODEL 1: EVENING PRODUCTIVITY (Reviewer 2 Fix)
    # ==========================================

    # 1. Dynamically find the raw Webropol question column
    prod_col_raw = [col for col in df_complete.columns if 'overall productivity level' in col.lower()][0]

    # 2. Extract the numeric score (1-5) from the text response and create our target variable
    df_complete['Evening_Productivity'] = df_complete[prod_col_raw].astype(str).str.extract(r'(\d)').astype(float)

    model1_vars = ['Evening_Productivity', 'Within_Daily_CO2', 'Within_Daily_Temperature', 'ID']
    df_m1 = df_complete.dropna(subset=model1_vars)

    print(f"\n📊 Running LMM for Evening_Productivity on {len(df_m1)} complete-case workdays...")
    formula_m1 = "Evening_Productivity ~ Within_Daily_CO2 + Within_Daily_Temperature"

    try:
        model1 = smf.mixedlm(formula_m1, df_m1, groups=df_m1["ID"])
        result1 = model1.fit()
        print("\n" + "=" * 60)
        print("🏆 PHASE 6: HYPOTHESIS 1 (EVENING PRODUCTIVITY) - COMPLETE CASE")
        print("=" * 60)
        print(result1.summary())

        print("\n🔬 EXACT P-VALUES & CONFIDENCE INTERVALS FOR TABLES")
        for term in result1.params.index:
            coef = result1.params[term]
            se = result1.bse[term]
            pval = result1.pvalues[term]
            ci_low, ci_high = result1.conf_int().loc[term]
            print(f"{term:25}: Coef={coef:.4f}, SE={se:.4f}, p={pval:.4e}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]")
    except Exception as e:
        print(f"Model 1 failed to converge: {e}")

    # ==========================================
    # 😰 MODEL 2: STRESS DELTA
    # ==========================================
    model2_vars = ['Stress_Delta', 'Within_Daily_CO2', 'Within_Daily_Temperature', 'ID']
    df_m2 = df_complete.dropna(subset=model2_vars)

    print(f"\n📊 Running LMM for Stress Delta on {len(df_m2)} complete-case workdays...")
    formula_m2 = "Stress_Delta ~ Within_Daily_CO2 + Within_Daily_Temperature"

    try:
        model2 = smf.mixedlm(formula_m2, df_m2, groups=df_m2["ID"])
        result2 = model2.fit()
        print("\n" + "=" * 60)
        print("🏆 PHASE 6: HYPOTHESIS 2 (STRESS DELTA) - COMPLETE CASE")
        print("=" * 60)
        print(result2.summary())

        print("\n🔬 EXACT P-VALUES & CONFIDENCE INTERVALS FOR TABLES")
        for term in result2.params.index:
            coef = result2.params[term]
            se = result2.bse[term]
            pval = result2.pvalues[term]
            ci_low, ci_high = result2.conf_int().loc[term]
            print(f"{term:25}: Coef={coef:.4f}, SE={se:.4f}, p={pval:.4e}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]")
    except Exception as e:
        print(f"Model 2 failed to converge: {e}")


if __name__ == "__main__":
    run_phase6_models()
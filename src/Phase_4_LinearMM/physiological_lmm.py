import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from pathlib import Path
import warnings
from config import DATA_DIR

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def run_phase4_lmm():
    print("🚀 Loading Centered Cozie Data for Phase 4...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    try:
        df['Datetime'] = pd.to_datetime(df['index'])
        if df['Datetime'].dt.tz is None:
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Helsinki')
        else:
            df['Datetime'] = df['Datetime'].dt.tz_convert('Europe/Helsinki')

        df['Hour_of_Day'] = df['Datetime'].dt.hour + (df['Datetime'].dt.minute / 60.0)
        print("✅ Time successfully parsed and converted to Helsinki local time.")
    except Exception as e:
        print(f"⚠️ Error parsing time: {e}")
        return

    # =========================================================================
    # 🛠️ CRITICAL FIX (ISSUE 1): Calculate Between-Person Means for Mundlak
    # =========================================================================
    # Note: Using raw columns if present, otherwise reconstructing from Within + Between
    # If your CSV already has 'Between_CO2' and 'Between_Temperature', this ensures they are populated.
    if 'Between_CO2' not in df.columns and 'CO2_ppm' in df.columns:
        df['Between_CO2'] = df.groupby('ID')['CO2_ppm'].transform('mean')
    elif 'Between_CO2' not in df.columns:
        # Fallback: if raw column is missing, Between_CO2 should already be in your level 4 CSV
        print("⚠️ Warning: Could not find raw CO2_ppm to compute Between_CO2. Please check CSV columns.")

    if 'Between_Temperature' not in df.columns and 'Temperature_C' in df.columns:
        df['Between_Temperature'] = df.groupby('ID')['Temperature_C'].transform('mean')
    elif 'Between_Temperature' not in df.columns:
        print("⚠️ Warning: Could not find raw Temperature_C to compute Between_Temperature. Please check CSV columns.")

    # 1. Clean data: Drop rows with missing values (now including Between terms)
    model_vars = [
        'HR_Pre_10m_Avg', 'Within_CO2', 'Between_CO2',
        'Steps_Pre_10m_Sum', 'Within_Temperature', 'Between_Temperature',
        'Hour_of_Day', 'ID'
    ]

    # Check that all required columns actually exist before dropping NA
    available_vars = [v for v in model_vars if v in df.columns]
    df_clean = df.dropna(subset=available_vars)

    # 2. Apply Minimum Viable Data filter (>= 50 rows)
    valid_counts = df_clean.groupby('ID').size()
    valid_ids = valid_counts[valid_counts >= 50].index
    df_model = df_clean[df_clean['ID'].isin(valid_ids)]

    # 2.5 CRITICAL FIX: Sort the data so Participant groups are contiguous!
    df_model = df_model.sort_values(by=['ID', 'Datetime']).reset_index(drop=True)

    print(f"📊 Running GEE EXCHANGEABLE Model on {len(df_model)} observations across {len(valid_ids)} participants...")

    # 3. The Updated Formula (Full Mundlak Specification)
    formula = "HR_Pre_10m_Avg ~ Within_CO2 + Between_CO2 + Steps_Pre_10m_Sum + Within_Temperature + Between_Temperature + Hour_of_Day"

    # 4. Fit the model using Generalized Estimating Equations (GEE) with Exchangeable
    exch_cov_struct = sm.cov_struct.Exchangeable()

    model = smf.gee(
        formula,
        groups=df_model["ID"],
        data=df_model,
        cov_struct=exch_cov_struct
    )
    result = model.fit()

    # 5. Output the results
    print("\n" + "=" * 60)
    print("🏆 PHASE 4: THE EXCHANGEABLE ADJUSTED MODEL RESULTS (FULL MUNDLAK)")
    print("=" * 60)
    print(result.summary())

    # 6. Extract exact P-values
    print("\n" + "=" * 60)
    print("🔬 EXACT P-VALUES")
    print("=" * 60)

    exact_p_values = result.pvalues
    for term, p_val in exact_p_values.items():
        print(f"{term:25}: {p_val:.10e}")

    # =========================================================================
    # 🚀 SENSITIVITY ANALYSIS (Reviewer 1, Points 4 & 5)
    # Non-Overlapping 30-Minute Aggregation
    # =========================================================================
    print("\n" + "🚀" * 30)
    print("SENSITIVITY ANALYSIS: NON-OVERLAPPING 30-MIN BLOCKS")
    print("Addressing Reviewer 1 (Points 4 & 5) - Removing sliding window overlap")
    print("🚀" * 30)

    # Group by ID and strict 30-minute non-overlapping chronological chunks
    df_non_overlap = df_model.groupby([
        'ID',
        pd.Grouper(key='Datetime', freq='30min')
    ]).mean(numeric_only=True).reset_index()

    # Clean any NaNs created during the strict resampling grid
    df_non_overlap = df_non_overlap.dropna(
        subset=['HR_Pre_10m_Avg', 'Within_CO2', 'Steps_Pre_10m_Sum', 'Within_Temperature'])

    print(
        f"🧹 Aggregation applied: Reduced from {len(df_model)} overlapping observations to {len(df_non_overlap)} strict, non-overlapping blocks.")

    # Fit a Mixed Linear Model (LMM) on the non-overlapping data, which handles small cluster N (19) reliably
    try:
        sens_model = smf.mixedlm(
            formula,
            df_non_overlap,
            groups=df_non_overlap["ID"]
        )
        sens_result = sens_model.fit()
        print("\n" + "=" * 60)
        print("STATISTICAL SUMMARY: SENSITIVITY LMM (NON-OVERLAPPING)")
        print("=" * 60)
        print(sens_result.summary())

        print("\nExact P-Values (Sensitivity Model):")
        for term, p_val in sens_result.pvalues.items():
            print(f"{term:25}: {p_val:.10e}")
    except Exception as e:
        print(f"Sensitivity model failed to converge: {e}")
    # =========================================================================
    # 🛡️ LEAVE-ONE-OUT (LOO) ROBUSTNESS CHECK ON N=19
    # =========================================================================
    print("\n" + "🛡️" * 30)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (TESTING CLUSTER INFLUENCE)")
    print("🛡️" * 30)

    unique_ids = df_non_overlap['ID'].unique()
    loo_results = []

    for drop_id in unique_ids:
        # Create dataset excluding one participant
        df_loo = df_non_overlap[df_non_overlap['ID'] != drop_id]

        try:
            loo_model = smf.mixedlm(formula, df_loo, groups=df_loo["ID"])
            loo_fit = loo_model.fit(disp=False)
            coef = loo_fit.params['Within_CO2']
            pval = loo_fit.pvalues['Within_CO2']
            loo_results.append((drop_id, coef, pval))
        except Exception:
            pass

    print(f"{'Excluded ID':<15} | {'Within_CO2 Coef':<15} | {'P-Value'}")
    print("-" * 50)
    for res in loo_results:
        print(f"{res[0]:<15} | {res[1]:<15.4f} | {res[2]:.4e}")

if __name__ == "__main__":
    run_phase4_lmm()
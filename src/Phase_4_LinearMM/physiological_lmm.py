import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import warnings
from config import DATA_DIR

# We suppress matrix convergence warnings during recursive looping operations
warnings.filterwarnings("ignore")

# We define the path configuration to our master clean dataset
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def run_phase4_lmm():
    """
    We load our centered longitudinal data, parse and synchronize UTC/local timestamps,
    apply our full Mundlak disaggregation specification, and fit the physiological
    Generalized Estimating Equations (GEE) model.
    """
    print("INFO: Loading our centered physiological and environmental dataset...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    try:
        # We parse the ISO timestamps for chronological standardization
        df['Datetime'] = pd.to_datetime(df['index'])

        # We transform the timezone to Helsinki local time to preserve biological
        # alignment with the occupants' diurnal working hours.
        if df['Datetime'].dt.tz is None:
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Helsinki')
        else:
            df['Datetime'] = df['Datetime'].dt.tz_convert('Europe/Helsinki')

        # We calculate a fractional day-hour index for time-of-day circadian baseline tracking
        df['Hour_of_Day'] = df['Datetime'].dt.hour + (df['Datetime'].dt.minute / 60.0)
        print("INFO: Chronological parsing complete. Local timezone successfully applied.")
    except Exception as e:
        print(f"ERROR: Timestamp synchronization failure: {e}")
        return

    # We dynamically calculate the Between-Person means for the Mundlak specification
    # if they are not already present in the source matrix.
    if 'Between_CO2' not in df.columns and 'CO2_ppm' in df.columns:
        df['Between_CO2'] = df.groupby('ID')['CO2_ppm'].transform('mean')
    elif 'Between_CO2' not in df.columns:
        print("WARNING: Could not find raw CO2_ppm to compute Between_CO2. Please check CSV columns.")

    if 'Between_Temperature' not in df.columns and 'Temperature_C' in df.columns:
        df['Between_Temperature'] = df.groupby('ID')['Temperature_C'].transform('mean')
    elif 'Between_Temperature' not in df.columns:
        print("WARNING: Could not find raw Temperature_C to compute Between_Temperature. Please check CSV columns.")

    # We define our continuous predictors, target variables, and group identifiers
    model_vars = [
        'HR_Pre_10m_Avg', 'Within_CO2', 'Between_CO2',
        'Steps_Pre_10m_Sum', 'Within_Temperature', 'Between_Temperature',
        'Hour_of_Day', 'ID'
    ]

    # We ensure all required columns exist before filtering to prevent key errors
    available_vars = [v for v in model_vars if v in df.columns]
    df_clean = df.dropna(subset=available_vars)

    # We apply our compliance threshold filter, retaining only active participants
    # with 50 or more tracking observations.
    valid_counts = df_clean.groupby('ID').size()
    valid_ids = valid_counts[valid_counts >= 50].index
    df_model = df_clean[df_clean['ID'].isin(valid_ids)]

    # CRITICAL: We strictly sort the data chronologically by ID to ensure contiguous
    # grouping, which is mathematically required for accurate GEE execution.
    df_model = df_model.sort_values(by=['ID', 'Datetime']).reset_index(drop=True)

    print(
        f"INFO: Fitting GEE EXCHANGEABLE model on {len(df_model)} observations across {len(valid_ids)} participants...")

    # We specify our fully disaggregated longitudinal regression formula
    formula = "HR_Pre_10m_Avg ~ Within_CO2 + Between_CO2 + Steps_Pre_10m_Sum + Within_Temperature + Between_Temperature + Hour_of_Day"

    # We fit the model using Generalized Estimating Equations (GEE) with an Exchangeable covariance structure
    exch_cov_struct = sm.cov_struct.Exchangeable()

    model = smf.gee(
        formula,
        groups=df_model["ID"],
        data=df_model,
        cov_struct=exch_cov_struct
    )
    result = model.fit()

    # We print the core statistical outputs to the terminal for review
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY: THE EXCHANGEABLE ADJUSTED MODEL (FULL MUNDLAK)")
    print("=" * 60)
    print(result.summary())

    print("\n" + "=" * 60)
    print("STATISTICAL INFERENCE: EXACT VARIABLE P-VALUES")
    print("=" * 60)

    # We systematically iterate through the floating-point p-values for manuscript reporting
    exact_p_values = result.pvalues
    for term, p_val in exact_p_values.items():
        print(f"{term:25}: {p_val:.10e}")


if __name__ == "__main__":
    run_phase4_lmm()
# phase_4_lagged_sedation_test.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from config import DATA_DIR

# Path configuration to the master clean dataset
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def run_lagged_sedation_test():
    """
    Executes a cluster-robust regression utilizing a 30-minute lagging exposure window
    to isolate and verify delayed cardiovascular sedation effects across the cohort.
    """
    start_time = time.time()
    print("INFO: Loading centered physiological and environmental dataset...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    print("INFO: Sorting data and parsing chronological sequences...")
    # ISO timestamp parsing and immediate normalization to Helsinki local time for biological accuracy
    df['Timestamp_DT'] = pd.to_datetime(df['index'], errors='coerce', utc=True)
    df['Timestamp_DT'] = df['Timestamp_DT'].dt.tz_convert('Europe/Helsinki')

    df = df.dropna(subset=['Timestamp_DT', 'HR_Pre_10m_Avg', 'Within_CO2'])
    df = df.sort_values(['ID', 'Timestamp_DT'])

    print("INFO: Building 30-minute environmental exposure lag buffers...")
    # Shifting by 6 rows aligns with the 5-minute environmental sampling frequency (6 * 5 = 30 minutes)
    df['CO2_Lag30m'] = df.groupby('ID')['Within_CO2'].shift(6)

    # Calculation of diurnal time indicator centered around the typical workday midpoint
    df['Hour'] = df['Timestamp_DT'].dt.hour + df['Timestamp_DT'].dt.minute / 60
    df['Hour_Centered'] = df['Hour'] - 13

    # Dropping records where historical lagging parameters could not be computed
    df_model = df.dropna(subset=['CO2_Lag30m', 'HR_Pre_10m_Avg'])

    print(f"INFO: Data preparation complete. Total observations available: {len(df_model)}")

    print("\nINFO: Executing cluster-robust ordinary least squares regression...")

    # Predictor matrix assembly incorporating the active steps count, temperature, and time covariates
    X = sm.add_constant(df_model[['CO2_Lag30m', 'Steps_Pre_10m_Sum', 'Within_Temperature', 'Hour_Centered']])
    y = df_model['HR_Pre_10m_Avg']

    try:
        # Standard errors are clustered on individual participant IDs to control for nested correlation structures
        result = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_model['ID']})

        print("\n" + "=" * 60)
        print("STATISTICAL OLS SUMMARY: 30-MINUTE LAGGED SEDATION PROFILE")
        print("=" * 60)
        print(result.summary())

        print("\nSTATISTICAL INFERENCE: EXACT VARIABLE P-VALUES")
        for variable, p_value in result.pvalues.items():
            print(f"{variable:<20} | Scientific notation: {p_value:.5e}")

        co2_lag_coef = result.params['CO2_Lag30m']
        co2_lag_p = result.pvalues['CO2_Lag30m']

        print("\n" + "=" * 60)
        print("PIPELINE ANALYSIS AND INFERENCE VALUE RECOGNITION")
        print("=" * 60)
        if co2_lag_coef < 0 and co2_lag_p < 0.05:
            print(f"ANALYSIS: The 30-minute lagged CO2 parameter is significantly negative (p={co2_lag_p:.5e}).")
            print("ANALYSIS: Statistical trend lines support a delayed physiological sedation response.")
        elif co2_lag_coef > 0:
            print(f"ANALYSIS: The 30-minute lagged environmental parameter is positive (Beta={co2_lag_coef:.5f}).")
            print("ANALYSIS: Statistical trend lines indicate alternative physiological activation patterns.")
        else:
            print("ANALYSIS: Indeterminate or statistically non-significant delayed relationship encountered.")

        print(f"\nINFO: Pipeline block execution finalized in {time.time() - start_time:.2f} seconds.")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Robust model processing optimization failed: {e}")


if __name__ == "__main__":
    run_lagged_sedation_test()
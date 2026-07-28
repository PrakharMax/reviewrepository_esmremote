import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from config import DATA_DIR

# We define the path to our master clean centered dataset
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def run_lagged_sedation_test():
    """
    We execute a cluster-robust regression utilizing a 30-minute lagging exposure window
    to isolate and verify delayed cardiovascular sedation effects across our cohort.
    """
    start_time = time.time()
    print("INFO: Loading our centered physiological and environmental dataset...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    print("INFO: Sorting data and parsing chronological sequences...")

    # We parse ISO timestamps and immediately normalize them to Helsinki local time
    # to ensure biological accuracy in our diurnal covariates.
    df['Timestamp_DT'] = pd.to_datetime(df['index'], errors='coerce', utc=True)
    df['Timestamp_DT'] = df['Timestamp_DT'].dt.tz_convert('Europe/Helsinki')

    df = df.dropna(subset=['Timestamp_DT', 'HR_Pre_10m_Avg', 'Within_CO2'])
    df = df.sort_values(['ID', 'Timestamp_DT'])

    print("INFO: Building 30-minute environmental exposure lag buffers...")

    # We shift by 6 rows to perfectly align with our 5-minute environmental
    # sampling frequency (6 * 5 = 30 minutes).
    df['CO2_Lag30m'] = df.groupby('ID')['Within_CO2'].shift(6)

    # We calculate our diurnal time indicator, centered around the typical workday midpoint (13:00)
    df['Hour'] = df['Timestamp_DT'].dt.hour + df['Timestamp_DT'].dt.minute / 60
    df['Hour_Centered'] = df['Hour'] - 13

    # We drop records where historical lagging parameters could not be computed,
    # and we explicitly include all model covariates to prevent statsmodels length mismatch crashes.
    df_model = df.dropna(subset=[
        'CO2_Lag30m',
        'HR_Pre_10m_Avg',
        'Steps_Pre_10m_Sum',
        'Within_Temperature',
        'Hour_Centered'
    ])

    print(f"INFO: Data preparation complete. Total observations available for our model: {len(df_model)}")

    print("\nINFO: Executing cluster-robust ordinary least squares regression...")

    # We assemble the predictor matrix, incorporating our active steps count, temperature, and time covariates
    X = sm.add_constant(df_model[['CO2_Lag30m', 'Steps_Pre_10m_Sum', 'Within_Temperature', 'Hour_Centered']])
    y = df_model['HR_Pre_10m_Avg']

    try:
        # We cluster the standard errors on individual participant IDs to safely control
        # for nested correlation structures in our longitudinal data.
        result = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_model['ID']})

        # We output the statistical summary to the terminal for the reviewer
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

        # We parse the exact p-values to provide clear, automated inference for the narrative
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
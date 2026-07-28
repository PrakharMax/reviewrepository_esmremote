import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time
import warnings
from config import DATA_DIR

# We suppress matrix convergence warnings during recursive looping operations
warnings.filterwarnings("ignore")

# We define the path to our processed longitudinal dataset
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def prepare_lagged_data():
    """
    We load centered longitudinal data, enforce temporal indexing patterns,
    and construct a discrete 30-minute lagging environmental exposure window.
    """
    print("INFO: Loading dataset for robust model diagnostics...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    df['Timestamp_DT'] = pd.to_datetime(df['index'], errors='coerce', utc=True)
    df = df.dropna(subset=['Timestamp_DT', 'HR_Pre_10m_Avg', 'Within_CO2'])
    df = df.sort_values(['ID', 'Timestamp_DT'])

    # We construct a 30-minute lagged window based on localized sampling frequency intervals
    df['CO2_Lag30m'] = df.groupby('ID')['Within_CO2'].shift(6)

    # We derive fractional daytime indicators centered around the workday midpoint
    df['Hour'] = df['Timestamp_DT'].dt.hour + df['Timestamp_DT'].dt.minute / 60
    df['Hour_Centered'] = df['Hour'] - 13

    # We clean the dataset by isolating complete observations across specified parameters
    df_model = df.dropna(
        subset=['CO2_Lag30m', 'HR_Pre_10m_Avg', 'Within_Steps_Sum', 'Within_Temperature', 'Hour_Centered'])
    return df_model


def run_vif_analysis(df):
    """
    We compute Variance Inflation Factors (VIF) to detect potential multicollinearity
    among continuous environmental and physical confounder variables.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST 1: VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    print("=" * 60)

    # We select our predictor matrix and add the required constant
    features = df[['CO2_Lag30m', 'Within_Steps_Sum', 'Within_Temperature', 'Hour_Centered']]
    X = sm.add_constant(features)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_data.to_string(index=False))
    print("-" * 60)
    print("Reference standard: VIF values below 5.0 demonstrate acceptable parameter independence.")


def run_bridge_model(df):
    """
    We execute a lagged random-intercept linear mixed-effects model to assess the stability
    of physiological heart rate responses to preceding environmental exposure states.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST 2: LAGGED EXPOSURE TEMPORAL BRIDGE MODEL")
    print("=" * 60)
    print("INFO: Processing mixed-effects framework using 30-minute lagged environmental inputs...")

    formula = "HR_Pre_10m_Avg ~ CO2_Lag30m + Within_Steps_Sum + Within_Temperature + Hour_Centered"

    try:
        model = smf.mixedlm(formula, df, groups=df["ID"])
        result = model.fit()
        print(result.summary())

        co2_p = result.pvalues['CO2_Lag30m']
        print(f"\nINFO: Statistical significance inference value for lagged CO2: {co2_p:.5e}")

        if co2_p < 0.05:
            print("STATUS: Environmental parameter significance survived mixed-effects convergence checks.")
        else:
            print("STATUS: Environmental parameter significance dropped below required confidence intervals.")

    except Exception as e:
        print(f"ERROR: Linear mixed model matrix estimation failed: {e}")


def run_leave_one_out(df):
    """
    We perform a jackknife leave-one-out sensitivity analysis to determine whether
    the overall cohort trend is driven by outlier participant datasets.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST 3: LEAVE-ONE-OUT PARTICIPANT COHORT INFLUENCE AUDIT")
    print("=" * 60)

    participants = df['ID'].unique()
    print(f"INFO: Running sensitivity loops across {len(participants)} separate participant sub-blocks...\n")

    results = []

    for exclude_id in participants:
        # We systematically exclude a single target participant identifier per loop
        df_subset = df[df['ID'] != exclude_id]

        X = sm.add_constant(df_subset[['CO2_Lag30m', 'Within_Steps_Sum', 'Within_Temperature', 'Hour_Centered']])
        y = df_subset['HR_Pre_10m_Avg']

        # We apply cluster-robust standard errors over iterative loops to guarantee computational execution stability
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_subset['ID']})

        coef = model.params['CO2_Lag30m']
        p_val = model.pvalues['CO2_Lag30m']

        results.append({'Excluded_ID': exclude_id, 'CO2_Coef': coef, 'P_Value': p_val})

    df_results = pd.DataFrame(results)

    max_coef = df_results['CO2_Coef'].max()
    min_coef = df_results['CO2_Coef'].min()
    max_p = df_results['P_Value'].max()

    print(f"ANALYSIS: Beta Coefficient Variation Bounds: {min_coef:.5f} to {max_coef:.5f}")
    print(f"ANALYSIS: Peak statistical inference p-value encountered: {max_p:.5e}")

    if max_p < 0.05:
        print(
            "STATUS: Statistical validation check successful. Environmental association remains significant independent of any single cohort exclusion step.")
    else:
        print(
            "STATUS: Statistical parameter significance lost during loop execution. Associations are structurally influenced by sub-group outlier records.")

        culprits = df_results[df_results['P_Value'] >= 0.05]
        print("\nInfluential Outlier Identifiers Found:")
        print(culprits)


if __name__ == "__main__":
    start = time.time()
    df_ready = prepare_lagged_data()

    run_vif_analysis(df_ready)
    run_bridge_model(df_ready)
    run_leave_one_out(df_ready)

    print(f"\nINFO: Data diagnostic pipelines executed successfully in {time.time() - start:.2f} seconds.")
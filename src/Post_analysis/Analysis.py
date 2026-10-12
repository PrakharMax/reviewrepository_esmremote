import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# We suppress minor warnings for clean terminal output
warnings.filterwarnings("ignore")

# Set standardized publication plot typography
plt.rcParams['font.family'] = 'Arial'

# --- 1. PATHS ---
FILE_PATH = DATA_DIR / "Master_Cozie_Centered.csv"


def run_diagnostics():
    """
    We execute a fast OLS proxy model to generate marginal effect predictions
    and temporal residual diagnostics (ACF/PACF) to validate model assumptions.
    """
    print("INFO: Initializing temporal diagnostics and marginal effects pipeline...")

    # --- 2. LOAD & PREP ---
    try:
        df = pd.read_csv(FILE_PATH, low_memory=False)
    except Exception as e:
        print(f"ERROR: Failed to load specified data asset: {e}")
        sys.exit(1)

    # Recreate Hour_Centered
    if 'index' in df.columns:
        df['timestamp'] = pd.to_datetime(df['index'], errors='coerce')
    else:
        df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    df['Hour_Centered'] = df['hour'] - df['hour'].mean()

    predictors = ['Within_CO2', 'Within_Temperature', 'Within_Steps_Sum', 'Hour_Centered']
    dependent = 'HR_Pre_10m_Avg'

    # Ensure all required columns exist before dropping NAs
    required_cols = ['ID', dependent] + predictors
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in dataset: {missing_cols}")
        sys.exit(1)

    df_diag = df[required_cols].dropna().copy()

    # --- 3. FAST OLS MODEL (The Proxy) ---
    print("INFO: Fitting Fast OLS Model proxy for residual extraction...")
    X = sm.add_constant(df_diag[predictors])
    ols_model = sm.OLS(df_diag[dependent], X).fit()
    df_diag['residuals'] = ols_model.resid

    # --- 4. DIAGNOSTIC PLOTS ---
    print("INFO: Generating statistical diagnostic plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

    # A: Marginal Effects
    # We use the established Coef: -0.005 and Intercept: 75.386 from our preliminary analysis
    co2_range = np.linspace(df_diag['Within_CO2'].min(), df_diag['Within_CO2'].max(), 100)
    hr_pred = 75.386 + (-0.005) * co2_range

    axes[0].plot(co2_range, hr_pred, color='green', lw=3)
    axes[0].set_title('Predicted HR vs. $CO_2$ Deviation', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Within-Person $CO_2$ Deviation (ppm)', fontsize=12)
    axes[0].set_ylabel('Heart Rate (BPM)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # B: ACF
    plot_acf(df_diag['residuals'], ax=axes[1], lags=50, alpha=0.05)
    axes[1].set_title('Residual ACF (Temporal Check)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lags', fontsize=12)

    # C: PACF
    plot_pacf(df_diag['residuals'], ax=axes[2], lags=50, alpha=0.05)
    axes[2].set_title('Residual PACF', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Lags', fontsize=12)

    plt.tight_layout()

    # Save output to the unified images directory
    plot_output = OUTPUT_IMAGE_DIR / "Fast_Diagnostics.png"
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"INFO: Diagnostic plots saved successfully to: {plot_output.name}")


if __name__ == "__main__":
    run_diagnostics()
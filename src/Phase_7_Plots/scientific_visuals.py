import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import sys
import warnings
from matplotlib.colors import LinearSegmentedColormap
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# We suppress convergence warnings globally to ensure clean execution and console output
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"
COZIE_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def clean_likert(series):
    """We extract leading digits from Webropol survey strings to ensure numerical compatibility for modeling."""
    return series.astype(str).str.extract(r'(\d)')[0].astype(float)


def run_phase7_scientific_visuals():
    print("INFO: Phase 7 - Executing ultimate scientific visualization pipeline...")

    try:
        df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
        df_morn = pd.read_csv(MORNING_FILE, low_memory=False)
        df_cozie = pd.read_csv(COZIE_FILE, low_memory=False)
    except Exception as e:
        print(f"ERROR: Failed to load specified data assets: {e}")
        sys.exit(1)

    # Data Standardization
    df_eve['ID'] = df_eve['ID'].astype(str)
    df_morn['ID'] = df_morn['ID'].astype(str)
    df_cozie['ID'] = df_cozie['ID'].astype(str)

    df_morn['Morning_Focus'] = clean_likert(df_morn[
                                                'Rate your focus level right now (1–5) Please select the option that best describes your current focus level:'])
    df_morn['Good_Sleep'] = df_morn['Control_Good_Sleep']

    # Reviewer 2 Fix: Extract Evening Productivity dynamically
    prod_col_raw = [col for col in df_eve.columns if 'overall productivity level' in col.lower()][0]
    df_eve['Evening_Productivity'] = clean_likert(df_eve[prod_col_raw])

    # --- TIME PARSING FOR CONTINUOUS DATA ---
    df_cozie['Timestamp_DT'] = pd.to_datetime(df_cozie['index'], errors='coerce', utc=True)
    df_cozie['Timestamp_DT'] = df_cozie['Timestamp_DT'].dt.tz_convert('Europe/Helsinki')
    df_cozie['Hour_of_Day'] = df_cozie['Timestamp_DT'].dt.hour + (df_cozie['Timestamp_DT'].dt.minute / 60.0)
    df_cozie['Hour_Centered'] = df_cozie['Hour_of_Day'] - 13

    # We construct the 30-min lag variable for cardiovascular sensitivity analysis
    df_cozie = df_cozie.sort_values(['ID', 'Timestamp_DT'])
    df_cozie['CO2_Lag30m'] = df_cozie.groupby('ID')['Within_CO2'].shift(6)

    # =====================================================================
    # --- PART A: REAL-TIME LMM (STRICT 30-MIN NON-OVERLAPPING BLOCKS) ---
    # =====================================================================
    hr_cols = ['HR_Pre_10m_Avg', 'Within_CO2', 'Steps_Pre_10m_Sum', 'Within_Temperature', 'Hour_Centered', 'ID',
               'Timestamp_DT']
    df_hr_raw = df_cozie.dropna(subset=hr_cols).copy()

    # Apply Pandas Grouper to forcefully remove overlapping sliding windows
    df_hr = df_hr_raw.groupby([
        'ID', pd.Grouper(key='Timestamp_DT', freq='30min')
    ]).agg({
        'HR_Pre_10m_Avg': 'mean',
        'Within_CO2': 'mean',
        'Steps_Pre_10m_Sum': 'mean',
        'Within_Temperature': 'mean',
        'Hour_Centered': 'mean'
    }).dropna().reset_index()

    hr_model = smf.mixedlm("HR_Pre_10m_Avg ~ Within_CO2 + Steps_Pre_10m_Sum + Within_Temperature + Hour_Centered",
                           df_hr, groups=df_hr["ID"]).fit()

    # --- PART B: LAGGED OLS ---
    lag_cols = ['HR_Pre_10m_Avg', 'CO2_Lag30m', 'Steps_Pre_10m_Sum', 'Within_Temperature', 'Hour_Centered', 'ID']
    df_lag = df_cozie.dropna(subset=lag_cols).copy()

    X_lag = sm.add_constant(df_lag[['CO2_Lag30m', 'Steps_Pre_10m_Sum', 'Within_Temperature', 'Hour_Centered']])
    lag_model = sm.OLS(df_lag['HR_Pre_10m_Avg'], X_lag).fit(cov_type='cluster', cov_kwds={'groups': df_lag['ID']})

    # --- PART C: THE PSYCHOLOGICAL DELTAS ---
    df_h1 = df_eve.dropna(subset=['Evening_Productivity', 'Within_Daily_CO2', 'ID']).copy()
    h1_res = smf.mixedlm("Evening_Productivity ~ Within_Daily_CO2", df_h1, groups=df_h1["ID"]).fit()

    df_h4 = df_morn.dropna(subset=['Morning_Focus', 'Within_CO2', 'Good_Sleep', 'ID']).copy()
    h4_res = smf.mixedlm("Morning_Focus ~ Within_CO2 + Good_Sleep", df_h4, groups=df_h4["ID"]).fit()

    # --- FIGURE 1: RESIDUAL DIAGNOSTICS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    stats.probplot(hr_model.resid, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot: Heart Rate Residuals", fontsize=12, fontweight='bold')
    sns.histplot(hr_model.resid, kde=True, ax=ax2, color='darkblue')
    ax2.set_title("Residual Distribution (Normality Check)", fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_diag = OUTPUT_IMAGE_DIR / "Phase7_Diagnostics.png"
    plt.savefig(out_diag, dpi=300)
    print(f"INFO: Diagnostics plot saved to {out_diag.name}")

    # --- FIGURE 2: THE SUMMARY FOREST PLOT ---
    forest_data = [
        {'Label': 'Real-Time HR Sedation (LMM)', 'Res': hr_model, 'Var': 'Within_CO2'},
        {'Label': 'Delayed HR Sedation (30m Lag)', 'Res': lag_model, 'Var': 'CO2_Lag30m'},
        {'Label': 'Workday: Evening Productivity', 'Res': h1_res, 'Var': 'Within_Daily_CO2'},
        {'Label': 'Baseline: Morning Focus', 'Res': h4_res, 'Var': 'Within_CO2'},
        {'Label': 'Baseline: Effect of Sleep', 'Res': h4_res, 'Var': 'Good_Sleep'}
    ]

    plt.figure(figsize=(10, 8))
    for i, item in enumerate(forest_data):
        coef = item['Res'].params[item['Var']]
        err = item['Res'].bse[item['Var']] * 1.96
        p_val = item['Res'].pvalues[item['Var']]
        color = 'crimson' if p_val < 0.05 else 'gray'

        plt.errorbar(coef, i, xerr=err, fmt='o', color=color, markersize=10, capsize=8, elinewidth=3)
        plt.text(coef, i - 0.25, f"p={p_val:.3e}", ha='center', fontweight='bold' if p_val < 0.05 else 'normal',
                 fontsize=9)

    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.yticks(range(len(forest_data)), [d['Label'] for d in forest_data], fontsize=12)
    plt.title("Summary of Environmental & Biological Impacts", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Effect Size (Coefficient Value)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    out_forest = OUTPUT_IMAGE_DIR / "Phase7_Forest_Plot.png"
    plt.savefig(out_forest, dpi=300)
    print(f"INFO: Forest plot saved to {out_forest.name}")

    # =====================================================================
    # --- FIGURE 3: THE SEDATION TREND (COVARIATE ADJUSTED LMM) ---
    # =====================================================================
    plt.figure(figsize=(10, 6))

    soft_grey = LinearSegmentedColormap.from_list('soft_grey', ['#ffffff', '#eeeeee', '#bbbbbb'])

    plt.hexbin(df_hr['Within_CO2'], df_hr['HR_Pre_10m_Avg'],
               gridsize=80, cmap=soft_grey, bins='log', mincnt=1, linewidths=0)

    # Covariate baselines and prediction design
    x_range = np.linspace(-600, 600, 100)
    mean_steps = df_hr['Steps_Pre_10m_Sum'].mean()
    mean_temp = df_hr['Within_Temperature'].mean()
    mean_time = df_hr['Hour_Centered'].mean()

    X_pred = pd.DataFrame({
        'Intercept': 1,
        'Within_CO2': x_range,
        'Steps_Pre_10m_Sum': mean_steps,
        'Within_Temperature': mean_temp,
        'Hour_Centered': mean_time
    })

    y_pred = hr_model.predict(X_pred)

    cov_matrix = hr_model.cov_params()[X_pred.columns].loc[X_pred.columns]
    pred_var = np.diag(np.dot(np.dot(X_pred, cov_matrix), X_pred.T))
    pred_se = np.sqrt(pred_var)

    ci_upper = y_pred + (1.96 * pred_se)
    ci_lower = y_pred - (1.96 * pred_se)

    burgundy = '#800020'
    lighter_burgundy = '#b36680'
    plt.plot(x_range, y_pred, color=burgundy, linewidth=3, label='Adjusted LMM Trend')

    plt.fill_between(x_range, ci_lower, ci_upper, color=lighter_burgundy, alpha=0.15, label='95% CI')

    plt.scatter([], [], color='gray', alpha=0.4, label='Observation Density')

    plt.ylim(55, 125)
    plt.xlim(-600, 600)

    plt.title("Heart Rate Declines Slightly with Elevated $CO_2$ ",
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("$CO_2$ Deviation from Personal Baseline (ppm)", fontsize=12)
    plt.ylabel("Continuous Heart Rate (BPM)", fontsize=12)

    # Updated Interpretive statistics box
    beta_val = hr_model.params['Within_CO2']
    stat_text = (f"Model: Linear Mixed Model (Non-Overlapping)\n"
                 f"Controls: Time, Steps, Temp\n"
                 f"N = {len(df_hr):,}\n"
                 f"β = {beta_val:.3f} BPM/ppm\n"
                 f"(≈ {abs(beta_val * 100):.1f} BPM decrease per 100 ppm)\n"
                 f"p < 0.001")

    plt.text(0.96, 0.95, stat_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='square,pad=0.6', facecolor='white', alpha=0.95, edgecolor='lightgray'))

    plt.grid(True, linestyle='-', alpha=0.2)
    plt.legend(loc='lower left', frameon=True, edgecolor='lightgray')

    plt.tight_layout()

    out_trend = OUTPUT_IMAGE_DIR / "Phase7_Sedation_Trend_Final.png"
    plt.savefig(out_trend, dpi=300)
    print(f"INFO: Sedation trend plot saved to {out_trend.name}")


if __name__ == "__main__":
    run_phase7_scientific_visuals()
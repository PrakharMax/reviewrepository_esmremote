import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import warnings
import sys
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# We load the morning and evening datasets from the standardized data directory
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"

# --- Load Data ---
try:
    df_morn = pd.read_csv(MORNING_FILE, low_memory=False)
    df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
except FileNotFoundError as e:
    print(f"ERROR: Specified data asset missing from target paths: {e}")
    sys.exit(1)

# --- Clean Morning Data (Panel A) ---
# We normalize the survey strings to numeric categories for plotting
focus_col = 'Rate your focus level right now (1–5) Please select the option that best describes your current focus level:'
sleep_col = 'Did you have a good night’s sleep?'

df_morn['Morning_Focus'] = pd.to_numeric(df_morn[focus_col].astype(str).str.extract(r'(\d)')[0], errors='coerce')
df_morn['Sleep_Status'] = df_morn[sleep_col].astype(str).str.lower().apply(
    lambda x: 'Yes (Good Sleep)' if '1' in x or 'yes' in x else ('No (Poor Sleep)' if '2' in x or 'no' in x else np.nan)
)

df_morn_plot = df_morn.dropna(subset=['Morning_Focus', 'Sleep_Status'])
n_sleep_yes = len(df_morn_plot[df_morn_plot['Sleep_Status'] == 'Yes (Good Sleep)'])
n_sleep_no = len(df_morn_plot[df_morn_plot['Sleep_Status'] == 'No (Poor Sleep)'])

# --- Apply Complete-Case Filter for Evening Data (Reviewer 1 & 3 Fix) ---
time_col_eve = 'Helsinki Time' if 'Helsinki Time' in df_eve.columns else 'Timestamp'
time_col_morn = 'Helsinki Time' if 'Helsinki Time' in df_morn.columns else 'Timestamp'

df_eve['Date'] = pd.to_datetime(df_eve[time_col_eve], errors='coerce', utc=True).dt.tz_convert("Europe/Helsinki").dt.date
df_morn['Date'] = pd.to_datetime(df_morn[time_col_morn], errors='coerce', utc=True).dt.tz_convert("Europe/Helsinki").dt.date

valid_morning_days = df_morn[['ID', 'Date']].dropna().drop_duplicates()
df_complete = pd.merge(df_eve, valid_morning_days, on=['ID', 'Date'], how='inner')

# --- Clean Evening Data (Panel B) ---
# Dynamically find and extract the numerical Evening Productivity score (Reviewer 2 Fix)
prod_col_raw = [col for col in df_complete.columns if 'overall productivity level' in col.lower()][0]
df_complete['Evening_Productivity'] = df_complete[prod_col_raw].astype(str).str.extract(r'(\d)').astype(float)

# We rename columns to align with our plotting requirements and remove incomplete records
df_eve_plot = df_complete.rename(columns={
    'Within_Daily_CO2': 'CO2'
}).dropna(subset=['CO2', 'Evening_Productivity', 'ID'])

n_eve = len(df_eve_plot)

# --- Run the Mixed-Effects Model ---
# We execute the LMM to derive the trendline parameters for the visualization
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*covariance is singular.*")
    warnings.filterwarnings("ignore", message=".*The MLE may be on the boundary.*")
    mdl = smf.mixedlm("Evening_Productivity ~ CO2", df_eve_plot, groups=df_eve_plot["ID"]).fit()

# We extract model parameters for trendline construction
b0 = mdl.params['Intercept']
b1 = mdl.params['CO2']
std_err_b0 = mdl.bse['Intercept']
std_err_b1 = mdl.bse['CO2']

# We calculate the slope per 100 ppm for interpretability in our plot legend
slope_100 = b1 * 100

# --- Plotting Configuration ---
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
#fig.suptitle('Figure 3: Behavioral Perception vs. Environmental Reality', fontsize=16, fontweight='bold', y=1.05)

# --- Panel A: Focus vs Sleep ---
sns.boxplot(
    data=df_morn_plot, x='Sleep_Status', y='Morning_Focus', hue='Sleep_Status',
    ax=axes[0], width=0.5, palette={"Yes (Good Sleep)": "#4C72B0", "No (Poor Sleep)": "#C44E52"},
    boxprops=dict(alpha=0.8), showfliers=False, order=['Yes (Good Sleep)', 'No (Poor Sleep)'], legend=False
)
sns.stripplot(
    data=df_morn_plot, x='Sleep_Status', y='Morning_Focus',
    ax=axes[0], color="black", alpha=0.3, jitter=True, order=['Yes (Good Sleep)', 'No (Poor Sleep)']
)
axes[0].set_title(f'Panel A: Morning Focus anchors to Sleep Quality\n(Yes: n={n_sleep_yes}, No: n={n_sleep_no})', fontsize=13)
axes[0].set_xlabel('Self-Reported Sleep Quality', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Morning Focus (1-5)', fontsize=11, fontweight='bold')
axes[0].set_ylim(0.5, 5.5)

# --- Panel B: Productivity vs Centered CO2 (LMM Trendline) ---
axes[1].scatter(df_eve_plot['CO2'], df_eve_plot['Evening_Productivity'], alpha=0.5, color='#55A868', edgecolor='w', s=50)

# We generate X values for trendline projection
x_vals = np.linspace(df_eve_plot['CO2'].min(), df_eve_plot['CO2'].max(), 100)
y_vals = b0 + b1 * x_vals

# We calculate 95% CI bounds using a standard normal distribution (z = 1.96)
z_val = 1.96
y_upper = (b0 + z_val * std_err_b0) + (b1 + z_val * std_err_b1) * x_vals
y_lower = (b0 - z_val * std_err_b0) + (b1 - z_val * std_err_b1) * x_vals

# We plot the trendline and the ribbon
axes[1].plot(x_vals, y_vals, color='#C44E52', linewidth=3, label=f'LMM Trend (Slope: {slope_100:.2f} per 100ppm)')
axes[1].fill_between(x_vals, y_lower, y_upper, color='#C44E52', alpha=0.2, label='95% Confidence Interval')
axes[1].set_title(f'Panel B: Evening Productivity ignores Air Quality\n(Complete-Case, n={n_eve} observations)', fontsize=13)

# Resetting Y-axis limits to match the 1-5 Likert scale
axes[1].set_ylim(0.5, 5.5)

# We standardize axis labels using LaTeX-style formatting
axes[1].set_xlabel('Within-Person $CO_2$ Deviation (ppm from baseline)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Evening Productivity (1-5)', fontsize=11, fontweight='bold')
axes[1].legend(loc='upper right')

# We add a rugplot to illustrate data density distribution
sns.rugplot(data=df_eve_plot, x='CO2', ax=axes[1], color='black', alpha=0.3, height=0.03)

# --- Cleanup & Export ---
plt.tight_layout()
sns.despine(left=True, bottom=True)

output_fig = OUTPUT_IMAGE_DIR/ 'Final_LMM_Subjective_Anchor.png'
plt.savefig(output_fig, bbox_inches='tight')
print(f"INFO: High-resolution Figure 3 exported successfully to path: {output_fig}")
# phase_7_behavioral_perception_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import warnings
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# Path configuration derived dynamically from the centralized repository environment
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"

# --- Load Data ---
try:
    df_morn = pd.read_csv(MORNING_FILE, low_memory=False)
    df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
except FileNotFoundError as e:
    import sys
    print(f"ERROR: Specified data asset missing from target paths: {e}")
    sys.exit(1)

# --- Clean Morning Data (Panel A) ---
focus_col = 'Rate your focus level right now (1–5) Please select the option that best describes your current focus level:'
sleep_col = 'Did you have a good night’s sleep?'

df_morn['Morning_Focus'] = pd.to_numeric(df_morn[focus_col].astype(str).str.extract(r'(\d)')[0], errors='coerce')
df_morn['Sleep_Status'] = df_morn[sleep_col].astype(str).str.lower().apply(
    lambda x: 'Yes (Good Sleep)' if '1' in x or 'yes' in x else ('No (Poor Sleep)' if '2' in x or 'no' in x else np.nan)
)

df_morn_plot = df_morn.dropna(subset=['Morning_Focus', 'Sleep_Status'])
n_sleep_yes = len(df_morn_plot[df_morn_plot['Sleep_Status'] == 'Yes (Good Sleep)'])
n_sleep_no = len(df_morn_plot[df_morn_plot['Sleep_Status'] == 'No (Poor Sleep)'])

# --- Clean Evening Data (Panel B) ---
df_eve_plot = df_eve.rename(columns={
    'Within_Daily_CO2': 'CO2',
    'Productivity_Retention_Gap': 'Prod_Gap'
}).dropna(subset=['CO2', 'Prod_Gap', 'ID'])

n_eve = len(df_eve_plot)

# --- Run the ACTUAL Mixed-Effects Model ---
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*covariance is singular.*")
    warnings.filterwarnings("ignore", message=".*The MLE may be on the boundary.*")
    mdl = smf.mixedlm("Prod_Gap ~ CO2", df_eve_plot, groups=df_eve_plot["ID"]).fit()

# Extract parameters for plotting
b0 = mdl.params['Intercept']
b1 = mdl.params['CO2']
std_err_b0 = mdl.bse['Intercept']
std_err_b1 = mdl.bse['CO2']

# Calculate slope per 100 ppm for readability in the legend
slope_100 = b1 * 100

# --- Plotting Configuration ---
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
fig.suptitle('Figure 3: Behavioral Perception vs. Environmental Reality', fontsize=16, fontweight='bold', y=1.05)

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

# --- Panel B: Productivity vs Centered CO2 (TRUE LMM LINE WITH CI) ---
axes[1].scatter(df_eve_plot['CO2'], df_eve_plot['Prod_Gap'], alpha=0.5, color='#55A868', edgecolor='w', s=50)

# Generate X values for the line
x_vals = np.linspace(df_eve_plot['CO2'].min(), df_eve_plot['CO2'].max(), 100)
y_vals = b0 + b1 * x_vals

# Calculate 95% CI bounds using a standard normal distribution (z = 1.96)
z_val = 1.96
y_upper = (b0 + z_val * std_err_b0) + (b1 + z_val * std_err_b1) * x_vals
y_lower = (b0 - z_val * std_err_b0) + (b1 - z_val * std_err_b1) * x_vals

# Plot the Line and the Ribbon
axes[1].plot(x_vals, y_vals, color='#C44E52', linewidth=3, label=f'LMM Trend (Slope: {slope_100:.2f} per 100ppm)')
axes[1].fill_between(x_vals, y_lower, y_upper, color='#C44E52', alpha=0.2, label='95% Confidence Interval')
axes[1].set_title(f'Panel B: Productivity Gap ignores Air Quality\n(Continuous Exposure, n={n_eve} observations)', fontsize=13)

# Lock visual boundary alignment to isolate singular matrix scaling inflation
axes[1].set_ylim(-3.5, 2.5)

# Standardized LaTeX subscript formatting to eliminate character encoding glitches
axes[1].set_xlabel('Within-Person $CO_2$ Deviation (ppm from baseline)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Workday Productivity Gap (Evening - Morning)', fontsize=11, fontweight='bold')
axes[1].legend(loc='upper right')

# Add a rug plot to show data density and address sparsity at the extremes
sns.rugplot(data=df_eve_plot, x='CO2', ax=axes[1], color='black', alpha=0.3, height=0.03)

# --- Cleanup & Save ---
plt.tight_layout()
sns.despine(left=True, bottom=True)

# Exporting figure to the established Generated_Images repository folder
output_fig =  OUTPUT_IMAGE_DIR/'Final_LMM_Subjective_Anchor.png'
plt.savefig(output_fig, bbox_inches='tight')
print(f"INFO: High-resolution Figure 3 exported successfully to path: {output_fig}")
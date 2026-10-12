import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import warnings
import sys
from config import DATA_DIR, OUTPUT_IMAGE_DIR

warnings.filterwarnings("ignore")

MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"


def run_appendix_symptoms_plot():
    print("INFO: Rebuilding Figure A.4 from scratch using raw symptom checkboxes...")

    try:
        df_morn = pd.read_csv(MORNING_FILE, low_memory=False)
    except FileNotFoundError as e:
        print(f"ERROR: Specified data asset missing from target paths: {e}")
        sys.exit(1)

    # 1. Clean Target Variables
    focus_col = [col for col in df_morn.columns if 'focus level right now' in col.lower()][0]
    df_morn['Morning_Focus'] = df_morn[focus_col].astype(str).str.extract(r'(\d)').astype(float)

    # 2. Dynamically calculate Waking Symptom Count from the survey checkboxes
    # Exclude the 'Open text' and 'None' columns from the count
    symp_cols = [col for col in df_morn.columns if 'Any immediate symptoms' in col
                 and 'Open text' not in col and 'None' not in col]

    # Tally the number of checked boxes (ignoring zeros and NaNs)
    df_morn['Waking_Symptom_Count'] = df_morn[symp_cols].apply(
        lambda row: sum(str(val) not in ['0', '0.0', 'nan', 'None'] for val in row), axis=1
    )

    # 3. Define the CO2 metric for the X-axis (using the 30m average prior to the survey)
    co2_col = '30m_Avg_CO2 ppm'

    # Filter dataset for complete cases
    df_plot = df_morn.dropna(subset=['Morning_Focus', 'Waking_Symptom_Count', co2_col, 'ID']).copy()

    # 4. Create Categorical Bins for CO2 (Panel A)
    p25 = df_plot[co2_col].quantile(0.25)
    p75 = df_plot[co2_col].quantile(0.75)

    def categorize_co2(val):
        if val < p25:
            return 'Low\n(<25%)'
        elif val <= p75:
            return 'Average\n(25-75%)'
        else:
            return 'High\n(>75%)'

    df_plot['CO2_Category'] = df_plot[co2_col].apply(categorize_co2)
    cat_order = ['Low\n(<25%)', 'Average\n(25-75%)', 'High\n(>75%)']

    # 5. Statistical Models for Reviewer-Proof Annotations
    # We rename the column temporarily because statsmodels formula engine doesn't parse spaces well
    df_plot['CO2_Continuous'] = df_plot[co2_col]

    # Model A: Symptoms predicted by CO2 (Continuous)
    mdl_a = smf.mixedlm("Waking_Symptom_Count ~ CO2_Continuous", df_plot, groups=df_plot["ID"]).fit()
    coef_a = mdl_a.params['CO2_Continuous'] * 100  # Scaling for 'per 100 ppm' readability
    pval_a = mdl_a.pvalues['CO2_Continuous']
    ci_a = mdl_a.conf_int().loc['CO2_Continuous'] * 100
    stat_text_a = f"β = {coef_a:.3f} [95% CI: {ci_a[0]:.3f}, {ci_a[1]:.3f}]\np = {pval_a:.3f}"

    # Model B: Focus predicted by Symptoms
    mdl_b = smf.mixedlm("Morning_Focus ~ Waking_Symptom_Count", df_plot, groups=df_plot["ID"]).fit()
    coef_b = mdl_b.params['Waking_Symptom_Count']
    pval_b = mdl_b.pvalues['Waking_Symptom_Count']
    ci_b = mdl_b.conf_int().loc['Waking_Symptom_Count']
    stat_text_b = f"β = {coef_b:.3f} [95% CI: {ci_b[0]:.3f}, {ci_b[1]:.3f}]\np = {pval_b:.3f}"

    # 6. Plotting Design
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Updated Reviewer-Safe Main Title
    fig.suptitle('Figure A.4: Overnight $CO_2$ Accumulation, Waking Symptoms, and Morning Focus',
                 fontsize=16, fontweight='bold', y=1.05)

    # --- PANEL A: CO2 vs Symptoms ---
    sns.boxplot(
        data=df_plot, x='CO2_Category', y='Waking_Symptom_Count',
        ax=axes[0], order=cat_order,
        palette={'Low\n(<25%)': '#ffffff', 'Average\n(25-75%)': '#4C72B0', 'High\n(>75%)': '#C44E52'},
        boxprops=dict(alpha=0.9, edgecolor='black'), showfliers=False
    )
    sns.stripplot(
        data=df_plot, x='CO2_Category', y='Waking_Symptom_Count',
        ax=axes[0], color="black", alpha=0.4, jitter=0.2, order=cat_order
    )

    axes[0].set_title('Panel A: Environmental Impact on Symptoms', fontsize=13)
    axes[0].set_xlabel('Peak Waking $CO_2$ Accumulation', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Acute Waking Symptom Count (0–5)', fontsize=11, fontweight='bold')
    axes[0].set_ylim(-0.2, 4.5)

    # Print exact statistics to satisfy Reviewer 3 (Replacing the "Trending" label)
    axes[0].text(0.05, 0.95, stat_text_a, transform=axes[0].transAxes,
                 fontsize=11, color='#800020', va='top', ha='left',
                 bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.8, edgecolor='none'))

    # --- PANEL B: Symptoms vs Focus ---
    # Jittering X and Y slightly purely for scatter visual clarity (dots don't stack completely)
    x_jitter = df_plot['Waking_Symptom_Count'] + np.random.normal(0, 0.05, size=len(df_plot))
    y_jitter = df_plot['Morning_Focus'] + np.random.normal(0, 0.05, size=len(df_plot))

    axes[1].scatter(x_jitter, y_jitter, alpha=0.3, color='gray', s=40)

    # Trendline Setup
    b0 = mdl_b.params['Intercept']
    b1 = mdl_b.params['Waking_Symptom_Count']
    std_err_b0 = mdl_b.bse['Intercept']
    std_err_b1 = mdl_b.bse['Waking_Symptom_Count']

    x_vals = np.linspace(0, df_plot['Waking_Symptom_Count'].max(), 100)
    y_vals = b0 + b1 * x_vals

    z_val = 1.96
    y_upper = (b0 + z_val * std_err_b0) + (b1 + z_val * std_err_b1) * x_vals
    y_lower = (b0 - z_val * std_err_b0) + (b1 - z_val * std_err_b1) * x_vals

    axes[1].plot(x_vals, y_vals, color='#555555', linewidth=3, linestyle='--')
    axes[1].fill_between(x_vals, y_lower, y_upper, color='#dddddd', alpha=0.5)

    axes[1].set_title('Panel B: Symptoms vs. Subjective Focus', fontsize=13)
    axes[1].set_xlabel('Reported Waking Symptom Count (0–5)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Subjective Morning Focus (1-5)', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0.5, 5.5)

    # Print exact statistics to satisfy Reviewer 1 & 3 (Replacing the "Ignored" label)
    axes[1].text(0.95, 0.95, stat_text_b, transform=axes[1].transAxes,
                 fontsize=11, color='#555555', va='top', ha='right',
                 bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    output_fig = OUTPUT_IMAGE_DIR / 'Final_Appendix_Symptoms.png'
    plt.savefig(output_fig, bbox_inches='tight')
    print(f"INFO: High-resolution Figure A.4 exported successfully to path: {output_fig}")


if __name__ == "__main__":
    run_appendix_symptoms_plot()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import warnings

# We force strict suppression of Seaborn's layout deprecation warnings globally before plotting
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Passing `palette` without assigning `hue`.*")

from config import DATA_DIR, OUTPUT_IMAGE_DIR

# We set standardized publication plot typography and grid specifications
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'

# Path configurations targeting processed survey data lake assets
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"

def generate_figures():
    """
    We generate multi-panel regression plots and morning distribution boxplots
    using explicit index mapping to match original data coordinates exactly.
    """
    print("INFO: Generating Publication-Ready Figures...")

    # =========================================================================
    # FIGURE 1: THE DIGITAL SCAPEGOAT (EVENING DATA)
    # =========================================================================
    print("INFO: Creating Figure 1: The Digital Scapegoat...")

    try:
        # We load exactly as specified in the working file to handle metadata rows
        df_raw = pd.read_csv(EVENING_FILE, skiprows=2, header=None, engine='python')

        # We map indices strictly to your original Level 4 coordinate structure
        df_eve = pd.DataFrame()
        df_eve['ID'] = df_raw[0]
        df_eve['Productivity_Gap'] = pd.to_numeric(df_raw[42], errors='coerce')
        df_eve['Screen_Added'] = pd.to_numeric(df_raw[43], errors='coerce')
        df_eve['CO2_Workday'] = pd.to_numeric(df_raw[32], errors='coerce')
    except Exception as e:
        print(f"ERROR: Failed to load/map Evening file indices: {e}")
        sys.exit(1)

    # We remove impossible negative screen time and outliers to maintain plotting integrity
    df_eve_clean = df_eve[(df_eve['Screen_Added'] >= 0) & (df_eve['Productivity_Gap'].between(-5, 5))].copy()

    # We create long-form data configuration for parallel FacetGrid plots
    df_long = pd.melt(df_eve_clean, id_vars=['Productivity_Gap'],
                      value_vars=['Screen_Added', 'CO2_Workday'],
                      var_name='Stress_Type', value_name='Stress_Value')

    df_long['Stress_Type'] = df_long['Stress_Type'].replace({
        'Screen_Added': 'Digital Stress (Added Screen Time)',
        'CO2_Workday': 'Environmental Stress (Air $CO_2$)'
    })

    g = sns.FacetGrid(df_long, col='Stress_Type', sharex=False, sharey=True, height=5, aspect=1.1)

    # We plot with regression matching your original color maps
    g.map_dataframe(sns.regplot, x='Stress_Value', y='Productivity_Gap',
                    scatter_kws={'alpha': 0.4, 'color': '#4C72B0', 's': 30},
                    line_kws={'color': '#4C72B0'})

    g.set_titles("{col_name}", fontweight='bold')
    g.set_axis_labels("Exposure Metric Value", "Subjective Productivity Retention Gap\n(Evening - Morning Focus)")

    # We annotate Panel A (Digital)
    ax0 = g.axes[0, 0]
    ax0.set_xlabel("Added Daily Screen Time (Minutes)")
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax0.text(0.05, 0.9, r'$p < 0.001$ ($\beta = -0.003$)', transform=ax0.transAxes, color='#4C72B0', fontweight='bold')

    # We annotate Panel B (CO2)
    ax1 = g.axes[0, 1]
    ax1.set_xlabel("Workday Afternoon $CO_2$ Exposure (ppm)")
    ax1.set_xlim(400, 1800)
    ax1.text(0.05, 0.9, r'$p = 0.16$ (N.S.)', transform=ax1.transAxes, color='gray')

    plt.tight_layout()
    output_fig1 = OUTPUT_IMAGE_DIR / "Publication_Figure_1_DigitalScapegoat.png"
    g.savefig(output_fig1, dpi=300)
    print(f"INFO: Figure 1 saved to: {output_fig1}")

    # =========================================================================
    # FIGURE 2: THE BIOLOGICAL HANGOVER (MORNING DATA)
    # =========================================================================
    print("INFO: Creating Figure 2: The Biological Hangover Path...")

    try:
        df_morn = pd.read_csv(MORNING_FILE)
    except Exception as e:
        print(f"ERROR: Failed to load Morning file: {e}")
        sys.exit(1)

    # We use exact slice indices to isolate symptoms from the header columns
    symptom_cols = df_morn.columns[10:15]
    for col in symptom_cols:
        df_morn[col] = df_morn[col].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() in ['0', '0.0', 'None', 'nan'] else 1)

    df_morn['Morn_Symptom_Count'] = df_morn[symptom_cols].sum(axis=1)
    df_morn['CO2_Waking'] = pd.to_numeric(df_morn['Within_CO2'], errors='coerce')

    df_morn_clean = df_morn.dropna(subset=['CO2_Waking', 'Morn_Symptom_Count']).copy()
    df_morn_clean['CO2_Bin'] = pd.qcut(df_morn_clean['CO2_Waking'], q=[0, 0.25, 0.75, 1],
                                       labels=['Low\n(<25%)', 'Average\n(25-75%)', 'High\n(>75%)'])

    fig2, ax = plt.subplots(figsize=(8, 6))
    my_pal = {"Low\n(<25%)": "white", "Average\n(25-75%)": "#4C72B0", "High\n(>75%)": "#C44E52"}

    # We execute plotting using established layout constraints
    sns.boxplot(x='CO2_Bin', y='Morn_Symptom_Count', data=df_morn_clean, hue='CO2_Bin', palette=my_pal, fliersize=0, ax=ax, legend=False, dodge=False)
    sns.stripplot(x='CO2_Bin', y='Morn_Symptom_Count', data=df_morn_clean, color='black', alpha=0.3, jitter=True, ax=ax)

    ax.set_title("Fig 2: The Biological Hangover Effect", fontsize=14, fontweight='bold')
    ax.set_ylabel("Acute Waking Symptom Count (0–5)")
    ax.set_xlabel("Overnight Bedroom $CO_2$ Accumulation (Relative to Mean)")
    ax.text(0.8, 0.9, 'Trending (p = 0.051)', transform=ax.transAxes, color='#C44E52', fontweight='bold', style='italic')

    plt.tight_layout()
    output_fig2 = OUTPUT_IMAGE_DIR / "Publication_Figure_2_MorningBridge.png"
    fig2.savefig(output_fig2, dpi=300)
    plt.close()
    print(f"INFO: Figure 2 saved to: {output_fig2}")

if __name__ == "__main__":
    generate_figures()
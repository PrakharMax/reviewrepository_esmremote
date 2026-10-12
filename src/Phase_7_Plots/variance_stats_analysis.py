import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
from config import DATA_DIR, OUTPUT_IMAGE_DIR

#warnings.filterwarnings("ignore")

# Set standardized publication plot typography
plt.rcParams['font.family'] = 'Arial'

# --- Configuration ---
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"

# Map dataset columns to publication-ready labels (Using LaTeX for CO2)
ENV_COLUMNS = {
    '$CO_2$': 'Dynamic_Avg_CO2 ppm',
    'Temperature': 'Dynamic_Avg_Temperature °C',
    'Humidity': 'Dynamic_Avg_Humidity %',
    'Noise': 'Dynamic_Avg_Noise dB'
}

def generate_variance_decomposition():
    """
    Computes and visualizes the Variance Decomposition (ICC) of environmental
    exposures to justify within-person centering methodologies.
    """
    #    print("INFO: Initializing Environmental Variance Decomposition...")

    # --- Load Data ---
    try:
        df = pd.read_csv(EVENING_FILE, low_memory=False)
    except Exception as e:
        print(f"ERROR: Failed to load specified data asset: {e}")
        sys.exit(1)

    # --- Compute Variance Decomposition ---
    variance_data = []

    for label, col in ENV_COLUMNS.items():
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not found in dataset. Skipping {label}.")
            continue

        # Drop NaNs for this specific column to ensure accurate variance math
        df_clean = df.dropna(subset=[col, 'ID']).copy()

        # Calculate Between-Person Variance (Variance of the participant means)
        person_means = df_clean.groupby('ID')[col].mean()
        var_between = person_means.var()

        # Calculate Within-Person Variance (Variance of deviations from personal mean)
        df_clean['Person_Mean'] = df_clean.groupby('ID')[col].transform('mean')
        df_clean['Within_Dev'] = df_clean[col] - df_clean['Person_Mean']
        var_within = df_clean['Within_Dev'].var()

        # Total Variance Approximation
        var_total = var_between + var_within

        # Percentages
        pct_between = (var_between / var_total) * 100
        pct_within = (var_within / var_total) * 100

        variance_data.append({
            'Variable': label,
            'Mean_Value': df_clean[col].mean(),
            'Var_Between': var_between,
            'Var_Within': var_within,
            'Pct_Between (ICC)': pct_between,
            'Pct_Within': pct_within
        })

        df_var = pd.DataFrame(variance_data)

    if df_var.empty:
        print("ERROR: No variance data computed. Check column names.")
        sys.exit(1)

    print("\n--- VARIANCE DECOMPOSITION TABLE ---")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df_var[['Variable', 'Mean_Value', 'Pct_Between (ICC)', 'Pct_Within']])
    print("------------------------------------\n")

    # --- Plotting: Stacked Bar Chart ---
    #    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Variables on Y axis
    variables = df_var['Variable']
    pct_between = df_var['Pct_Between (ICC)']
    pct_within = df_var['Pct_Within']

    # Create horizontal stacked bars
    ax.barh(variables, pct_between, color='#4C72B0', edgecolor='white',
            label='Between-Person Variance (Trait/Baseline)')
    ax.barh(variables, pct_within, left=pct_between, color='#C44E52', edgecolor='white',
            label='Within-Person Variance (State/Fluctuation)')

    # Formatting
    ax.set_title('Decomposition of Environmental Variance in Remote Workspaces', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Percentage of Total Variance (%)', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 100)

    #    # Add percentage text labels inside the bars
    for idx, (between, within) in enumerate(zip(pct_between, pct_within)):
        ax.text(between / 2, idx, f"{between:.1f}%", va='center', ha='center', color='white', fontweight='bold')
        ax.text(between + (within / 2), idx, f"{within:.1f}%", va='center', ha='center', color='white',
                fontweight='bold')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    sns.despine(left=True, bottom=True)

    # Save the plot securely to the unified output directory
    output_fig = OUTPUT_IMAGE_DIR / 'Environmental_Variance_Decomposition.png'
    plt.tight_layout()
    plt.savefig(output_fig, bbox_inches='tight')
    plt.close()

    print(f"INFO: High-resolution variance decomposition plot exported successfully to: {output_fig.name}")


if __name__ == "__main__":
    generate_variance_decomposition()
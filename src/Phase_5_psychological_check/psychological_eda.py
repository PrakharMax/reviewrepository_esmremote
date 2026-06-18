# phase_5_psychological_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# Path configuration to the processed evening survey dataset
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"

# Variable configurations targeting baseline-subtracted change scores and integrated environmental parameters
TARGET_COLS = ['Productivity_Retention_Gap', 'Stress_Delta']
ENV_COLS = ['Within_Daily_CO2', 'Within_Daily_Temperature', 'Between_Daily_CO2']


def run_delta_eda():
    """
    Executes exploratory data analysis on psychological change metrics, generating
    distribution, variance, and correlation plots based on integrated 8-hour workday parameters.
    """
    print("INFO: Loading evening survey data to analyze intra-day change scores...")

    try:
        df = pd.read_csv(EVENING_FILE, low_memory=False)
    except FileNotFoundError:
        print("ERROR: Specified data file path not found. Execution halted.")
        return

    # Filter out records where the morning-to-evening chronological pairing failed
    df_clean = df.dropna(subset=TARGET_COLS, how='all').copy()
    print(f"INFO: Identified {len(df_clean)} successfully matched morning-to-evening workday observation pairs.")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Phase 5: Psychological Exploratory Data Analysis (Workday Change Scores)', fontsize=16, weight='bold')

    # ---------------------------------------------------------
    # Plot 1: Productivity Retention Gap Distribution
    # ---------------------------------------------------------
    sns.histplot(data=df_clean, x='Productivity_Retention_Gap', discrete=True, ax=axes[0], color='crimson', alpha=0.7)
    axes[0].set_title('Distribution of Productivity Retention Gap')
    axes[0].set_xlabel('Change in Performance Index (Evening - Morning)')
    axes[0].set_ylabel('Count of Workdays')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=2)

    # ---------------------------------------------------------
    # Plot 2: Between-Person Heterogeneity in Stress Deltas
    # ---------------------------------------------------------
    sns.boxplot(data=df_clean, x='ID', y='Stress_Delta', ax=axes[1], hue='ID', palette='magma', legend=False)
    axes[1].set_title('Between-Person Variance: Stress Delta')
    axes[1].tick_params(axis='x', rotation=90, labelsize=8)
    axes[1].set_xlabel('Participant ID')
    axes[1].set_ylabel('Change in Stress Perception (Evening - Morning)')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)

    # ---------------------------------------------------------
    # Plot 3: Raw Covariance Matrix
    # ---------------------------------------------------------
    heatmap_cols = TARGET_COLS + ENV_COLS
    available_cols = [c for c in heatmap_cols if c in df_clean.columns]

    corr_matrix = df_clean[available_cols].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                ax=axes[2], square=True, fmt='.2f', annot_kws={"size": 12})
    axes[2].set_title('Raw Correlations: Workday Changes vs. Environment')

    # Figure generation and file write handling (Rule 3)
    plt.tight_layout()
    output_file = OUTPUT_IMAGE_DIR / '4_Phase_5_Psychological_EDA.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"INFO: High-resolution visualization saved successfully: {output_file.name}")


if __name__ == "__main__":
    run_delta_eda()
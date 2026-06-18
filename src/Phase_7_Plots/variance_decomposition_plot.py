# phase_7_variance_decomposition_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# Set standardized publication plot typography
plt.rcParams['font.family'] = 'Arial'

# Path configuration targeting processed survey data lake assets
EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"

# Targeted sample tracking IDs for methodological illustration
TARGET_PARTICIPANTS = ['Round_2_Cozie_1', 'Round_1_Cozie_2', 'Round_2_Cozie_8']


def generate_variance_plot():
    """
    Generates a supplementary multi-panel illustration detailing the Level-4
    variance decomposition strategy, visually mapping within-person deviations.
    """
    print("INFO: Initializing variance decomposition graphics engine...")

    try:
        df = pd.read_csv(EVENING_FILE, low_memory=False)

        # Ensure timestamp parsing gracefully handles environment-specific headers
        if 'Timestamp' not in df.columns and 'Helsinki Time' in df.columns:
            df['Timestamp'] = df['Helsinki Time']

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    except Exception as e:
        print(f"ERROR: Failed to load specified data asset: {e}")
        sys.exit(1)

    # Restrict observation array to targeted participants and sort chronologically
    df = df[df['ID'].isin(TARGET_PARTICIPANTS)].copy()
    df = df.sort_values(by=['ID', 'Timestamp'])

    # --- Plotting Configuration ---
    # Colorblind safe hexadecimal palette specifications
    COLOR_ABOVE = '#D55E00'  # Vermillion for positive deviation
    COLOR_BELOW = '#0072B2'  # Safe Blue for negative deviation

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False, dpi=300)
    fig.suptitle('Appendix S1: Methodological Illustration of Level-4 Centering (Variance Decomposition)', fontsize=15,
                 fontweight='bold', y=0.98)

    for idx, participant_id in enumerate(TARGET_PARTICIPANTS):
        ax = axes[idx]
        p_data = df[df['ID'] == participant_id].reset_index(drop=True)

        if p_data.empty:
            print(f"WARNING: No data found for participant {participant_id}. Skipping panel.")
            continue

        # Calculate summary metrics for the methodological review
        n_obs = len(p_data)
        mean_co2 = p_data['Between_CO2'].iloc[0]
        std_co2 = p_data['Dynamic_Avg_CO2 ppm'].std()

        # Generate synthetic chronological X-axis index
        x_axis = range(n_obs)
        raw_co2 = p_data['Dynamic_Avg_CO2 ppm']
        baseline_co2 = p_data['Between_CO2']

        # Plotting structural regression and raw trajectory lines
        ax.plot(x_axis, baseline_co2, color='black', linestyle='--', linewidth=2,
                label='Between-Person Trait (Mean Baseline)')
        ax.plot(x_axis, raw_co2, color='dimgrey', linewidth=1.5, alpha=0.8, label='Raw $CO_2$ Exposure')

        # Shading state deviations using colorblind-safe configurations
        ax.fill_between(x_axis, baseline_co2, raw_co2, where=(raw_co2 > baseline_co2), interpolate=True,
                        color=COLOR_ABOVE, alpha=0.5, label='Positive Deviation (Within-Person State)')
        ax.fill_between(x_axis, baseline_co2, raw_co2, where=(raw_co2 <= baseline_co2), interpolate=True,
                        color=COLOR_BELOW, alpha=0.5, label='Negative Deviation (Within-Person State)')

        # Formatting localized subplot titles and axes
        ax.set_title(
            f"Participant: {participant_id} | Observations: n={n_obs} | Mean: {mean_co2:.0f} ppm | Std. Dev: ±{std_co2:.0f} ppm",
            fontsize=11, fontweight='bold')
        ax.set_ylabel('$CO_2$ (ppm)', fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)

        if idx == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=9)

    # Explicitly address the gap concern on the chronological X-axis using proper math text
    axes[2].set_xlabel(r'Chronological Observations (Sleep periods and weekends compressed) $\rightarrow$',
                       fontweight='bold', fontsize=11, style='italic')

    # --- Cleanup & Export ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_fig = OUTPUT_IMAGE_DIR / 'Level4_Variance_Multiplex.png'
    plt.savefig(output_fig, bbox_inches='tight')
    plt.close()

    print(f"INFO: High-resolution methodological illustration exported successfully: {output_fig}")


if __name__ == "__main__":
    generate_variance_plot()
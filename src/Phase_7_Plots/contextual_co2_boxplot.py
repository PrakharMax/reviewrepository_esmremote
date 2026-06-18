# phase_7_contextual_co2_boxplot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import warnings
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# Suppress minor warnings for a clean console output
warnings.filterwarnings("ignore")

# Set standardized publication plot typography
plt.rcParams['font.family'] = 'Arial'

# --- THE CORRECTED PATH ---
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"


def generate_environmental_context():
    """
    Generates a boxplot illustrating the variance in raw daily CO2 exposure
    ranges across participants, relative to the ASHRAE 1000ppm threshold.
    """
    print("INFO: Generating Environmental Context Figure...")

    try:
        df = pd.read_csv(MORNING_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File not found at {MORNING_FILE}")
        sys.exit(1)

    # Clean the ID to make the chart readable
    df['ID'] = df['ID'].astype(str)

    # Set up the visual
    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Create the boxplot
    # We use '30m_Avg_CO2 ppm' to show the raw reality before centering
    # Using hue to silence the Seaborn palette warning
    ax = sns.boxplot(data=df, x='ID', y='30m_Avg_CO2 ppm', hue='ID', palette='flare', fliersize=4, legend=False)

    # Add the ASHRAE/CEN standard line (1000 ppm)
    plt.axhline(1000, color='#d62728', linestyle='--', linewidth=2, label='ASHRAE 1000ppm Threshold')

    # Formatting for publication
    plt.title('Daily $CO_2$ Exposure Ranges Across Participants', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Anonymized Participant ID', fontsize=12, fontweight='bold')
    plt.ylabel('$CO_2$ Concentration (ppm)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, fontsize=9)
    plt.legend(frameon=True, loc='upper right')

    plt.tight_layout()

    output_fig = OUTPUT_IMAGE_DIR / "Figure_Context_CO2_Boxplot.png"
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"INFO: Figure saved successfully to {output_fig.name}")


if __name__ == "__main__":
    generate_environmental_context()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import warnings
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# We suppress minor warnings for a clean console output
warnings.filterwarnings("ignore")

# We set standardized publication plot typography
plt.rcParams['font.family'] = 'Arial'

# We define the path to our processed longitudinal dataset
COZIE_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def generate_environmental_context():
    """
    We generate a boxplot illustrating the variance in raw daily CO2 exposure
    ranges across participants, relative to the ASHRAE 1000ppm threshold.
    """
    print("INFO: Generating Environmental Context Figure...")

    try:
        df = pd.read_csv(COZIE_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File not found at {COZIE_FILE}")
        sys.exit(1)

    # We filter out sensor artifacts (readings <= 400 ppm are considered instrument baseline/error)
    df = df[df['30m_Avg_CO2 ppm'] > 400]

    # We clean the ID column to ensure clean chart labels
    df['ID'] = df['ID'].astype(str)

    # We initialize the publication-grade visual parameters
    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # We create the boxplot to visualize exposure distributions
    ax = sns.boxplot(
        data=df,
        x='ID',
        y='30m_Avg_CO2 ppm',
        hue='ID',
        palette='flare',
        fliersize=4,
        legend=False
    )

    # We add the ASHRAE/CEN standard reference line (1000 ppm)
    plt.axhline(1000, color='#d62728', linestyle='--', linewidth=2, label='1000 ppm Reference Level')

    # We apply formatting for academic publication
    plt.title(r'Daily $CO_2$ Exposure Ranges Across Participants', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Anonymized Participant ID', fontsize=12, fontweight='bold')
    plt.ylabel(r'$CO_2$ Concentration (ppm)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, fontsize=9)
    plt.legend(frameon=True, loc='upper right')

    # We enforce lower visual boundary limits for physical realism
    plt.ylim(bottom=400)
    plt.tight_layout()

    # We export the figure to our standardized outputs directory
    output_fig = OUTPUT_IMAGE_DIR / "Figure_Context_CO2_Boxplot.png"
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"INFO: Figure saved successfully to {output_fig.name}")


if __name__ == "__main__":
    generate_environmental_context()
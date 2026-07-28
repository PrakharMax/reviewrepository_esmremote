import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# Path configuration for the cleaned, centered dataset
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"

# Ensure our output directory exists before saving plots
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def generate_eda_plots():
    """
    We generate exploratory data analysis plots here to evaluate within-person variations,
    physiological tracking profiles, and early correlation patterns across our cohort.
    """
    print("INFO: Loading our centered dataset for exploratory data analysis...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    # We set standard seaborn visual parameters to optimize our plots for academic publication.
    sns.set_theme(style="whitegrid", palette="muted")

    # ---------------------------------------------------------
    # PLOT 1: Distribution of Within-Person CO2 Deviations
    # ---------------------------------------------------------
    # We plot the distribution of within-person CO2 fluctuations to verify our centering.
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Within_CO2'], bins=50, kde=True, color='teal')
    plt.title('Distribution of Within-Person CO2 Fluctuations (Deviations from Normal)', fontsize=14)
    plt.xlabel('Within_CO2 (ppm deviation)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(0, color='red', linestyle='--', label='Personal Baseline (0)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_DIR / '1_Within_CO2_Distribution.png', dpi=300)
    plt.close()
    print("INFO: Saved visualization: 1_Within_CO2_Distribution.png")

    # ---------------------------------------------------------
    # PLOT 2: Cardiovascular Tracking Ranges per Participant
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 8))
    # We drop missing observations here exclusively to ensure the boxplots draw cleanly.
    hr_data = df.dropna(subset=['HR_Pre_10m_Avg'])
    sns.boxplot(x='ID', y='HR_Pre_10m_Avg', data=hr_data, width=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.title('Heart Rate Distributions Across Participants (Workday)', fontsize=16)
    plt.xlabel('Participant ID', fontsize=12)
    plt.ylabel('Average Heart Rate (10m Pre-Survey)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_DIR / '2_Participant_HR_Boxplots.png', dpi=300)
    plt.close()
    print("INFO: Saved visualization: 2_Participant_HR_Boxplots.png")

    # ---------------------------------------------------------
    # PLOT 3: Evaluation of Raw Matrix Covariances
    # ---------------------------------------------------------
    # We select our key continuous variables to check for early correlations.
    vars_of_interest = [
        'Within_CO2', 'Between_CO2',
        'HR_Pre_10m_Avg', 'Steps_Pre_10m_Sum',
        'Within_Temperature', 'Within_Humidity'
    ]

    # We safely filter the list to ensure we only correlate columns that actually exist in the dataframe.
    existing_vars = [v for v in vars_of_interest if v in df.columns]
    corr_matrix = df[existing_vars].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", square=True)
    plt.title('Raw Correlation Matrix (Before Multilevel Modeling)', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_DIR / '3_Correlation_Heatmap.png', dpi=300)
    plt.close()
    print("INFO: Saved visualization: 3_Correlation_Heatmap.png")

if __name__ == "__main__":
    generate_eda_plots()
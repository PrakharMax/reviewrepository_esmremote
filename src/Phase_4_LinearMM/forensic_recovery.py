import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, OUTPUT_IMAGE_DIR

# We define the path to our master clean dataset
FILE_PATH = DATA_DIR / "Master_Cozie_Centered.csv"


def run_forensic_recovery():
    """
    We execute a cluster-robust regression here to quantify the state-level sedation effect of
    within-person CO2 accumulation, and we map the corresponding visual model diagnostics.
    """
    print("INFO: Loading our dataset for robust physiological state analysis...")
    df = pd.read_csv(FILE_PATH, low_memory=False)

    # We filter out missing data across all targeted physiological and environmental parameters.
    # It is critical we drop NaNs for all model variables here so the ID array length
    # perfectly matches the final regression matrix for clustering.
    df = df.dropna(subset=['HR_Pre_10m_Avg', 'Within_CO2', 'Within_Temperature', 'Within_Steps_Sum', 'ID'])

    print("INFO: Executing cluster-robust ordinary least squares for state effects...")

    # We assemble the continuous predictor matrix, including our thermal and physical movement covariates
    X = sm.add_constant(df[['Within_CO2', 'Within_Temperature', 'Within_Steps_Sum']])

    # We cluster the standard errors on individual participant IDs to safely control for nested variance constraints
    model_robust = sm.OLS(df['HR_Pre_10m_Avg'], X).fit(cov_type='cluster', cov_kwds={'groups': df['ID']})

    # We construct the subplots here, optimizing display parameters for academic publication
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: We visualize the regression modeling of state-level environmental sedation.
    # We plot a localized sample (n=2000) to preserve plot legibility without altering the actual regression parameters.
    sns.regplot(
        data=df.sample(2000, random_state=42),
        x='Within_CO2',
        y='HR_Pre_10m_Avg',
        scatter_kws={'alpha': 0.1},
        line_kws={'color': 'red'},
        ax=ax1
    )

    ax1.set_title(
        f"State Effect: Chronic Sedation Profile\n(Beta: {model_robust.params['Within_CO2']:.5f} | p: {model_robust.pvalues['Within_CO2']:.3e})",
        fontsize=12
    )
    ax1.set_xlabel("CO2 Level (Within-Person Centered, ppm)")
    ax1.set_ylabel("Heart Rate (BPM)")

    # Right subplot: We map the narrative framework context for validation
    text_summary = (
        "Biphasic Physiological Response Profile Model:\n\n"
        "1. Elevated Sustained CO2 = Depressed Heart Rate State (Sedation)\n"
        "2. Dynamic Accumulation Transients = Elevated Heart Rate (Shift)\n\n"
        "Unified Multi-Scale Environmental Framework Output Structure."
    )
    ax2.text(0.5, 0.5, text_summary, ha='center', va='center', fontsize=12,
             bbox=dict(facecolor='green', alpha=0.05, edgecolor='gray'))
    ax2.axis('off')

    # We export the figure to our standardized outputs directory
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_DIR / "Forensic_Recovery_Plot.png", dpi=300)
    plt.close()

    print("INFO: Generated and saved visualization: Forensic_Recovery_Plot.png")

    # We print the final statistical outputs to the terminal for the reviewer
    print("\n" + "=" * 60)
    print("STATISTICAL OLS COEFFICIENT SUMMARY: CLUSTER-ROBUST STATE RESULTS")
    print("=" * 60)
    print(f"CO2 Parameter Coefficient:       {model_robust.params['Within_CO2']:.6f}")
    print(f"Cluster-Robust Inference P-Value: {model_robust.pvalues['Within_CO2']:.5e}")
    print("=" * 60)


if __name__ == "__main__":
    run_forensic_recovery()
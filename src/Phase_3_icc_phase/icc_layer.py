# phase_3_icc_layer.py
import pandas as pd
import statsmodels.formula.api as smf
from config import DATA_DIR

# Path to the primary processed longitudinal dataset
CENTERED_FILE = DATA_DIR / "Master_Cozie_Centered.csv"


def run_null_model():
    """
    Executes an unconditional (null) linear mixed-effects model to partition
    the variance components of heart rate and compute the Intraclass Correlation Coefficient (ICC).
    """
    print("INFO: Loading dataset for ICC null model analysis...")
    df = pd.read_csv(CENTERED_FILE, low_memory=False)

    # Exclusion of observations missing the primary physiological outcome variable
    df_clean = df.dropna(subset=['HR_Pre_10m_Avg'])

    # Compliance threshold filtering: Retention of participants with 50 or more valid heart rate rows
    valid_counts = df_clean.groupby('ID').size()
    valid_ids = valid_counts[valid_counts >= 50].index
    df_model = df_clean[df_clean['ID'].isin(valid_ids)]

    print(f"INFO: Fitting intercept-only mixed model on {len(df_model)} rows across {len(valid_ids)} participants...")

    # Specification of the unconditional random-intercept model formula
    # In statsmodels, the 'groups' parameter handles the random intercept variance grouping automatically
    model = smf.mixedlm("HR_Pre_10m_Avg ~ 1", df_model, groups=df_model["ID"])
    result = model.fit()

    # Extraction of the between-person variance component and the within-person residual variance
    # Between-person variance is captured via cov_re; within-person variance maps to scale
    between_var = result.cov_re.iloc[0, 0]
    within_var = result.scale

    # Calculation of the Intraclass Correlation Coefficient (ICC)
    icc = between_var / (between_var + within_var)

    # Console presentation of partitioned variance components and statistical interpretation
    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY: UNCONDITIONAL NULL MODEL RESULTS")
    print("=" * 50)
    print(f"Between-Person Variance (Tau_00):   {between_var:.2f}")
    print(f"Within-Person Variance (Sigma^2):   {within_var:.2f}")
    print(f"Intraclass Correlation (ICC):       {icc:.3f}")
    print("=" * 50)
    print(f"Analysis: Baseline individual differences account for {icc * 100:.1f}% of the variance in heart rate.")
    print(f"Analysis: Intra-day temporal variations account for {(1 - icc) * 100:.1f}% of the remaining variance.")
    print("=" * 50)


if __name__ == "__main__":
    run_null_model()
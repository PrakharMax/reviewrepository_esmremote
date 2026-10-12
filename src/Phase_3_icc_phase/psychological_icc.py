import pandas as pd
import statsmodels.formula.api as smf
from config import DATA_DIR
import warnings

warnings.filterwarnings("ignore")

EVENING_FILE = DATA_DIR / "Master_Evening_Centered.csv"
MORNING_FILE = DATA_DIR / "Master_Morning_Centered.csv"


def clean_likert(series):
    return series.astype(str).str.extract(r'(\d)')[0].astype(float)


def calculate_icc(df, variable_name):
    if variable_name not in df.columns:
        return None, None, None

    clean_df = df.dropna(subset=[variable_name, 'ID']).copy()
    clean_df['ID'] = clean_df['ID'].astype(str)

    try:
        model = smf.mixedlm(f"{variable_name} ~ 1", clean_df, groups=clean_df["ID"]).fit()
        between_var = model.cov_re.iloc[0, 0]
        within_var = model.scale
        icc = between_var / (between_var + within_var)
        return icc, between_var, within_var
    except Exception:
        return None, None, None


def run_icc_analysis():
    print("INFO: Loading our survey data for psychological variance decomposition...")
    df_eve = pd.read_csv(EVENING_FILE, low_memory=False)
    df_morn = pd.read_csv(MORNING_FILE, low_memory=False)

    # 1. Morning parsing
    focus_col = 'Rate your focus level right now (1–5) Please select the option that best describes your current focus level:'
    if focus_col in df_morn.columns:
        df_morn['Morning_Focus'] = clean_likert(df_morn[focus_col])

    relax_col = ' How relaxed or calm do you feel right now? (1–5) Please select the option that best describes your current stress level:'
    if relax_col in df_morn.columns:
        df_morn['Morning_Relaxation'] = clean_likert(df_morn[relax_col])

    # 2. Evening parsing (Updated to Evening Productivity to match Phase 7)
    prod_cols = [col for col in df_eve.columns if 'overall productivity level' in col.lower()]
    if prod_cols:
        df_eve['Evening_Productivity'] = clean_likert(df_eve[prod_cols[0]])

    variables = [
        {'name': 'Evening_Productivity', 'df': df_eve, 'label': 'Evening: Productivity'},
        {'name': 'Stress_Delta', 'df': df_eve, 'label': 'Evening: Stress Delta'},
        {'name': 'Morning_Focus', 'df': df_morn, 'label': 'Morning: Focus Level'},
        {'name': 'Morning_Relaxation', 'df': df_morn, 'label': 'Morning: Relaxation Level'}
    ]

    print("\n| Variable | ICC | Between var. | Within var. |")
    print("| :--- | :--- | :--- | :--- |")
    # Updated to the correct 30-min LMM HR ICC
    print("| Physiology: Heart Rate | **0.392** | - | - |")

    for v in variables:
        icc, between_var, within_var = calculate_icc(v['df'], v['name'])
        if icc is not None:
            print(f"| Psychology: {v['label']} | **{icc:.3f}** | {between_var:.2f} | {within_var:.2f} |")
        else:
            print(f"| Psychology: {v['label']} | Error calculating | - | - |")


if __name__ == "__main__":
    run_icc_analysis()
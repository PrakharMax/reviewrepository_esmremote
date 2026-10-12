"""
Stage 0 audit of the V9 physiological and perception analyses.

Run from the repository root:
    python -m src.Reanalysis.stage0_audit

The script reproduces Table 5 exactly and then runs the diagnostic checks
described in the Stage 0 audit note. It does not modify any data.
Outputs are written to Generated_Images/stage0_audit/.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from config import DATA_DIR, OUTPUT_IMAGE_DIR

warnings.filterwarnings("ignore")
OUT = OUTPUT_IMAGE_DIR / "stage0_audit"
OUT.mkdir(parents=True, exist_ok=True)

V9_FORMULA = ("HR_Pre_10m_Avg ~ Within_CO2 + Between_CO2 + Steps_Pre_10m_Sum"
              " + Within_Temperature + Between_Temperature + Hour_of_Day")
AWAY_STATES = {"Done for the day!", "Break", "Lunch / Meal", "Errand", "Change workspace"}


def load_v9_rows():
    """Rebuild the exact 76,903-row input used for Table 5."""
    df = pd.read_csv(DATA_DIR / "Master_Cozie_Centered.csv", low_memory=False)
    df["Datetime"] = pd.to_datetime(df["index"], format="mixed", utc=True).dt.tz_convert("Europe/Helsinki")
    df["Hour_of_Day"] = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
    need = ["HR_Pre_10m_Avg", "Within_CO2", "Between_CO2", "Steps_Pre_10m_Sum",
            "Within_Temperature", "Between_Temperature", "Hour_of_Day", "ID"]
    df = df.dropna(subset=need)
    keep = df.groupby("ID").size().loc[lambda s: s >= 50].index
    df = df[df.ID.isin(keep)].sort_values(["ID", "Datetime"]).reset_index(drop=True)
    df["hour"] = df.Datetime.dt.hour
    df["weekday"] = df.Datetime.dt.dayofweek
    return df


def to_blocks(rows):
    """Replicates the V9 block construction (mean of all rows in each 30-min clock bin)."""
    b = (rows.groupby(["ID", pd.Grouper(key="Datetime", freq="30min")])
             .mean(numeric_only=True).reset_index())
    return b.dropna(subset=["HR_Pre_10m_Avg", "Within_CO2", "Steps_Pre_10m_Sum", "Within_Temperature"])


def fit_ri(blocks):
    r = smf.mixedlm(V9_FORMULA, blocks, groups=blocks.ID).fit(reml=True)
    return r.params["Within_CO2"] * 100, r.bse["Within_CO2"] * 100, r.pvalues["Within_CO2"]


def check_1_reproduce_and_subsets(rows):
    lines = []
    subsets = {
        "A. V9 as published (all hours, all days)": rows,
        "B. Night only (00:00 to 07:00)": rows[rows.hour < 7],
        "C. Weekdays 09:00 to 17:00": rows[(rows.weekday < 5) & (rows.hour.between(9, 16))],
    }
    c = subsets["C. Weekdays 09:00 to 17:00"]
    c = c[c.Is_Recovery_Halo == 0]
    subsets["D. C plus recovery filter applied"] = c
    subsets["E. D plus last Cozie status not an away state"] = c[~c.Last_Activity_Context.isin(AWAY_STATES)]
    for label, sub in subsets.items():
        b = to_blocks(sub)
        est, se, p = fit_ri(b)
        lines.append(dict(subset=label, blocks=len(b), participants=b.ID.nunique(),
                          beta_per_100ppm=round(est, 3), se=round(se, 3), p=float(f"{p:.3g}")))
    out = pd.DataFrame(lines)
    out.to_csv(OUT / "check1_subsets.csv", index=False)
    comp = pd.DataFrame({
        "rows_in_V9_input": [len(rows)],
        "rows_00_to_07": [(rows.hour < 7).sum()],
        "rows_weekend": [(rows.weekday >= 5).sum()],
        "rows_recovery_flag_not_removed": [int(rows.Is_Recovery_Halo.sum())],
        "rows_after_done_for_the_day": [(rows.Last_Activity_Context == "Done for the day!").sum()],
    })
    comp.to_csv(OUT / "check1_composition.csv", index=False)
    return out, comp


def check_2_random_slope_and_bootstrap(rows, n_boot=500, seed=1):
    b = to_blocks(rows).copy()
    b["co2w"] = b.Within_CO2 / 100
    b["co2b"] = (b.Between_CO2 - b.Between_CO2.mean()) / 100
    f = ("HR_Pre_10m_Avg ~ co2w + co2b + Steps_Pre_10m_Sum + Within_Temperature"
         " + Between_Temperature + Hour_of_Day")
    rs = smf.mixedlm(f, b, groups=b.ID, re_formula="~co2w").fit(reml=True, method=["lbfgs", "powell"])
    t = rs.params["co2w"] / rs.bse["co2w"]
    res = {"random_slope_beta_per_100ppm": rs.params["co2w"], "se": rs.bse["co2w"],
           "p_z": rs.pvalues["co2w"], "p_t_df18": 2 * stats.t.sf(abs(t), b.ID.nunique() - 1),
           "slope_sd_per_100ppm": float(np.sqrt(rs.cov_re.loc["co2w", "co2w"]))}

    rng = np.random.default_rng(seed)
    ids = b.ID.unique()
    est = []
    for _ in range(n_boot):
        draw = rng.choice(ids, len(ids), replace=True)
        bb = pd.concat([b[b.ID == s].assign(ID=f"{s}_{k}") for k, s in enumerate(draw)])
        try:
            est.append(smf.mixedlm(f, bb, groups=bb.ID).fit(reml=True).params["co2w"])
        except Exception:
            pass
    est = np.asarray(est)
    res.update({"bootstrap_reps": len(est), "bootstrap_ci_low": np.percentile(est, 2.5),
                "bootstrap_ci_high": np.percentile(est, 97.5), "bootstrap_share_negative": (est < 0).mean()})
    pd.Series(res).to_csv(OUT / "check2_random_slope_bootstrap.csv")

    per = []
    for pid, g in b.groupby("ID"):
        row = {"ID": pid, "blocks": len(g), "mean_co2_in_blocks": g.Between_CO2.iloc[0]}
        if len(g) >= 20:
            m = smf.ols("HR_Pre_10m_Avg ~ co2w + Steps_Pre_10m_Sum + Within_Temperature + Hour_of_Day", g).fit()
            row.update(slope_per_100ppm=m.params["co2w"], se=m.bse["co2w"])
        per.append(row)
    per = pd.DataFrame(per).sort_values("slope_per_100ppm")
    per.to_csv(OUT / "check2_participant_slopes.csv", index=False)
    return res, per


def check_3_perceived_air_quality():
    m = pd.read_csv(DATA_DIR / "Master_Morning_Centered.csv")
    m["co2w"] = m.Within_CO2 / 100
    m["co2b"] = (m.Between_CO2 - m.Between_CO2.mean()) / 100
    d = m.dropna(subset=["Baseline_Subj_AQ", "co2w"])
    f = "Baseline_Subj_AQ ~ co2w + co2b + Within_Temperature + Between_Temperature"
    ri = smf.mixedlm(f, d, groups=d.ID).fit()
    rs = smf.mixedlm(f, d, groups=d.ID, re_formula="~co2w").fit(method=["lbfgs", "powell"])
    out = pd.DataFrame([
        dict(model="Morning perceived AQ, random intercept", n=len(d), N=d.ID.nunique(),
             beta_within_per_100ppm=ri.params["co2w"], se=ri.bse["co2w"], p=ri.pvalues["co2w"],
             beta_between_per_100ppm=ri.params["co2b"], p_between=ri.pvalues["co2b"]),
        dict(model="Morning perceived AQ, random slope", n=len(d), N=d.ID.nunique(),
             beta_within_per_100ppm=rs.params["co2w"], se=rs.bse["co2w"], p=rs.pvalues["co2w"],
             beta_between_per_100ppm=rs.params["co2b"], p_between=rs.pvalues["co2b"]),
    ])
    out.to_csv(OUT / "check3_perceived_air_quality.csv", index=False)
    return out


if __name__ == "__main__":
    rows = load_v9_rows()
    print(f"V9 input rows: {len(rows)}; participants: {rows.ID.nunique()}")
    s, comp = check_1_reproduce_and_subsets(rows)
    print("\nCheck 1. Row composition of the V9 physiological input\n", comp.T.to_string(header=False))
    print("\nCheck 1. Within-person CO2 estimate by data subset (V9 model, random intercept)\n", s.to_string(index=False))
    res, per = check_2_random_slope_and_bootstrap(rows)
    print("\nCheck 2. Random slope and participant cluster bootstrap")
    for k, v in res.items():
        print(f"  {k:32s} {v:.4g}")
    print("\nCheck 2. Participant-specific OLS slopes (per 100 ppm)\n", per.round(3).to_string(index=False))
    print("\nCheck 3. Morning perceived air quality versus measured CO2\n",
          check_3_perceived_air_quality().round(4).to_string(index=False))
    print(f"\nAll tables written to {OUT}")

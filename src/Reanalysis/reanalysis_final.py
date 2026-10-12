"""
Final reanalysis for the revised manuscript.

I run this from the repository root with
    python -m src.Reanalysis.reanalysis_final
or through main.py by choosing option 2.

The script reads only the three files in Processed_Database/ and does not modify
any existing file. Every table needed for the revised paper is written to
Generated_Images/reanalysis_final/.

Analyses
    1. Heart rate against within-person CO2 (this replaces Table 5)
    2. Robustness through leave-one-out models and participant-specific slopes
       (this replaces the text in Section 3.2)
    3. Perceived air quality against measured CO2 (new; this addresses RQ2)
    4. Subjective outcome models, reported in full (this replaces Table 6)

I kept all decision rules as constants in the RULES block below, so that the
Methods section describes exactly what the code does.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from config import DATA_DIR, OUTPUT_IMAGE_DIR

warnings.filterwarnings("ignore")

# =============================================================================
# RULES
# =============================================================================
# These are the decision rules behind the analysis. Each one is described in the Methods.
TZ = "Europe/Helsinki"
WORK_START_HOUR = 9           # occupied working time is weekdays from 09:00 to 16:59 local time
WORK_END_HOUR = 17
PRESENT_CONTEXT = "Skip"      # the latest Cozie event was a logged return to the workspace
BLOCK = "30min"               # non-overlapping clock blocks, so neighbouring observations share no data
MIN_BLOCKS_PER_PERSON = 10    # participants with fewer occupied blocks are left out of Analysis 1
CO2_SCALE = 100.0             # every CO2 coefficient is expressed per 100 ppm
N_BOOT = 1000                 # number of bootstrap replicates, resampling whole participants
SEED = 2026                   # fixed so that the bootstrap gives the same result on every run

OUT = OUTPUT_IMAGE_DIR / "reanalysis_final"
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def centre(df, col, new_prefix, scale=1.0):
    """Person-mean centring on the analysed sample (Mundlak approach). The _w column
    holds each observation's deviation from that person's own mean, and the _b column
    holds the person's mean relative to the sample average."""
    person_mean = df.groupby("ID")[col].transform("mean")
    df[f"{new_prefix}_w"] = (df[col] - person_mean) / scale
    df[f"{new_prefix}_b"] = (person_mean - person_mean.mean()) / scale
    return df


def fit_mixed(formula, data, slope=None):
    """Fits a random intercept model, or a random intercept and slope model when
    `slope` is given. If the preferred optimisers fail, I fall back to the
    statsmodels default."""
    kw = dict(groups=data["ID"])
    if slope:
        kw["re_formula"] = f"~{slope}"
    model = smf.mixedlm(formula, data, **kw)
    try:
        res = model.fit(reml=True, method=["lbfgs", "powell"])
    except Exception:
        res = model.fit(reml=True)
    return res


def coef_row(res, term, data, label, extra=None):
    """Collects one coefficient into a table row. Besides the usual z based p value,
    I also report a p value from a t distribution with N minus 1 degrees of freedom,
    where N is the number of participants, because the number of participants is small."""
    n_groups = data["ID"].nunique()
    b, se = res.params[term], res.bse[term]
    t = b / se
    row = {
        "model": label, "term": term, "beta": b, "se": se,
        "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se, "p_z": res.pvalues[term],
        "p_t_df_N_minus_1": 2 * stats.t.sf(abs(t), n_groups - 1),
        "n_obs": len(data), "n_participants": n_groups,
        "converged": bool(getattr(res, "converged", True)),
    }
    if extra:
        row.update(extra)
    return row


def variance_summary(res, data):
    """Variance components and the Nakagawa and Schielzeth R2 for a random intercept
    model. The marginal R2 reflects the fixed effects alone, while the conditional R2
    also includes the variance between participants."""
    fe = res.model.exog @ res.fe_params.values
    v_fixed = np.var(fe)
    v_group = float(res.cov_re.iloc[0, 0])
    v_resid = float(res.scale)
    total = v_fixed + v_group + v_resid
    return {"var_participant": v_group, "var_residual": v_resid,
            "R2_marginal": v_fixed / total, "R2_conditional": (v_fixed + v_group) / total}


def cluster_bootstrap(data, formula, term, n_boot=N_BOOT, seed=SEED):
    """Bootstrap that resamples whole participants with replacement. A participant
    drawn more than once receives a new ID for each copy, so the copies are treated
    as separate people. Replicates that fail to fit are skipped, and boot_reps
    records how many succeeded."""
    rng = np.random.default_rng(seed)
    ids = data["ID"].unique()
    est = []
    for _ in range(n_boot):
        draw = rng.choice(ids, len(ids), replace=True)
        boot = pd.concat([data[data.ID == pid].assign(ID=f"{pid}#{k}") for k, pid in enumerate(draw)])
        try:
            est.append(smf.mixedlm(formula, boot, groups=boot["ID"]).fit(reml=True).params[term])
        except Exception:
            continue
    est = np.asarray(est)
    return {"boot_reps": len(est), "boot_ci_low": np.percentile(est, 2.5),
            "boot_ci_high": np.percentile(est, 97.5), "boot_share_below_zero": float((est < 0).mean())}


def first_digit(series):
    """The survey answers begin with their numeric score, so I take the first
    character and convert it to a number."""
    return pd.to_numeric(series.astype(str).str.strip().str[0], errors="coerce")


def find_col(df, text):
    """Returns the first column whose name contains the given text. The survey
    questions are long, so I match on a short part of each one."""
    return next(c for c in df.columns if text in c)


# =============================================================================
# Data preparation
# =============================================================================
def load_cozie():
    """Loads the Cozie data, converts the timestamps to Helsinki time, gives the main
    columns short names, and keeps only rows where heart rate, CO2, temperature and
    steps are all available."""
    df = pd.read_csv(DATA_DIR / "Master_Cozie_Centered.csv", low_memory=False)
    df["t"] = pd.to_datetime(df["index"], format="mixed", utc=True).dt.tz_convert(TZ)
    df = df.rename(columns={"30m_Avg_CO2 ppm": "co2", "30m_Avg_Temperature °C": "temp",
                            "HR_Pre_10m_Avg": "hr", "Steps_Pre_10m_Sum": "steps"})
    return df.dropna(subset=["hr", "co2", "temp", "steps"])


def make_blocks(rows):
    """Averages the rows into 30 minute blocks for each participant. hour_c is the
    time of day centred on 13:00."""
    b = (rows.groupby(["ID", pd.Grouper(key="t", freq=BLOCK)])[["hr", "co2", "temp", "steps"]]
             .mean().reset_index())
    b["hour_c"] = b["t"].dt.hour + b["t"].dt.minute / 60 - 13
    return b


def physiological_datasets(df):
    # The V9 dataset uses all rows at all hours, the same as physiological_lmm.py,
    # so that I can reproduce the earlier result.
    v9 = make_blocks(df)
    # The primary dataset keeps only occupied working time: weekdays within working
    # hours, outside the recovery halo, and while the participant was at the desk.
    hour = df["t"].dt.hour
    occ = df[(df["t"].dt.dayofweek < 5) & (hour >= WORK_START_HOUR) & (hour < WORK_END_HOUR)
             & (df["Is_Recovery_Halo"] == 0) & (df["Last_Activity_Context"] == PRESENT_CONTEXT)]
    prim = make_blocks(occ)
    counts = prim.groupby("ID").size()
    excluded = counts[counts < MIN_BLOCKS_PER_PERSON]
    prim = prim[prim.ID.isin(counts[counts >= MIN_BLOCKS_PER_PERSON].index)].copy()
    for d in (v9, prim):
        centre(d, "co2", "co2", CO2_SCALE)
        centre(d, "temp", "temp")
    return v9, prim, excluded


# =============================================================================
# Analysis 1 and 2: heart rate
# =============================================================================
# The heart rate model used throughout: within and between person CO2 and
# temperature, steps, and a quadratic term for time of day.
HR_FORMULA = "hr ~ co2_w + co2_b + temp_w + temp_b + steps + hour_c + I(hour_c**2)"


def analysis_heart_rate(v9, prim, excluded):
    """Fits the heart rate model on both datasets, with and without a random CO2
    slope, and adds a participant bootstrap of the random intercept estimate on the
    primary data."""
    rows = []
    for label, data in [("V9 data (all hours), random intercept", v9),
                        ("V9 data (all hours), random slope", v9),
                        ("Occupied working time, random intercept", prim),
                        ("Occupied working time, random slope (PRIMARY)", prim)]:
        slope = "co2_w" if "random slope" in label else None
        res = fit_mixed(HR_FORMULA, data, slope)
        extra = {}
        if slope:
            extra["slope_sd_between_participants"] = float(np.sqrt(res.cov_re.loc["co2_w", "co2_w"]))
        rows.append(coef_row(res, "co2_w", data, label, extra))
        # For the primary model I also save every fixed effect, not only the CO2 term.
        if label.endswith("(PRIMARY)"):
            full = pd.DataFrame({"beta": res.fe_params, "se": res.bse[res.fe_params.index],
                                 "p_z": res.pvalues[res.fe_params.index]})
            full["ci_low"] = full.beta - 1.96 * full.se
            full["ci_high"] = full.beta + 1.96 * full.se
            full.to_csv(OUT / "A1_primary_model_all_terms.csv")
    table = pd.DataFrame(rows)
    table = pd.concat([table, pd.DataFrame([{
        "model": "Occupied working time, participant cluster bootstrap of random intercept estimate",
        **cluster_bootstrap(prim, HR_FORMULA, "co2_w")}])], ignore_index=True)
    table.to_csv(OUT / "A1_heart_rate_models.csv", index=False)

    # To judge practical significance, I compare the CO2 effect with the typical
    # within-person variation in block heart rate.
    within_sd = (prim.hr - prim.groupby("ID").hr.transform("mean")).std()
    # A short description of each participant's data in the primary sample.
    desc = prim.groupby("ID").agg(blocks=("hr", "size"), days=("t", lambda s: s.dt.date.nunique()),
                                  mean_co2=("co2", "mean"), sd_co2=("co2", "std"),
                                  mean_hr=("hr", "mean")).round(1)
    desc.to_csv(OUT / "A1_participant_blocks.csv")
    pd.Series({"within_person_sd_block_hr_bpm": within_sd,
               "participants_excluded_below_min_blocks": ", ".join(f"{k} ({v})" for k, v in excluded.items()) or "none",
               "blocks_primary": len(prim), "participants_primary": prim.ID.nunique(),
               "blocks_v9": len(v9), "participants_v9": v9.ID.nunique()}).to_csv(OUT / "A1_sample_summary.csv")
    return table


def analysis_robustness(prim):
    """Two robustness checks on the primary sample. First, I refit the model leaving
    out one participant at a time. Second, I fit a separate regression for each
    participant to see how much the individual CO2 slopes differ."""
    loo = []
    for pid in prim.ID.unique():
        d = prim[prim.ID != pid]
        ri = fit_mixed(HR_FORMULA, d)
        rs = fit_mixed(HR_FORMULA, d, "co2_w")
        loo.append({"participant_removed": pid,
                    "beta_random_intercept": ri.params["co2_w"], "p_random_intercept": ri.pvalues["co2_w"],
                    "beta_random_slope": rs.params["co2_w"], "p_random_slope": rs.pvalues["co2_w"]})
    pd.DataFrame(loo).to_csv(OUT / "A2_leave_one_out.csv", index=False)

    per = []
    for pid, g in prim.groupby("ID"):
        m = smf.ols("hr ~ co2_w + temp_w + steps + hour_c", g).fit()
        per.append({"ID": pid, "blocks": len(g), "slope_per_100ppm": m.params["co2_w"],
                    "ci_low": m.conf_int().loc["co2_w", 0], "ci_high": m.conf_int().loc["co2_w", 1]})
    per = pd.DataFrame(per).sort_values("slope_per_100ppm")
    per.to_csv(OUT / "A2_participant_slopes.csv", index=False)
    return per


# =============================================================================
# Analysis 3: perceived air quality
# =============================================================================
def analysis_perceived_aq():
    """Tests whether morning perceived air quality follows the measured CO2, both
    within and between participants."""
    m = pd.read_csv(DATA_DIR / "Master_Morning_Centered.csv", low_memory=False)
    m["t"] = pd.to_datetime(m["Timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    m["hour_c"] = m["t"].dt.hour + m["t"].dt.minute / 60 - 13
    m = m.rename(columns={"30m_Avg_CO2 ppm": "co2", "30m_Avg_Temperature °C": "temp"})
    d = m.dropna(subset=["Baseline_Subj_AQ", "co2", "temp"]).copy()
    centre(d, "co2", "co2", CO2_SCALE)
    centre(d, "temp", "temp")
    f = "Baseline_Subj_AQ ~ co2_w + co2_b + temp_w + temp_b + hour_c"
    rows = []
    ri = fit_mixed(f, d)
    rs = fit_mixed(f, d, "co2_w")
    for res, lab in [(ri, "Morning perceived AQ, random intercept"), (rs, "Morning perceived AQ, random slope")]:
        rows.append(coef_row(res, "co2_w", d, lab))
        rows.append(coef_row(res, "co2_b", d, lab))
    boot_w = cluster_bootstrap(d, f, "co2_w")
    boot_b = cluster_bootstrap(d, f, "co2_b")
    rows.append({"model": "Morning perceived AQ, bootstrap", "term": "co2_w", **boot_w})
    rows.append({"model": "Morning perceived AQ, bootstrap", "term": "co2_b", **boot_b})

    # The evening survey asks how the air feels now compared with the morning, so I
    # relate that answer to the change in CO2 between the two surveys.
    e = pd.read_csv(DATA_DIR / "Master_Evening_Centered.csv", low_memory=False)
    pairs = match_morning(e, m)
    pairs["dco2"] = (pairs["30m_Avg_CO2 ppm"] - pairs["co2_morning"]) / CO2_SCALE
    p = pairs.dropna(subset=["Subj_AQ_Relative", "dco2"])
    rr = fit_mixed("Subj_AQ_Relative ~ dco2", p)
    rows.append(coef_row(rr, "dco2", p, "Evening relative AQ versus CO2 change since morning survey"))
    pd.DataFrame(rows).to_csv(OUT / "A3_perceived_air_quality.csv", index=False)
    return pd.DataFrame(rows)


def match_morning(e, m):
    """Pairs each evening survey with the closest earlier morning survey from the
    same participant, provided it falls within 18 hours (the Level 2 rule)."""
    e = e.copy()
    e["t"] = pd.to_datetime(e["Timestamp"], unit="s", utc=True)
    mm = m[["ID", "Timestamp", "co2"]].rename(columns={"Timestamp": "ts_morning", "co2": "co2_morning"})
    mm = mm.assign(t_m=pd.to_datetime(mm.ts_morning, unit="s", utc=True)).sort_values("t_m")
    e = e.sort_values("t")
    out = pd.merge_asof(e, mm, left_on="t", right_on="t_m", by="ID", direction="backward",
                        tolerance=pd.Timedelta(hours=18))
    out["interval_h"] = (out["t"] - out["t_m"]).dt.total_seconds() / 3600
    return out


# =============================================================================
# Analysis 4: subjective outcomes
# =============================================================================
def analysis_subjective():
    """Models the subjective outcomes from the morning and evening surveys, reporting
    every term together with the variance components."""
    m = pd.read_csv(DATA_DIR / "Master_Morning_Centered.csv", low_memory=False)
    e = pd.read_csv(DATA_DIR / "Master_Evening_Centered.csv", low_memory=False)
    m = m.rename(columns={"30m_Avg_CO2 ppm": "co2", "30m_Avg_Temperature °C": "temp"})
    m["t"] = pd.to_datetime(m["Timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    m["hour_c"] = m["t"].dt.hour + m["t"].dt.minute / 60 - 13
    m["focus"] = first_digit(m[find_col(m, "Rate your focus level")])
    m["relax_m"] = first_digit(m[find_col(m, "How relaxed or calm")])
    m["good_sleep"] = pd.to_numeric(m["Control_Good_Sleep"], errors="coerce")
    # The symptom score counts how many of the five listed symptoms were reported.
    sym_cols = [c for c in m.columns if c.startswith(" Any immediate symptoms") or c.startswith("Any immediate symptoms")]
    sym_cols = [c for c in sym_cols if any(k in c for k in ["Dry throat", "Dry eyes", "Headache", "Stuffy nose", "Cough"])]
    m["symptoms"] = sum((~m[c].astype(str).str.strip().isin(["0", "nan", "", "False"])).astype(int) for c in sym_cols)

    e = match_morning(e, m.assign(co2=m["co2"]))
    e["productivity"] = first_digit(e[find_col(e, "Rate your overall productivity")])
    e["relax_e"] = first_digit(e[find_col(e, "How relaxed or calm")])
    e = e.merge(m[["ID", "Timestamp", "relax_m"]].rename(columns={"Timestamp": "ts_morning"}),
                on=["ID", "ts_morning"], how="left")
    # Negative screen time values are recording errors, so I set them to missing
    # and keep a count of how many there were.
    screen = pd.to_numeric(e["Workday_Screen_Time_Added"], errors="coerce")
    invalid = int((screen < 0).sum())
    screen[screen < 0] = np.nan
    e["screen_rate"] = screen / 60 / e["interval_h"]          # hours of phone use per hour between the two surveys
    e.loc[(e.interval_h <= 0) | e.interval_h.isna(), "screen_rate"] = np.nan
    e = e.rename(columns={"Dynamic_Avg_CO2 ppm": "wco2", "Dynamic_Avg_Temperature °C": "wtemp"})

    # Each entry gives the label, data, outcome, formula and the variables to centre.
    specs = [
        ("Evening productivity ~ workday CO2", e, "productivity",
         "productivity ~ wco2_w + wco2_b + wtemp_w + wtemp_b", ["wco2", "wtemp"]),
        ("Evening productivity ~ workday CO2 + phone use rate", e, "productivity",
         "productivity ~ wco2_w + wco2_b + wtemp_w + wtemp_b + scr_w + scr_b", ["wco2", "wtemp", "scr"]),
        ("Evening relaxation ~ workday CO2, adjusted for morning relaxation", e, "relax_e",
         "relax_e ~ wco2_w + wco2_b + wtemp_w + wtemp_b + relax_m", ["wco2", "wtemp"]),
        ("Morning focus ~ pre-survey CO2 + good sleep", m, "focus",
         "focus ~ co2_w + co2_b + temp_w + temp_b + good_sleep + hour_c", ["co2", "temp"]),
        ("Morning relaxation ~ pre-survey CO2 + good sleep", m, "relax_m",
         "relax_m ~ co2_w + co2_b + temp_w + temp_b + good_sleep + hour_c", ["co2", "temp"]),
    ]
    # Maps each centred prefix to its source column and scale.
    source = {"wco2": ("wco2", CO2_SCALE), "wtemp": ("wtemp", 1.0), "scr": ("screen_rate", 1.0),
              "co2": ("co2", CO2_SCALE), "temp": ("temp", 1.0)}
    rows = []
    for label, base, outcome, formula, centred in specs:
        needed = [outcome] + [source[c][0] for c in centred]
        needed += [v for v in ["relax_m", "good_sleep", "hour_c"] if v in formula]
        d = base.dropna(subset=needed).copy()
        for c in centred:
            centre(d, source[c][0], c, source[c][1])
        res = fit_mixed(formula, d)
        vs = variance_summary(res, d)
        for term in res.fe_params.index:
            if term == "Intercept":
                continue
            rows.append(coef_row(res, term, d, label, vs))

    # The symptom score is a count, so I use a Poisson GEE with exchangeable
    # correlation instead of a linear mixed model.
    d = m.dropna(subset=["symptoms", "co2", "temp", "good_sleep"]).copy()
    centre(d, "co2", "co2", CO2_SCALE)
    centre(d, "temp", "temp")
    g = smf.gee("symptoms ~ co2_w + co2_b + temp_w + temp_b + good_sleep", groups=d["ID"], data=d,
                family=sm.families.Poisson(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    for term in ["co2_w", "co2_b", "temp_w", "temp_b", "good_sleep"]:
        ci = g.conf_int().loc[term]
        rows.append({"model": "Morning symptom count (Poisson GEE; beta is log rate ratio)", "term": term,
                     "beta": g.params[term], "se": g.bse[term], "ci_low": ci[0], "ci_high": ci[1],
                     "p_z": g.pvalues[term], "n_obs": len(d), "n_participants": d.ID.nunique()})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "A4_subjective_models.csv", index=False)
    pd.Series({"invalid_negative_screen_time_records_set_missing": invalid,
               "median_morning_to_evening_interval_h": e.interval_h.median(),
               "evening_surveys_with_matched_morning": int(e.ts_morning.notna().sum()),
               "evening_surveys_total": len(e)}).to_csv(OUT / "A4_data_notes.csv")
    return table


# =============================================================================
if __name__ == "__main__":
    cozie = load_cozie()
    v9, prim, excluded = physiological_datasets(cozie)
    fmt = lambda df: df.round(4).to_string(index=False)
    print("ANALYSIS 1. Heart rate versus within-person CO2 (per 100 ppm)")
    print(fmt(analysis_heart_rate(v9, prim, excluded)))
    print("\nANALYSIS 2. Participant-specific slopes (per 100 ppm)")
    print(fmt(analysis_robustness(prim)))
    print("\nANALYSIS 3. Perceived air quality (CO2 per 100 ppm)")
    print(fmt(analysis_perceived_aq()))
    print("\nANALYSIS 4. Subjective outcomes")
    print(fmt(analysis_subjective()))
    print(f"\nAll tables written to {OUT}")
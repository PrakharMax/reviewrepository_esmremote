"""
Figures for the revised manuscript , drawn from my reanalysis output tables.

Run from the repository root, after reanalysis_final.py:
    python -m src.Reanalysis.make_figures

Writes Fig3_participant_slopes.png, FigA3_primary_model_diagnostics.png,
FigA1_environmental_variance_occupied.png, and the A5 and A6 tables
to Generated_Images/reanalysis_final/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from config import OUTPUT_IMAGE_DIR

OUT = OUTPUT_IMAGE_DIR / "reanalysis_final"


def figure3_participant_slopes():
    # Sorting these so we can clearly see the extreme individual differences I found in the reanalysis
    per = pd.read_csv(OUT / "A2_participant_slopes.csv").sort_values("slope_per_100ppm", ascending=False)
    models = pd.read_csv(OUT / "A1_heart_rate_models.csv")

    # Grabbing the new pooled estimate (+0.16) since I dropped the old rolling-window GEE
    pooled = models[models.model.str.contains("PRIMARY")].iloc[0]

    n = len(per)
    # Fixed the missing asterisk from the earlier diff
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * (n + 2) + 1.0))
    y = list(range(2, n + 2))

    # Highlighting which individual slopes are actually significant (CI excludes zero)
    excludes_zero = (per.ci_low > 0) | (per.ci_high < 0)
    colours = ["black" if e else "0.65" for e in excludes_zero]

    ax.hlines(y, per.ci_low, per.ci_high, colors=colours, linewidth=1.6)
    ax.scatter(per.slope_per_100ppm, y, c=colours, s=22, zorder=3)

    # Plotting my pooled estimate at the bottom
    ax.hlines(0.6, pooled.ci_low, pooled.ci_high, colors="#1f5fbf", linewidth=2.6)
    ax.scatter([pooled.beta], [0.6], marker="D", color="#1f5fbf", s=40, zorder=3)
    ax.axvline(0, color="0.3", linewidth=0.9)
    ax.axhline(1.3, color="0.85", linewidth=0.8)

    ax.set_yticks([0.6] + y)
    ax.set_yticklabels(["Pooled estimate"] + [s.replace("_", " ") for s in per.ID], fontsize=8)
    ax.get_yticklabels()[0].set_color("#1f5fbf")
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(0, n + 2.8)
    ax.set_xlabel("Change in block-level heart rate (bpm) per 100 ppm within-person CO$_2$", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "Fig3_participant_slopes.png", dpi=300)
    plt.close(fig)


def figure_a3_diagnostics():
    """Residual diagnostics and VIFs for my new primary heart rate model (LMM with random intercepts/slopes)."""
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from src.Reanalysis.reanalysis_final import HR_FORMULA, fit_mixed, load_cozie, physiological_datasets

    # Building the exact dataset used in the paper
    _, prim, _ = physiological_datasets(load_cozie())
    res = fit_mixed(HR_FORMULA, prim, "co2_w")
    fitted = res.fittedvalues
    resid = res.resid

    # Reviewer 3 wanted robustness checks, so I'm generating the Q-Q and residual plots here
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax1.scatter(osm, osr, s=6, color="0.35")
    ax1.plot(osm, slope * np.asarray(osm) + intercept, color="#1f5fbf", linewidth=1.2)
    ax1.set_xlabel("Theoretical quantiles", fontsize=9)
    ax1.set_ylabel("Residual quantiles (bpm)", fontsize=9)
    ax1.set_title("Normal Q-Q plot of residuals", fontsize=9)

    ax2.scatter(fitted, resid, s=6, color="0.35", alpha=0.6)
    ax2.axhline(0, color="#1f5fbf", linewidth=1.0)
    ax2.set_xlabel("Fitted heart rate (bpm)", fontsize=9)
    ax2.set_ylabel("Residual (bpm)", fontsize=9)
    ax2.set_title("Residuals against fitted values", fontsize=9)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "FigA3_primary_model_diagnostics.png", dpi=300)
    plt.close(fig)

    # Checking for collinearity
    X = prim[["co2_w", "co2_b", "temp_w", "temp_b", "steps", "hour_c"]].copy()
    X["hour_c_sq"] = X.hour_c ** 2
    X = sm.add_constant(X)
    vif = pd.DataFrame({"term": X.columns[1:],
                        "VIF": [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]})
    vif.to_csv(OUT / "A5_vif_primary_model.csv", index=False)
    pd.Series({"residual_skew": stats.skew(resid), "residual_excess_kurtosis": stats.kurtosis(resid),
               "n_blocks": len(resid)}).to_csv(OUT / "A5_residual_summary.csv")
    return vif


def figure_a1_variance_decomposition():
    """Between- and within-person variance shares of the environmental variables in the
    occupied working blocks. I need this to justify my Mundlak centering approach."""
    from src.Reanalysis import reanalysis_final as R

    df = pd.read_csv(R.DATA_DIR / "Master_Cozie_Centered.csv", low_memory=False)
    df["t"] = pd.to_datetime(df["index"], format="mixed", utc=True).dt.tz_convert(R.TZ)
    df = df.rename(columns={"30m_Avg_CO2 ppm": "co2", "30m_Avg_Temperature °C": "temp",
                            "30m_Avg_Humidity %": "hum", "30m_Avg_Noise dB": "noise",
                            "HR_Pre_10m_Avg": "hr", "Steps_Pre_10m_Sum": "steps"})
    df = df.dropna(subset=["hr", "co2", "temp", "steps"])
    hour = df["t"].dt.hour

    # Strict filtering for occupied desk time only
    occ = df[(df["t"].dt.dayofweek < 5) & (hour >= R.WORK_START_HOUR) & (hour < R.WORK_END_HOUR)
             & (df["Is_Recovery_Halo"] == 0) & (df["Last_Activity_Context"] == R.PRESENT_CONTEXT)]
    blocks = (occ.groupby(["ID", pd.Grouper(key="t", freq=R.BLOCK)])[["co2", "temp", "hum", "noise"]]
                 .mean().reset_index())
    counts = blocks.groupby("ID").size()
    blocks = blocks[blocks.ID.isin(counts[counts >= R.MIN_BLOCKS_PER_PERSON].index)]

    labels = {"co2": "CO$_2$", "temp": "Temperature", "hum": "Relative humidity", "noise": "Noise"}
    rows = []

    for col, label in labels.items():
        d = blocks.dropna(subset=[col])
        # Calculating the within-person slice. Added the missing asterisks here.
        within = 100 * (d[col] - d.groupby("ID")[col].transform("mean")).var() / d[col].var()
        rows.append({"variable": label.replace("$_2$", "2"), "pct_between": 100 - within,
                     "pct_within": within, "n_blocks": len(d), "n_participants": d.ID.nunique()})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "A6_environmental_variance_shares.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    names = list(labels.values())[::-1]
    tb = table.iloc[::-1]
    ax.barh(names, tb.pct_between, color="#4C72B0", edgecolor="white", label="Between person (dwelling level)")
    ax.barh(names, tb.pct_within, left=tb.pct_between, color="#C44E52", edgecolor="white",
            label="Within person (fluctuation)")

    for i, (b, w) in enumerate(zip(tb.pct_between, tb.pct_within)):
        ax.text(b / 2, i, f"{b:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
        ax.text(b + w / 2, i, f"{w:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of total variance (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "FigA1_environmental_variance_occupied.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table


def figure2_workday_co2():
    """Drawing the daily workday mean CO2 available for all 20 dwellings."""
    import seaborn as sns
    from src.Reanalysis import reanalysis_final as R

    e = pd.read_csv(R.DATA_DIR / "Master_Evening_Centered.csv", low_memory=False)
    df = e[e["Dynamic_Avg_CO2 ppm"] > 400][["ID", "Dynamic_Avg_CO2 ppm"]].copy()
    df["ID"] = df["ID"].astype(str)
    key = lambda i: (int(i.split("_")[1]), int(i.split("_")[3]))
    order = sorted(df["ID"].unique(), key=key)
    df.groupby("ID")["Dynamic_Avg_CO2 ppm"].agg(["count", "median", "min", "max"]).loc[order].to_csv(
        OUT / "A7_workday_co2_by_dwelling.csv")

    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    sns.boxplot(data=df, x="ID", y="Dynamic_Avg_CO2 ppm", order=order, hue="ID", hue_order=order,
                palette="flare", fliersize=4, legend=False)

    # Reviewers caught my old ASHRAE label, so I changed it to "1000 ppm Reference Level"
    plt.axhline(1000, color="#d62728", linestyle="--", linewidth=2, label="1000 ppm Reference Level")

    plt.title(r"Daily $CO_2$ Exposure Ranges Across Participants", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Anonymized Participant ID", fontsize=12, fontweight="bold")
    plt.ylabel(r"$CO_2$ Concentration (ppm)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=9)
    plt.legend(frameon=True, loc="upper right")
    plt.ylim(bottom=400)
    plt.tight_layout()
    plt.savefig(OUT / "Fig2_workday_co2_by_dwelling.png", dpi=300, bbox_inches="tight")
    plt.close()
    sns.reset_defaults()
    return df.groupby("ID")["Dynamic_Avg_CO2 ppm"].median().loc[order]


def figure_a2_centring_illustration():
    """This perfectly visualizes my within-person centering strategy for the appendix."""
    import numpy as np
    from src.Reanalysis import reanalysis_final as R

    ids = [("Round_2_Cozie_1", "Low-baseline dwelling"),
           ("Round_1_Cozie_2", "Elevated-baseline dwelling"),
           ("Round_2_Cozie_8", "Near-reference dwelling")]
    e = pd.read_csv(R.DATA_DIR / "Master_Evening_Centered.csv", low_memory=False)
    ts = "Timestamp" if "Timestamp" in e.columns else "Helsinki Time"
    e["_t"] = pd.to_datetime(e[ts], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), dpi=300)
    rows = []
    for ax, (pid, label) in zip(axes, ids):
        s = e.loc[e.ID == pid].sort_values("_t")["Dynamic_Avg_CO2 ppm"].dropna().reset_index(drop=True)
        m, sd = s.mean(), s.std()
        x = np.arange(1, len(s) + 1)
        ax.plot(x, s, color="0.35", linewidth=1.4, marker="o", markersize=3, label="Daily workday mean CO$_2$")
        ax.axhline(m, color="black", linestyle="--", linewidth=1.5, label="Participant mean")
        ax.fill_between(x, m, s, where=s > m, interpolate=True, color="#D55E00", alpha=0.45,
                        label="Above participant mean")
        ax.fill_between(x, m, s, where=s <= m, interpolate=True, color="#0072B2", alpha=0.45,
                        label="Below participant mean")
        ax.set_title(f"{label}: {pid} (n = {len(s)} workdays; mean {m:.0f} ppm; SD {sd:.0f} ppm)", fontsize=10)
        ax.set_ylabel("CO$_2$ (ppm)")
        ax.set_xlim(0.5, len(s) + 0.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        rows.append({"ID": pid, "n_workdays": len(s), "mean": m, "sd": sd, "max": s.max()})

    axes[0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[2].set_xlabel("Monitored workday (chronological order)")
    fig.tight_layout()
    fig.savefig(OUT / "FigA2_within_person_centring.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    figure3_participant_slopes()
    print(figure_a2_centring_illustration().round(0).to_string(index=False))
    print(figure_a3_diagnostics().round(2).to_string(index=False))
    print(figure_a1_variance_decomposition().round(1).to_string(index=False))
    print(figure2_workday_co2().round(0).to_string())
    print(f"Figures written to {OUT}")
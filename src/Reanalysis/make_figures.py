"""
Figures for the revised manuscript, drawn from the tables that reanalysis_final.py produces.

I run this from the repository root once reanalysis_final.py has finished:
    python -m src.Reanalysis.make_figures
Option 2 in main.py runs both scripts in the correct order.

The script writes Fig3_participant_slopes.png, FigA3_primary_model_diagnostics.png,
FigA1_environmental_variance_occupied.png, and the A5 and A6 tables to
Generated_Images/reanalysis_final/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_IMAGE_DIR

OUT = OUTPUT_IMAGE_DIR / "reanalysis_final"


def figure3_participant_slopes():
    """Forest plot of the CO2 slope for each participant, with the pooled estimate from
    the primary model shown at the bottom. Participants whose interval excludes zero
    are drawn in black, and the others in grey."""
    per = pd.read_csv(OUT / "A2_participant_slopes.csv").sort_values("slope_per_100ppm", ascending=False)
    models = pd.read_csv(OUT / "A1_heart_rate_models.csv")
    pooled = models[models.model.str.contains("PRIMARY")].iloc[0]

    n = len(per)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * (n + 2) + 1.0))
    y = list(range(2, n + 2))
    excludes_zero = (per.ci_low > 0) | (per.ci_high < 0)
    colours = ["black" if e else "0.65" for e in excludes_zero]

    ax.hlines(y, per.ci_low, per.ci_high, colors=colours, linewidth=1.6)
    ax.scatter(per.slope_per_100ppm, y, c=colours, s=22, zorder=3)
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
    """Residual diagnostics and variance inflation factors for the primary heart rate
    model (random intercept and slope). I refit the model here because the residuals
    are not saved by reanalysis_final.py."""
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from src.Reanalysis.reanalysis_final import HR_FORMULA, fit_mixed, load_cozie, physiological_datasets

    _, prim, _ = physiological_datasets(load_cozie())
    res = fit_mixed(HR_FORMULA, prim, "co2_w")
    fitted = res.fittedvalues
    resid = res.resid

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

    # Variance inflation factors for the fixed effect predictors, to see whether
    # any of them are strongly collinear.
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
    """Shows how much of the variance in each environmental variable lies between
    participants and how much lies within them, using the same occupied working blocks
    as the primary heart rate model. The within share is the variance of the deviations
    from each participant's mean, divided by the total variance."""
    from src.Reanalysis import reanalysis_final as R

    # I repeat the occupancy filter from reanalysis_final.py here, because this figure
    # also needs humidity and noise, and make_blocks there averages only heart rate,
    # CO2, temperature and steps.
    df = pd.read_csv(R.DATA_DIR / "Master_Cozie_Centered.csv", low_memory=False)
    df["t"] = pd.to_datetime(df["index"], format="mixed", utc=True).dt.tz_convert(R.TZ)
    df = df.rename(columns={"30m_Avg_CO2 ppm": "co2", "30m_Avg_Temperature °C": "temp",
                            "30m_Avg_Humidity %": "hum", "30m_Avg_Noise dB": "noise",
                            "HR_Pre_10m_Avg": "hr", "Steps_Pre_10m_Sum": "steps"})
    df = df.dropna(subset=["hr", "co2", "temp", "steps"])
    hour = df["t"].dt.hour
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


if __name__ == "__main__":
    figure3_participant_slopes()
    print(figure_a3_diagnostics().round(2).to_string(index=False))
    print(figure_a1_variance_decomposition().round(1).to_string(index=False))
    print(f"Figures written to {OUT}")
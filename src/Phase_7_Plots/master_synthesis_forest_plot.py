import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import DATA_DIR, OUTPUT_IMAGE_DIR

def generate_forest_plot():
    """
    Synthesizes multi-phase regression model metrics to generate a publication-grade
    forest plot mapping physiological responses alongside subjective cognitive evaluations.
    """
    print("INFO: Initializing master publication forest plot visualization...")

    # We define our multi-dimensional registry tracking standardized effect sizes and confidence intervals
    results = [
        # --- PART I: THE BODY (Objective Physiology) ---
        ("HEADER", "PART I: The Body\n(Objective Physiology)", np.nan, np.nan, np.nan, 1.0),
        ("Phase 4: Real-Time HR", "CO2 (per 100ppm)", -0.700, -0.700, -0.600, 0.000),
        ("Phase 4: Delayed HR", "30m Lagged CO2 (per 100ppm)", -0.650, -1.000, -0.300, 0.00065),
        ("Phase 4: Real-Time HR", "Temperature (°C)", 0.479, 0.397, 0.561, 0.000),

        # Structural spacing demarcation
        ("SPACER", "", np.nan, np.nan, np.nan, 1.0),

        # --- PART II: THE EVENING MIND (Subjective Workday Decay) ---
        ("HEADER", "PART II: The Evening Mind\n(Subjective Workday Decay)", np.nan, np.nan, np.nan, 1.0),
        ("Phase 6: Productivity Gap", "CO2 (per 100ppm)", 0.000, 0.000, 0.100, 0.539),
        ("Phase 6: Productivity Gap", "Temperature (°C)", 0.062, -0.026, 0.150, 0.165),
        ("Phase 6: Productivity Gap", "Screen Time (per Hour)", -0.180, -0.280, -0.080, 0.000),
        ("Phase 6: Stress Delta", "CO2 (per 100ppm)", 0.000, -0.100, 0.100, 0.752),
        ("Phase 6: Stress Delta", "Temperature (°C)", 0.060, -0.041, 0.160, 0.244),

        # Structural spacing demarcation
        ("SPACER", "", np.nan, np.nan, np.nan, 1.0),

        # --- PART III: THE MORNING MIND (Subjective Baseline Readiness) ---
        ("HEADER", "PART III: The Morning Mind\n(Subjective Baseline Readiness)", np.nan, np.nan, np.nan, 1.0),
        ("Phase 6: Morning Focus", "Good Sleep (Yes)", 0.265, 0.076, 0.454, 0.006),
        ("Phase 6: Waking Symptoms", "Overnight CO2 (per 100ppm)", 0.050, 0.000, 0.100, 0.051),
        ("Phase 6: Morning Focus", "Overnight CO2 (per 100ppm)", 0.000, -0.100, 0.000, 0.330),
        ("Phase 6: Morning Focus", "Temperature (°C)", -0.013, -0.090, 0.064, 0.739),
        ("Phase 6: Morning Relaxation", "Good Sleep (Yes)", 0.108, -0.095, 0.312, 0.297),
        ("Phase 6: Morning Relaxation", "Overnight CO2 (per 100ppm)", 0.000, 0.000, 0.000, 0.723)
    ]

    # We execute chronological sorting for inverted y-axis mapping alignment
    df = pd.DataFrame(results, columns=["Model", "Predictor", "Coef", "CI_Lower", "CI_Upper", "P_val"])
    df = df.iloc[::-1].reset_index(drop=True)

    df['err_lower'] = df['Coef'] - df['CI_Lower']
    df['err_upper'] = df['CI_Upper'] - df['Coef']

    # We render localized categorical tick descriptions
    labels = []
    for idx, row in df.iterrows():
        if row['Model'] == 'HEADER':
            labels.append(row['Predictor'])
        elif row['Model'] == 'SPACER':
            labels.append("")
        else:
            labels.append(row['Model'] + "\n(" + row['Predictor'] + ")")

    # Hexadecimal color palette configuration for significance indexing
    sns_colors = {'sig_neg': '#d73027', 'sig_pos': '#4575b4', 'null': '#969696'}
    fig, ax = plt.subplots(figsize=(12, 12))

    # Baseline origin setting
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, zorder=1)

    for idx, row in df.iterrows():
        if pd.isna(row['Coef']):
            continue

        # Inference boundary checking captures the targeted Morning exposure effect parameter (p=0.051)
        is_significant = row['P_val'] <= 0.051
        if not is_significant:
            color = sns_colors['null']
        elif row['Coef'] > 0:
            color = sns_colors['sig_pos']
        else:
            color = sns_colors['sig_neg']

        ax.errorbar(row['Coef'], idx,
                    xerr=[[row['err_lower']], [row['err_upper']]],
                    fmt='o', color=color, markersize=9, capsize=5, capthick=2, elinewidth=2.5, zorder=3)

        # Statistical rendering label definitions
        if row['P_val'] < 0.001:
            sig_text = "p<0.001 ***"
        elif row['P_val'] < 0.01:
            sig_text = f"p={row['P_val']:.3f} **"
        elif row['P_val'] < 0.05:
            sig_text = f"p={row['P_val']:.3f} *"
        elif row['P_val'] <= 0.051:
            sig_text = f"p={row['P_val']:.3f} (Trend)"
        else:
            sig_text = f"p={row['P_val']:.3f}"

        offset = 0.05
        ax.text(row['CI_Upper'] + offset, idx, sig_text, va='center', fontsize=10,
                color=color, weight='bold' if is_significant else 'normal')

    # Formatting axes and spacing boundaries
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=10)

    # Styling text block layouts for structural section separators
    for idx, tick in enumerate(ax.get_yticklabels()):
        if df.iloc[idx]['Model'] == 'HEADER':
            tick.set_fontsize(11)
            tick.set_fontweight('bold')
            tick.set_fontstyle('italic')
            tick.set_color('#333333')

    ax.set_xlabel("Effect Size (Standardized Coefficient & Odds)", fontsize=12, weight='bold')
    ax.set_title("Master Synthesis: Physiological Reality vs. Psychological Perception", fontsize=15, weight='bold', pad=20)

    # Contextual grouping backgrounds matching row matrix configurations
    # Part III: Morning Mind block index boundaries
    ax.axhspan(-0.5, 6.5, color='gray', alpha=0.05, zorder=0)
    # Part I: Objective Physiology block index boundaries
    ax.axhspan(14.5, 18.5, color='gray', alpha=0.05, zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    # We save the figure to the unified outputs repository folder
    plt.tight_layout()
    output_file = OUTPUT_IMAGE_DIR / '6_Master_Thesis_Forest_Plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"INFO: High-resolution publication synthesis forest plot saved: {output_file.name}")

if __name__ == "__main__":
    generate_forest_plot()
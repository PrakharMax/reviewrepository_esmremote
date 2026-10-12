# Indoor CO₂, Heart Rate and Perceived Air Quality in Residential Remote Workspaces: A Longitudinal Within-Person Field Study

> [!IMPORTANT]
> **Repository status.** This repository contains the analysis code only. The dataset is not included, so the pipeline will not run without it. Researchers who need access to the anonymised, processed dataset can request it from the corresponding author (see [Citation and contact](#citation-and-contact)).

This repository contains the analysis pipeline for a longitudinal field study conducted at Aalto University, Finland. The study examined whether indoor CO₂ in residential workspaces was associated with heart rate, perceived air quality and self-reported well-being among remote knowledge workers. Twenty participants were monitored in their own dwellings during winter and early spring 2026 with workspace CO₂ sensors, smartwatch heart rate, logged workspace presence and twice-daily surveys.

Linear mixed-effects models separate within-person variation from stable differences between dwellings. The primary heart rate analysis is restricted to occupied working time, defined as weekdays between 09:00 and 17:00 when the participant's most recent watch log was a return to the workspace, excluding the 15 minutes after each return.

## Main findings

- During occupied working time (16 participants, 1,234 thirty-minute blocks), within-person CO₂ was not reliably associated with heart rate (+0.16 bpm per 100 ppm; 95% CI −0.40 to 0.71). Participant-specific estimates ranged from −3.1 to +2.7 bpm per 100 ppm.
- An inverse association seen when all recorded hours were analysed with random intercepts only did not persist after restriction to occupied time and inclusion of random slopes.
- Participants in dwellings with higher mean CO₂ rated their air quality as worse (−0.13 points per 100 ppm on a five-point scale; 95% CI −0.25 to −0.02). Day-to-day CO₂ variation within dwellings was not reliably reflected in ratings.
- Productivity, relaxation, focus and morning symptoms were not reliably associated with measured CO₂. A reported good night's sleep was associated with higher morning focus and fewer morning symptoms.

## Instrumentation

| Stream | Device or platform | Use in the analysis |
|---|---|---|
| Indoor environment | Netatmo indoor module, fixed at the primary workstation (CO₂, temperature, humidity, noise; 5-minute intervals) | Exposure |
| Heart rate and steps | Apple Watch Series 9 running the Cozie application | Outcome and covariate |
| Workspace presence and events | Cozie smartwatch prompts (leaving and returning to the workspace, symptoms, clothing, ventilation actions, other people present, appliances) | Occupancy classification |
| Daily surveys | Webropol 3.0 (Webropol Oy, Helsinki, Finland), morning and evening on weekdays | Perceived air quality and well-being outcomes |

Cozie: Tartarini, F., Frei, M., Schiavon, S., Chua, Y. X., & Miller, C. (2023). Cozie Apple: An iOS mobile and smartwatch application for environmental quality satisfaction and physiological data collection. *Journal of Physics: Conference Series*, 2600(14), 142003. https://doi.org/10.1088/1742-6596/2600/14/142003

## Data privacy and GDPR

To protect participants' personal data under the GDPR, raw data parsing and merging (Phase 1) is excluded from this repository. The pipeline starts from anonymised, analysis-ready files that would be placed in `Processed_Database/`:

- `Master_Cozie_Centered.csv`: watch transmissions with heart rate, steps, workspace presence and matched indoor environment
- `Master_Morning_Centered.csv`: morning survey responses with matched exposure
- `Master_Evening_Centered.csv`: evening survey responses with matched exposure

These files are not distributed publicly and are available on reasonable request.

## Running the pipeline

Requirements: Python 3 with the packages in `requirements.txt` (pandas, numpy, scipy, statsmodels, matplotlib, seaborn).

```bash
pip install -r requirements.txt
python main.py
```

`main.py` is the single entry point. Run from the repository root, it prints a header and the Python environment in use, switches the console to UTF-8 (so that CO₂ displays correctly on Windows), and asks which analysis to run:

1. **Original analysis:** all recorded hours, one common CO₂ effect, overlapping windows treated as independent. Runs the scripts in Phases 2 to 7 and `Post_analysis`. Kept for transparency and comparison.
2. **Occupancy-based reanalysis:** weekday desk time only, an individual CO₂ slope for each participant, checked with participant-level resampling. Produces every result reported in the current manuscript.

The reanalysis scripts can also be run directly from the repository root:

```bash
python -m src.Reanalysis.reanalysis_final   # all models and tables
python -m src.Reanalysis.make_figures       # figures and diagnostics (run after reanalysis_final)
python -m src.Reanalysis.stage0_audit       # optional: audit of the original analysis
```

### Reanalysis outputs

Written to `Generated_Images/reanalysis_final/`:

| File | Content |
|---|---|
| `A1_heart_rate_models.csv` | Heart rate models: occupied time and all hours, random intercept and random slope, bootstrap interval |
| `A1_primary_model_all_terms.csv` | All coefficients of the primary heart rate model |
| `A1_participant_blocks.csv`, `A1_sample_summary.csv` | Blocks per participant and sample description |
| `A2_leave_one_out.csv`, `A2_participant_slopes.csv` | Leave-one-participant-out estimates and participant-specific slopes |
| `A3_perceived_air_quality.csv` | Perceived air quality against measured CO₂ |
| `A4_subjective_models.csv`, `A4_data_notes.csv` | Productivity, relaxation, focus and symptom models; data notes |
| `A5_vif_primary_model.csv`, `A5_residual_summary.csv` | Collinearity and residual diagnostics |
| `A6_environmental_variance_shares.csv` | Between- and within-person variance shares of CO₂, temperature, humidity and noise |
| `Fig3_participant_slopes.png` | Participant-specific associations between CO₂ and heart rate |
| `FigA1_environmental_variance_occupied.png` | Variance decomposition of the environmental variables |
| `FigA3_primary_model_diagnostics.png` | Residual diagnostics of the primary model |

## Repository structure

```
├── .github/
├── Generated_Images/                  # Figures and tables
│   └── reanalysis_final/              # Outputs of the occupancy-based reanalysis
├── Processed_Database/                # Anonymised analysis-ready files (not included)
├── Questionnaires/                    # Survey instruments
├── src/
│   ├── Phase_2_eda_layer/             # Original analysis: exploratory data analysis
│   ├── Phase_3_icc_phase/             # Original analysis: intraclass correlations
│   ├── Phase_4_LinearMM/              # Original analysis: physiological models
│   ├── Phase_5_psychological_check/   # Original analysis: survey variable distributions
│   ├── Phase_6_psychological_model/   # Original analysis: survey models
│   ├── Phase_7_Plots/                 # Original analysis: figures
│   ├── Post_analysis/                 # Original analysis: synthesis
│   ├── Reanalysis/                    # Occupancy-based reanalysis (current manuscript)
│   │   ├── reanalysis_final.py
│   │   ├── make_figures.py
│   │   └── stage0_audit.py
│   └── __init__.py
├── config.py                          # Path configuration
├── main.py                            # Entry point with analysis menu
├── requirements.txt
└── README.md
```

## Questionnaires

The study used three instruments:

- **Onboarding survey** (once, Webropol): participant characteristics, residential arrangement, building age, ventilation configuration and usual working hours.
- **Daily surveys** (weekdays, Webropol): a morning survey, completed shortly after starting work, covering sleep quality, perceived air quality, mood, focus, relaxation, symptoms and phone screen time; and an evening survey, near the end of the working day, covering productivity, mood, relaxation, symptoms, air quality compared with the morning and phone screen time. Reminders were sent at each participant's stated start and end of work.
- **Smartwatch event reports** (Cozie): participant-initiated logs of leaving and returning to the workspace, new symptoms, clothing changes, ventilation actions, other people present and appliance use.

## Citation and contact

If you use this code, please cite:

> Prakash, P., & Huotari, M. **Indoor CO₂, Heart Rate and Perceived Air Quality in Residential Remote Workspaces: A Longitudinal Within-Person Field Study.** Manuscript under review.

**Prakhar Prakash**, Doctoral Researcher
Department of Electrical Engineering and Automation, Aalto University
Maarintie 8, 02150 Espoo, Finland
Prakhar.Prakash@aalto.fi

This research was funded by the Promotion Centre for Electrical Engineering and Energy Efficiency (STEK ry) under grant KITSI2.0.
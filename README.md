# Indoor CO2 Accumulation in Residential Workspaces: Cardiovascular Effects and Occupant Misattribution During Remote Work

> [!IMPORTANT]
> **Repository Status:** This repository contains the analytical pipeline code but **does not include the required dataset files**. Consequently, the pipeline will fail if executed without the source data. If you are a researcher requiring access to the processed dataset, please contact the corresponding author using the details provided in the "Citation & Contact" section below.

---

This repository contains the complete analytical pipeline and orchestration framework for a longitudinal Experience Sampling Method (ESM) study investigating the relationship between residential Indoor Air Quality (IAQ), continuous physiological metrics (wearables), and remote worker well-being. 

Conducted at Aalto University, this research utilizes high-resolution Netatmo weather station sensors and multi-level mixed-effects modeling (LMM) to separate within-person state deviations from between-person trait baselines.

---

## Instrumentation & Software

The data collection infrastructure relies on a combination of IoT environmental sensors, wearables, and survey platforms to capture high-resolution ESM data:

* **Indoor Environment Monitoring:** Netatmo weather stations.
* **Physiological Response & Event-Contingent Prompts:** Apple Watch Series 9 running the Cozie Application. 
  > *Citation: Tartarini, F., Frei, M., Schiavon, S., Chua, Y. X., & Miller, C. (2023). Cozie Apple: An iOS mobile and smartwatch application for environmental quality satisfaction and physiological data collection. Journal of Physics: Conference Series, 2600(14), 142003. https://doi.org/10.1088/1742-6596/2600/14/142003*
* **Time-Contingent Surveys:** Webropol 3.0 (Webropol Oy, Helsinki, Finland) utilized for recording morning and evening survey responses on weekdays.

---

## Data Privacy and GDPR Compliance

To comply with European GDPR regulations and protect the Personally Identifiable Information (PII) of our human subjects, Phase 1 (Raw Data Parsing & Processing) is strictly locked and excluded from this public repository. The public pipeline begins safely at Phase 2, utilizing fully anonymized, "Analysis-Ready Level-4 Centered" data matrices located in the `Processed_Database` directory. Researchers looking to reproduce this study will run the pipeline against these Level-4 files, ensuring complete methodological transparency without compromising participant privacy.

---

## Repository Architecture

The codebase is modularized into sequential scientific phases. The directory structure below mirrors the project environment.

```text
├── .github/
├── Generated_Images/                 # Output directory for publication figures
├── Processed_Database/               # Houses anonymized Level-4 Centered CSVs
├── src/                              # Source code for analytical pipeline
│   ├── Phase_1.1_centered_merge[LOCKED]/ # GDPR Restricted
│   ├── Phase_1_master_merge[LOCKED]/     # GDPR Restricted
│   ├── Phase_2_eda_layer/                # Exploratory Data Analysis
│   ├── Phase_3_icc_phase/                # Intraclass Correlation (ICC) & Variance math
│   ├── Phase_4_LinearMM/                 # Physiological & Lagged Sedation Modeling
│   ├── Phase_5_psychological_check/      # Psychological variable distributions
│   ├── Phase_6_psychological_model/      # Behavioral & Cognitive Mixed-Effects Models
│   ├── Phase_7_Plots/                    # Final publication-grade graphics engine
│   ├── Post_analysis/                    # Final synthesis and statistical reporting
│   └── __init__.py
├── .gitattributes
├── .gitignore
├── config.py                         # Centralized path configurations
├── main.py                           # Master CLI Orchestrator
└── README.md                         # Project documentation

 requirement.txt has the necessary ackages and version number.
---

## Questionnaires

This section contains the complete three-tier Experience Sampling Method (ESM) instruments deployed during the study. The surveys are structured into the following categories:

* **Baseline Pre-Survey (Onboarding):** Administered once via Webropol prior to the monitoring period. Collects participant characteristics, residential building typology, building age, ventilation configuration, and typical remote work routines.
* **Time-Contingent ESM (Daily Surveys):** Administered twice daily on weekdays via Webropol to capture within-day psychological and physical states.
  * **Morning Assessment (09:00):** Captures perceived sleep quality, baseline cognitive focus/readiness, and physical symptoms upon waking.
  * **Evening Assessment (17:00):** Evaluates perceived daily productivity, mood, perceived stress, and self-reported smartphone screen time.
* **Event-Contingent Micro-Prompts:** Logged actively by participants via the Cozie Apple Watch application. Designed to capture specific state changes and behavioural actions, including workspace entry/exit, acute physical symptoms, visitor arrivals, and manual window opening for ventilation.

---

## Citation & Contact

If you use this pipeline or the associated dataset in your research, please cite the following manuscript:

> **Indoor CO₂ Accumulation in Residential Workspaces: Cardiovascular Effects and Occupant Misattribution During Remote Work**

### Author

**Prakhar Prakash**  
Doctoral Researcher  
Aalto University, Finland

### Affiliation

Aalto University  
Maarintie 8  
02150 Espoo, Finland

### Contact

📧 Prakhar.Prakash@aalto.fi
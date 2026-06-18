# Indoor CO2 Accumulation in Residential Workspaces: Cardiovascular Effects and Occupant Misattribution During Remote Work

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

The codebase is modularized into sequential scientific phases. The directory structure below mirrors the project environment referenced in image_536185.png.

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
```
---
## Processed Database Structure

The `Processed_Database` directory contains the fully anonymized, centered data matrices used to drive the analytical pipeline. This ensures compliance with GDPR while allowing full reproducibility of the mixed-effects models and visualizations. 

The database consists of three primary CSV files:

* **`Master_Morning_Centered.csv`**: Contains the processed responses from the daily morning Webropol surveys. This tracks acute waking symptom counts, subjective morning focus, and self-reported sleep quality.
* **`Master_Evening_Centered.csv`**: Contains the processed responses from the daily evening Webropol surveys. This tracks subjective workday productivity retention gaps, screen time confounding variables, and daily perceived stress.
* **`Master_Cozie_Centered.csv`**: The high-frequency, event-contingent dataset. This integrates continuous physiological responses (e.g., continuous heart rate from wearables).

*Note: All core variables within these datasets have undergone multi-level processing and within-person centering to mathematically isolate state-level daily fluctuations from trait-level participant baselines.*
---

## Configuration Setup

Before running the pipeline, you must configure the central path routing. All scripts dynamically load their input and output destinations from `config.py`.

Create or update `config.py` in the root directory to match your local machine's absolute paths:

```python
# config.py
from pathlib import Path

# The directory containing the Processed Centered CSV files
DATA_DIR = Path("/absolute/path/to/your/Processed_Database")

# The directory where all generated plots, figures, and diagnostics will be saved
OUTPUT_IMAGE_DIR = Path("/absolute/path/to/your/Generated_Images")

# Ensure the output directory exists
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Execution Guide

This repository utilizes a hybrid Command Line Interface (CLI) orchestrator (`main.py`), allowing for both end-to-end reproducibility and surgical phase execution. 

### Option 1: The End-to-End Run
To completely reproduce the study's analytical findings from Phase 2 through Phase 8:

```bash
python main.py
```
*Note: As Phase 1 files have been removed to comply with GDPR protocols, the orchestrator will safely acknowledge this and sequentially execute all downstream scripts, saving the final models and figures to your configured directories.*

### Option 2: Target Specific Phases 
To run a specific phase without triggering the entire architecture, pass the `--phase` argument followed by the phase ID (2 through 8).

```bash
# Example: Generate only the final publication plots
python main.py --phase 7

# Example: Run only the linear mixed models
python main.py --phase 4
```

---

## Pipeline Phase Breakdown

* **Phase 2 (EDA Layer):** Evaluates raw distributions of environmental columns (CO2, Temperature, Humidity) before modeling.
* **Phase 3 (ICC Phase):** Calculates Intraclass Correlation Coefficients to mathematically justify the necessity of within-person centering (Level-4) over standard OLS regression.
* **Phase 4 (Linear MM):** Executes the core physiological mixed-effects models, including the real-time and 30-minute lagged sedation tests on continuous heart rate.
* **Phase 5 (Psychological Check):** Pre-processes and evaluates subjective ESM survey responses.
* **Phase 6 (Psychological Model):** A suite of LMMs testing behavioral interactions, morning cognitive hangovers, screen time confounding, and social presence.
* **Phase 7 (Plots):** The graphics engine. Generates all multi-panel regression plots, forest plots, variance decomposition charts, and statistical diagnostics using standard publication typography.
* **Phase 8 (Post Analysis):** Synthesizes final statistical outputs.

---

## Citation & Contact

If you utilize this pipeline or dataset in your research, please cite the original manuscript:

> **Indoor CO2 Accumulation in Residential Workspaces: Cardiovascular Effects and Occupant Misattribution During Remote Work**  

**Institution:** Aalto University, Department of Electrical Engineering and Automation  (Smart Building Technologies and Services)

**Author:** Prakhar Prakash, prakhar.prakash@aalto.fi, Doctoral Researcher
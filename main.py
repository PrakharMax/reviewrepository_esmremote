import subprocess
import argparse
import sys
import time
import os
from pathlib import Path

# =============================================================================
# PIPELINE ARCHITECTURE DEFINITION
# =============================================================================
# Mapped precisely to the repository's src/ directory structure.
PHASES = {
    2: [
        "src/Phase_2_eda_layer/eda_analysis_env_columns.py"
    ],
    3: [
        "src/Phase_3_icc_phase/icc_layer.py",
        "src/Phase_3_icc_phase/psychological_icc.py"
    ],
    4: [
        "src/Phase_4_LinearMM/forensic_recovery.py",
        "src/Phase_4_LinearMM/lagged_sedation_test.py",
        "src/Phase_4_LinearMM/model_diagnostics.py",
        "src/Phase_4_LinearMM/physiological_lmm.py"
    ],
    5: [
        "src/Phase_5_psychological_check/psychological_eda.py"
    ],
    6: [
        "src/Phase_6_psychological_model/behavioral_interaction_lmm.py",
        "src/Phase_6_psychological_model/environmental_moderation_lmm.py",
        "src/Phase_6_psychological_model/morning_cognitive_hangover_lmm.py",
        "src/Phase_6_psychological_model/morning_wellbeing_lmm.py",
        "src/Phase_6_psychological_model/psychological_mediation_lmm.py",
        "src/Phase_6_psychological_model/screentime_confounder_lmm.py",
        "src/Phase_6_psychological_model/social_presence_confounder_lmm.py",
        "src/Phase_6_psychological_model/subjective_wellbeing_lmm.py"
    ],
    7: [
        "src/Phase_7_Plots/behavioral_perception_plots.py",
        "src/Phase_7_Plots/contextual_co2_boxplot.py",
        "src/Phase_7_Plots/master_synthesis_forest_plot.py",
        "src/Phase_7_Plots/psychological_publication_figures.py",
        "src/Phase_7_Plots/scientific_visuals.py",
        "src/Phase_7_Plots/variance_stats_analysis.py",
        "src/Phase_7_Plots/variance_centering_viz.py"
    ],
    8: [
        "src/Post_analysis/Analysis.py"
    ]
}

def print_header(text, symbol="="):
    """Utility to print clean, professional section headers to the terminal."""
    print(f"\n{symbol * 70}")
    print(f"{text}")
    print(f"{symbol * 70}")


def run_script(script_path):
    """Executes a target Python script in an isolated subprocess memory environment."""
    script_file = Path(script_path)

    if not script_file.exists():
        print(f"[ERROR] FATAL: Script not found at {script_path}")
        print("        Please verify the file path and repository structure.")
        sys.exit(1)

    print(f" -> Executing: {script_file.name} ...")
    start_time = time.time()

    # FIX: Prepare the environment and inject PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    try:
        # Run the script with the custom environment
        subprocess.run([sys.executable, str(script_file)], check=True, env=env)
        elapsed = time.time() - start_time
        print(f"    [SUCCESS] Completed in {elapsed:.2f}s\n")
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] PIPELINE HALTED: Subprocess crashed while executing {script_file.name}")
        sys.exit(1)

def execute_phase(phase_id):
    """Handles the execution logic for a specific phase, including Phase 1 locks."""
    if phase_id == 1:
        print_header("PHASE 1: DATA PROCESSING [LOCKED]", symbol="-")
        print("INFO: Phase 1 (Raw Data Parsing & Level-1 to Level-4 Generation) is locked.")
        print("      Raw ESM and wearable logs contain highly sensitive PII and are withheld")
        print("      to ensure GDPR compliance. The public pipeline begins safely at Phase 2")
        print("      using the anonymized, Analysis-Ready Level-4 Centered matrices.")
        return

    if phase_id not in PHASES:
        print(f"[ERROR] Phase {phase_id} is not defined in the architecture.")
        sys.exit(1)

    print_header(f"PHASE {phase_id} EXECUTION", symbol="-")
    for script in PHASES[phase_id]:
        run_script(script)

def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestration Pipeline for Remote Worker ESM Study",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py            (Runs the entire open-source pipeline: Phases 2-8)\n"
            "  python main.py --phase 7  (Runs ONLY the publication plotting scripts)"
        )
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run a specific phase script by its integer ID."
    )

    args = parser.parse_args()
    total_start_time = time.time()

    print_header("INITIALIZING SCIENTIFIC REPRODUCIBILITY PIPELINE")
    print(f"Environment: {sys.executable}")

    # Execute a single phase if the user requested it via CLI arguments
    if args.phase:
        print(f"INFO: Manual override triggered. Executing target Phase {args.phase} only.")
        execute_phase(args.phase)

    # Execute the entire pipeline end-to-end (The Reviewer Run)
    else:
        print("INFO: Initiating complete end-to-end pipeline execution (Phases 1 through 8).")

        # Explicitly call Phase 1 first to print the GDPR compliance message
        execute_phase(1)

        # Sequentially run the open-source analytical phases
        for phase_id in sorted(PHASES.keys()):
            execute_phase(phase_id)

    total_elapsed = time.time() - total_start_time
    print_header("PIPELINE EXECUTION COMPLETE")
    print(f"Total Architecture Execution Time: {total_elapsed:.2f} seconds.")
    print("All configured analytical scripts have run successfully. Check output directories.")


if __name__ == "__main__":
    main()
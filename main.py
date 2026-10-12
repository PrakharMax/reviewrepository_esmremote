import subprocess
import argparse
import sys
import time
import os
from pathlib import Path

# The Windows console often uses an older code page that cannot print the
# subscript in CO₂, so I switch the output to UTF-8 for the menu text.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# =============================================================================
# PIPELINE ARCHITECTURE
# =============================================================================
# Each phase lists the scripts it runs, following the layout of the src/ folder.

# Option 1 is the original analysis, organised as Phases 2 to 8.
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

# Option 2 is the occupancy-based reanalysis. reanalysis_final.py runs first
# because make_figures.py reads the tables it writes. I leave stage0_audit.py
# out of the pipeline on purpose.
REANALYSIS_SCRIPTS = [
    "src/Reanalysis/reanalysis_final.py",
    "src/Reanalysis/make_figures.py"
]

MENU_TEXT = (
    "Press 1 for Old analysis: All recorded hours, one common CO₂ effect for everyone, "
    "overlapping windows treated as independent.\n"
    "Press 2 for new Occupancy-based reanalysis: Weekday time at the desk only, "
    "an individual CO₂ slope for each participant, verified with participant-level resampling."
)


def print_header(text, symbol="="):
    """Prints a section header so that the terminal output is easy to follow."""
    print(f"\n{symbol * 70}")
    print(f"{text}")
    print(f"{symbol * 70}")


def run_script(script_path):
    """Runs one analysis script in its own Python process, so that memory used by
    one script does not carry over into the next."""
    script_file = Path(script_path)

    if not script_file.exists():
        print(f"[ERROR] FATAL: Script not found at {script_path}")
        print("        Please verify the file path and repository structure.")
        sys.exit(1)

    print(f" -> Executing: {script_file.name} ...")
    start_time = time.time()

    # I add the repository root to PYTHONPATH so that each script can import
    # config and the src package.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    try:
        subprocess.run([sys.executable, str(script_file)], check=True, env=env)
        elapsed = time.time() - start_time
        print(f"    [SUCCESS] Completed in {elapsed:.2f}s\n")
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] PIPELINE HALTED: Subprocess crashed while executing {script_file.name}")
        sys.exit(1)


def execute_phase(phase_id):
    """Runs every script in one phase. Phase 1 only prints a note, because the raw
    data it processes is not part of the public repository."""
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


def run_old_analysis(phase=None):
    """Option 1: the original pipeline, kept exactly as it was before the reanalysis."""
    if phase:
        print(f"INFO: Manual override triggered. Executing target Phase {phase} only.")
        execute_phase(phase)
    else:
        print("INFO: Initiating complete end-to-end pipeline execution (Phases 1 through 8).")

        # I call Phase 1 first so that the GDPR note appears before the analysis starts.
        execute_phase(1)

        # The public analysis phases then run in order.
        for phase_id in sorted(PHASES.keys()):
            execute_phase(phase_id)


def run_reanalysis():
    """Option 2: the occupancy-based reanalysis, followed by the figures built from its output."""
    print("INFO: Initiating occupancy-based reanalysis.")
    print_header("REANALYSIS EXECUTION", symbol="-")
    for script in REANALYSIS_SCRIPTS:
        run_script(script)


def prompt_for_mode():
    """Shows the two analysis options and waits until a valid choice (1 or 2) is entered."""
    print()
    print(MENU_TEXT)
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nINFO: No selection made. Exiting.")
            sys.exit(0)
        if choice in ("1", "2"):
            return int(choice)
        print("[WARNING] Invalid input. Please press 1 or 2.")


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestration Pipeline for Remote Worker ESM Study",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py            (Shows the menu to choose old analysis or reanalysis)\n"
            "  python main.py --mode 1   (Runs the old analysis pipeline without the menu)\n"
            "  python main.py --mode 2   (Runs the occupancy-based reanalysis without the menu)\n"
            "  python main.py --phase 7  (Runs ONLY Phase 7 of the old analysis)"
        )
    )
    parser.add_argument(
        '--mode',
        type=int,
        choices=[1, 2],
        help="1 = old analysis, 2 = occupancy-based reanalysis. Skips the menu."
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run a specific phase of the old analysis by its integer ID."
    )

    args = parser.parse_args()

    if args.phase and args.mode == 2:
        parser.error("--phase applies only to the old analysis (mode 1).")

    total_start_time = time.time()

    print_header("INITIALIZING SCIENTIFIC REPRODUCIBILITY PIPELINE")
    print(f"Environment: {sys.executable}")

    # A phase number only exists in the old analysis, so when --phase is given
    # I skip the menu and go straight to option 1.
    if args.phase:
        mode = 1
    elif args.mode:
        mode = args.mode
    else:
        mode = prompt_for_mode()

    if mode == 1:
        run_old_analysis(args.phase)
    else:
        run_reanalysis()

    total_elapsed = time.time() - total_start_time
    print_header("PIPELINE EXECUTION COMPLETE")
    print(f"Total Architecture Execution Time: {total_elapsed:.2f} seconds.")
    print("All configured analytical scripts have run successfully. Check output directories.")


if __name__ == "__main__":
    main()
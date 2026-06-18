# config.py
import os
from pathlib import Path

# Absolute path to the directory containing this configuration file
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the directory housing the cleaned, anonymized, and centered datasets
DATA_DIR = PROJECT_ROOT / "Processed_Database"

# Destination directory for all generated figures, diagnostic charts, and visual outputs
OUTPUT_IMAGE_DIR = PROJECT_ROOT / "Generated_Images"

# Automated initialization of the directory structure to prevent execution write errors
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
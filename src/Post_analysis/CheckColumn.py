import os
import pandas as pd
from pandas.errors import ParserError

def read_csv_safe(path):
    """Read CSV with fallback to python engine and BOM handling."""
    try:
        return pd.read_csv(path, engine="c", encoding="utf-8-sig")
    except ParserError:
        return pd.read_csv(path, engine="python", encoding="utf-8-sig", on_bad_lines="skip")

def describe_csv(path):
    """Print column names, number of columns, and number of rows."""
    print(f"\n--- File: {path} ---")
    try:
        df = read_csv_safe(path)
        print("Column names:")
        for col in df.columns:
            print(f"  - {col}")

        print(f"Total columns: {len(df.columns)}")
        print(f"Total rows: {len(df)}")

    except Exception as e:
        print(f"Error reading {path}: {e}")

def process_path(input_path):
    """Process either a single CSV file or a folder containing CSVs."""
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".csv"):
            describe_csv(input_path)
        else:
            print("Provided file is not a CSV.")
        return

    if os.path.isdir(input_path):
        csv_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".csv")
        ]

        if not csv_files:
            print("No CSV files found in the folder.")
            return

        for csv in csv_files:
            describe_csv(csv)
        return

    print("Invalid path: neither a file nor a folder.")

def main():
    path = input("Enter CSV file path OR folder path: ").strip()
    process_path(path)

if __name__ == "__main__":
    main()

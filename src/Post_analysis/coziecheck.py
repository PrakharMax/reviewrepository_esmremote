import os
import pandas as pd
from pandas.errors import ParserError

def read_csv_safe(path):
    """Read CSV with robust fallback handling."""
    try:
        return pd.read_csv(path, engine="c", encoding="utf-8-sig")
    except ParserError:
        return pd.read_csv(path, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

def summarize_csv(path):
    """Print column names, total rows, total IDs, and ID record counts."""
    if not os.path.isfile(path):
        print(f"Invalid path: {path}")
        return

    print(f"\n--- File: {path} ---")

    df = read_csv_safe(path)

    # Column names
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    # Total rows
    print(f"\nTotal number of rows: {len(df)}")

    # Primary key counts
    if "ID" not in df.columns:
        print("\nColumn 'ID' not found in the CSV.")
        return

    id_counts = df["ID"].value_counts(dropna=False)

    # Total number of unique IDs
    print(f"\nTotal number of unique IDs: {id_counts.shape[0]}")

    # Record count per ID
    print("\nRecord count per ID:")
    for id_val, count in id_counts.items():
        print(f"  ID = {id_val} : {count} records")

def main():
    path = input("Enter the CSV file path: ").strip()
    summarize_csv(path)

if __name__ == "__main__":
    main()

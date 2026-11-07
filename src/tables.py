"""
Generate the tables presented in the paper from the aggregated results.

Input:
    results/conflict_metrics_all_day.csv

Outputs:
    results/dominant_pair_types.csv
    results/mean_psi_by_pair_type.csv
    results/c_all_vs_c_median_by_direction.csv

Usage:
    python src/tables.py

Note on encoding:
    All CSVs are written and read as UTF-8. This is required because pair
    types contain the Unicode arrow '→' (U+2192). Writing with the default
    locale encoding (e.g. cp1251 on Windows) corrupts the arrow into
    mojibake such as 'â†’' or '‚Üí'.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ALL_DAY_CSV,
    DIRECTION_NAMES,
    TABLE_C_ALL_VS_C_MEDIAN,
    TABLE_DOMINANT_PAIR_TYPES,
    TABLE_MEAN_PSI,
)

# Single source of truth for CSV encoding.
CSV_ENCODING = "utf-8"


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load the aggregated results CSV and parse the psi_frame JSON column."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Aggregated results not found: {csv_path}. "
            "Run `python src/run_all.py` first."
        )
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    if "psi_frame" in df.columns:
        df["psi_frame"] = df["psi_frame"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else {}
        )
    return df


def build_dominant_pair_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: frequency of dominant pair types across all (file, lane) records.
    """
    counts = (
        df["dominant_pair_type"]
        .dropna()
        .value_counts()
        .rename_axis("pair_type")
        .reset_index(name="count")
    )
    return counts


def build_mean_psi_by_pair_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: mean psi for each pair type across all (file, lane) records.

    For each record, psi_frame is a dict {pair_type: psi}. We aggregate
    the psi values per pair type and compute the mean, count, and std.
    """
    records = []
    for psi_dict in df["psi_frame"]:
        if not isinstance(psi_dict, dict):
            continue
        for pair_type, psi in psi_dict.items():
            if psi > 0:
                records.append({"pair_type": pair_type, "psi": float(psi)})

    if not records:
        return pd.DataFrame(columns=["pair_type", "n_records", "mean_psi", "std_psi"])

    psi_df = pd.DataFrame(records)
    agg = (
        psi_df.groupby("pair_type")["psi"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n_records", "mean": "mean_psi", "std": "std_psi"})
        .reset_index()
        .sort_values("mean_psi", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def build_c_all_vs_c_median_by_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: comparison of C_all and C_median_10s by direction (mean over the day).

    For each lane, compute the mean of C, C_median_10s, C_mean_10s, C_std_10s,
    M, Delta, r across all files.
    """
    if "C_median_10s" not in df.columns:
        raise KeyError(
            "Column 'C_median_10s' not found in the aggregated results. "
            "Re-run `python src/run_all.py` with the updated script."
        )

    agg = (
        df.groupby("lane")
        .agg(
            C_all=("C", "mean"),
            C_median_10s=("C_median_10s", "mean"),
            C_mean_10s=("C_mean_10s", "mean"),
            C_std_10s=("C_std_10s", "mean"),
            M=("M", "mean"),
            Delta=("Delta", "mean"),
            r=("r", "mean"),
            n_records=("C", "count"),
        )
        .reset_index()
    )
    agg["direction"] = agg["lane"].map(DIRECTION_NAMES)
    # Reorder columns for readability
    agg = agg[
        [
            "lane",
            "direction",
            "C_all",
            "C_median_10s",
            "C_mean_10s",
            "C_std_10s",
            "M",
            "Delta",
            "r",
            "n_records",
        ]
    ]
    return agg


def main() -> int:
    try:
        df = load_results(ALL_DAY_CSV)
    except Exception as exc:
        print(f"[tables] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[tables] Loaded {len(df)} records from {ALL_DAY_CSV}")

    # Table 1
    t1 = build_dominant_pair_types(df)
    t1.to_csv(TABLE_DOMINANT_PAIR_TYPES, index=False, encoding=CSV_ENCODING)
    print(f"[tables] Saved {TABLE_DOMINANT_PAIR_TYPES} ({len(t1)} rows)")
    print(t1.to_string(index=False))

    # Table 2
    t2 = build_mean_psi_by_pair_type(df)
    t2.to_csv(TABLE_MEAN_PSI, index=False, encoding=CSV_ENCODING)
    print(f"\n[tables] Saved {TABLE_MEAN_PSI} ({len(t2)} rows)")
    print(t2.to_string(index=False))

    # Table 3
    t3 = build_c_all_vs_c_median_by_direction(df)
    t3.to_csv(TABLE_C_ALL_VS_C_MEDIAN, index=False, encoding=CSV_ENCODING)
    print(f"\n[tables] Saved {TABLE_C_ALL_VS_C_MEDIAN} ({len(t3)} rows)")
    print(t3.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

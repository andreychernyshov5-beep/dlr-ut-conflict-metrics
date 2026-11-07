"""
Preprocess the DLR Urban Traffic (DLR UT) dataset into the schema
expected by the `conflict-metrics` library.

The raw DLR UT CSV contains one row per object per timestamp, with
six classification probability columns. This module:
  1. Assigns a single `agent_type` via argmax over the six classes.
  2. Keeps only motorized vehicles (car, van, truck, motorbike).
  3. Assigns a direction label (NE, NW, SW, SE) from the yaw angle.
  4. Projects UTM coordinates onto the mean direction of each sector
     to obtain a longitudinal coordinate `x`.
  5. Returns a DataFrame with the columns expected by conflict-metrics:
     frame, agent_id, x, y, vx, vy, type
     plus extra columns: agent_type, laneID.

Note: `yaw` in the DLR UT dataset is in **degrees** (0 = east,
counterclockwise, 0..360). Do not confuse with radians.

Usage (as a module):
    from preprocess import preprocess_file
    df_metrics = preprocess_file("path/to/trajectories_....csv")

Usage (as a script):
    python src/preprocess.py --file path/to/trajectories_....csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CLASS_COLS,
    CLASS_SHORT_NAMES,
    DIRECTION_LABELS,
    DIRECTION_TO_LANE,
    VEHICLE_TYPES,
)


# Columns we need from the raw DLR UT CSV
USECOLS = [
    "timestamp",
    "id",
    "center_easting",
    "center_northing",
    "velocity_magnitude",
    "yaw",
    *CLASS_COLS,
]


def _empty_metrics_df() -> pd.DataFrame:
    """Return an empty DataFrame with the conflict-metrics schema."""
    return pd.DataFrame(
        {
            "frame": pd.Series(dtype=int),
            "agent_id": pd.Series(dtype=str),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
            "vx": pd.Series(dtype=float),
            "vy": pd.Series(dtype=float),
            "type": pd.Series(dtype=str),
            "agent_type": pd.Series(dtype=str),
            "laneID": pd.Series(dtype=int),
        }
    )


def preprocess_file(csv_path: str | Path) -> pd.DataFrame:
    """
    Preprocess a single DLR UT trajectory CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to a `trajectories_*.csv` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: frame, agent_id, x, y, vx, vy, type,
        agent_type, laneID. Empty if no motorized vehicles are found.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # 1. Read the CSV
    df = pd.read_csv(csv_path, usecols=USECOLS)

    if df.empty:
        return _empty_metrics_df()

    # 2. Parse timestamp and assign frame indices (dense rank, starting at 0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["frame"] = df["timestamp"].rank(method="dense").astype(int) - 1

    # 3. Assign a single agent_type via argmax over the six classes
    df["agent_type"] = (
        df[CLASS_COLS]
        .idxmax(axis=1)
        .map(CLASS_SHORT_NAMES)
    )

    # 4. Keep only motorized vehicles
    df_veh = df[df["agent_type"].isin(VEHICLE_TYPES)].copy()
    if df_veh.empty:
        return _empty_metrics_df()

    # 5. Assign direction from yaw (in degrees, 0..360)
    yaw_deg = df_veh["yaw"] % 360.0
    df_veh["direction"] = pd.cut(
        yaw_deg,
        bins=[0.0, 90.0, 180.0, 270.0, 360.0],
        labels=DIRECTION_LABELS,
        include_lowest=True,
    )
    # Drop rows with undefined direction (should not happen with include_lowest)
    df_veh = df_veh.dropna(subset=["direction"]).copy()
    if df_veh.empty:
        return _empty_metrics_df()

    df_veh["laneID"] = df_veh["direction"].map(DIRECTION_TO_LANE).astype(int)

    # 6. Project UTM coordinates onto the mean direction of each sector
    #    yaw is in degrees, so convert to radians for cos/sin.
    mean_yaw_deg = df_veh.groupby("laneID")["yaw"].mean()
    df_veh["x_proj"] = 0.0
    df_veh["y_proj"] = 0.0
    for lane_id, group in df_veh.groupby("laneID"):
        yaw_rad = np.deg2rad(mean_yaw_deg[lane_id])
        dx = np.cos(yaw_rad)
        dy = np.sin(yaw_rad)
        df_veh.loc[group.index, "x_proj"] = (
            group["center_easting"] * dx + group["center_northing"] * dy
        )
        df_veh.loc[group.index, "y_proj"] = (
            -group["center_easting"] * dy + group["center_northing"] * dx
        )

    # 7. Build the conflict-metrics schema
    df_metrics = pd.DataFrame(
        {
            "frame": df_veh["frame"].astype(int),
            "agent_id": df_veh["id"].astype(str),
            "x": df_veh["x_proj"].astype(float),
            "y": 0.0,
            "vx": df_veh["velocity_magnitude"].astype(float),
            "vy": 0.0,
            "type": "HDV",
            "agent_type": df_veh["agent_type"].values,
            "laneID": df_veh["laneID"].astype(int),
        }
    )

    # Sort by frame and x so that leader-follower pairing is deterministic
    df_metrics = df_metrics.sort_values(["frame", "x"]).reset_index(drop=True)
    return df_metrics


def preprocess_all(
    trajectories_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Preprocess all `trajectories_*.csv` files in a directory.

    Parameters
    ----------
    trajectories_dir : str or Path
        Directory containing the raw DLR UT trajectory CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from time-range string (e.g. "230924-000000_230924-001500")
        to the preprocessed DataFrame.
    """
    trajectories_dir = Path(trajectories_dir)
    if not trajectories_dir.exists():
        raise FileNotFoundError(
            f"Trajectory directory not found: {trajectories_dir}"
        )

    files = sorted(trajectories_dir.glob("trajectories_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No trajectories_*.csv files found in {trajectories_dir}"
        )

    results: dict[str, pd.DataFrame] = {}
    for i, csv_path in enumerate(files, start=1):
        time_range = (
            csv_path.stem
            .replace("trajectories_", "")
        )
        try:
            df = preprocess_file(csv_path)
        except Exception as exc:
            print(f"[warn] Skipping {csv_path.name}: {exc}")
            continue
        results[time_range] = df
        if i % 10 == 0 or i == len(files):
            print(f"[preprocess] {i}/{len(files)} files processed")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess a single DLR UT trajectory CSV file."
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to a trajectories_*.csv file.",
    )
    args = parser.parse_args()

    try:
        df = preprocess_file(args.file)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"Processed: {args.file.name}")
    print(f"  Rows: {len(df)}")
    if not df.empty:
        print(f"  Frames: {df['frame'].nunique()}")
        print(f"  Agents: {df['agent_id'].nunique()}")
        print(f"  Lanes: {sorted(df['laneID'].unique())}")
        print(f"  Agent types: {df['agent_type'].value_counts().to_dict()}")
        print(df.head())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Preprocess the DLR Urban Traffic (DLR UT) dataset into the schema
expected by the `conflict-metrics` library.
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


USECOLS = [
    "timestamp",
    "id",
    "center_easting",
    "center_northing",
    "velocity_magnitude",
    "acceleration_magnitude",
    "yaw",
    *CLASS_COLS,
]


def _empty_metrics_df() -> pd.DataFrame:
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
            "acceleration": pd.Series(dtype=float),
        }
    )


def preprocess_file(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=USECOLS)
    if df.empty:
        return _empty_metrics_df()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["frame"] = df["timestamp"].rank(method="dense").astype(int) - 1

    df["agent_type"] = df[CLASS_COLS].idxmax(axis=1).map(CLASS_SHORT_NAMES)

    df_veh = df[df["agent_type"].isin(VEHICLE_TYPES)].copy()
    if df_veh.empty:
        return _empty_metrics_df()

    yaw_deg = df_veh["yaw"] % 360.0
    df_veh["direction"] = pd.cut(
        yaw_deg,
        bins=[0.0, 90.0, 180.0, 270.0, 360.0],
        labels=DIRECTION_LABELS,
        include_lowest=True,
    )
    df_veh = df_veh.dropna(subset=["direction"]).copy()
    if df_veh.empty:
        return _empty_metrics_df()

    df_veh["laneID"] = df_veh["direction"].map(DIRECTION_TO_LANE).astype(int)

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
            "acceleration": df_veh["acceleration_magnitude"].astype(float),
        }
    )

    df_metrics = df_metrics.sort_values(["frame", "x"]).reset_index(drop=True)
    return df_metrics


def preprocess_all(trajectories_dir: str | Path) -> dict[str, pd.DataFrame]:
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
        time_range = csv_path.stem.replace("trajectories_", "")
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
    parser.add_argument("--file", required=True, type=Path)
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

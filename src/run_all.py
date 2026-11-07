"""
Run the full analysis pipeline on all 96 DLR UT 15-minute files.

For each file and each direction (laneID), the script:
  1. Preprocesses the raw trajectories into the conflict-metrics schema.
  2. Builds leader-follower pairs.
  3. Calibrates the TTC threshold (5th percentile of the empirical TTC).
  4. Adds conflict flags (TTC and DRAC).
  5. Computes psi, M, Delta, r, C, C_median_10s, C_mean_10s, C_std_10s,
     and per-pair-type psi.
  6. Saves the aggregated results to results/conflict_metrics_all_day.csv.

Usage:
    python src/run_all.py [--skip-download]

Options:
    --skip-download   Do not download the dataset; assume it is already in data/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conflict_metrics import build_pairs, calibrate_ttc, add_conflict_flags

from config import (
    ALL_DAY_CSV,
    DRAC_THRESHOLD,
    DLRUT_TRAJECTORIES_DIR,
    N_FILES,
    TTC_PERCENTILE,
    WINDOW_FRAMES,
)
from preprocess import preprocess_all


def sliding_windows(sorted_frames: np.ndarray, win_frames: int, step_frac: float = 0.5):
    """
    Split a sorted array of frame indices into overlapping windows.

    Parameters
    ----------
    sorted_frames : np.ndarray
        Sorted unique frame indices.
    win_frames : int
        Window size in frames.
    step_frac : float
        Step as a fraction of the window size.

    Returns
    -------
    list[tuple[int, int]]
        List of (start_frame, end_frame) pairs (inclusive).
    """
    n = len(sorted_frames)
    if n == 0:
        return []
    step = max(1, int(win_frames * step_frac))
    wins = []
    i = 0
    while i < n:
        j = min(i + win_frames, n)
        wins.append((sorted_frames[i], sorted_frames[j - 1]))
        if j == n:
            break
        i += step
    return wins


def analyze_file_lane(
    df_metrics: pd.DataFrame,
    time_range: str,
    lane_id: int,
) -> dict | None:
    """
    Analyze a single (file, laneID) combination.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Preprocessed trajectories for one file, with columns:
        frame, agent_id, x, y, vx, vy, type, agent_type, laneID.
    time_range : str
        Time-range string (e.g. "230924-000000_230924-001500").
    lane_id : int
        Lane ID (1..4).

    Returns
    -------
    dict or None
        A record with the analysis results, or None if the data is insufficient.
    """
    df_lane = df_metrics[df_metrics["laneID"] == lane_id].copy()
    if df_lane.empty:
        return None

    df_lane = df_lane.sort_values(["frame", "x"]).reset_index(drop=True)

    # Build leader-follower pairs
    pairs = build_pairs(df_lane)
    if pairs.empty:
        return None

    # Extract leader_id and follower_id from pair_id
    pairs[["leader_id", "follower_id"]] = (
        pairs["pair_id"].str.split("__", expand=True)
    )

    # Attach real agent types (car, van, truck, motorbike)
    type_map = df_lane.set_index("agent_id")["agent_type"].to_dict()
    pairs["leader_type"] = pairs["leader_id"].map(type_map)
    pairs["follower_type"] = pairs["follower_id"].map(type_map)
    pairs["pair_type"] = (
        pairs["follower_type"].astype(str) + "→" + pairs["leader_type"].astype(str)
    )

    # Calibrate TTC threshold and add conflict flags
    try:
        ttc_thr = calibrate_ttc(pairs, percentile=TTC_PERCENTILE)
    except TypeError:
        # Older versions of conflict-metrics may not accept `percentile`
        ttc_thr = calibrate_ttc(pairs)
    pairs = add_conflict_flags(pairs, ttc_thr, DRAC_THRESHOLD)

    # Compute C over all frames
    n_frames_total = pairs["frame"].nunique()
    if n_frames_total == 0:
        return None
    n_frames_conflict = pairs.loc[pairs["is_conflict"], "frame"].nunique()
    C = n_frames_conflict / n_frames_total

    # Compute C over sliding windows (10 s)
    frame_list = np.sort(pairs["frame"].unique())
    Cw = []
    for f0, f1 in sliding_windows(frame_list, WINDOW_FRAMES):
        m = pairs[(pairs["frame"] >= f0) & (pairs["frame"] <= f1)]
        nfr = m["frame"].nunique()
        if nfr == 0:
            continue
        ncf = m.loc[m["is_conflict"], "frame"].nunique()
        Cw.append(ncf / nfr)

    if Cw:
        C_median = float(np.median(Cw))
        C_mean = float(np.mean(Cw))
        C_std = float(np.std(Cw))
    else:
        C_median = C_mean = C_std = float("nan")

    # Compute psi per pair type
    psi_frame: dict[str, float] = {}
    for pair_type, g in pairs.groupby("pair_type"):
        n_fc = g.loc[g["is_conflict"], "frame"].nunique()
        psi_frame[pair_type] = n_fc / n_frames_total

    # M, Delta, r
    M = max(psi_frame.values()) if psi_frame else 0.0
    S = sum(psi_frame.values())
    Delta = S - M
    r = float(np.sqrt(M ** 2 + Delta ** 2))

    # Dominant pair type (the one with the largest psi)
    dominant_pair_type = (
        max(psi_frame, key=psi_frame.get) if psi_frame else None
    )

    return {
        "time_range": time_range,
        "lane": int(lane_id),
        "TTC_thr": float(ttc_thr),
        "n_frames": int(n_frames_total),
        "n_pairs": int(len(pairs)),
        "n_vehicles": int(df_lane["agent_id"].nunique()),
        "C": float(C),
        "C_median_10s": C_median,
        "C_mean_10s": C_mean,
        "C_std_10s": C_std,
        "M": float(M),
        "Delta": float(Delta),
        "r": float(r),
        "dominant_pair_type": dominant_pair_type,
        "psi_frame": psi_frame,
    }


def run_all(
    trajectories_dir: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Run the analysis on all files and save the aggregated results.

    Parameters
    ----------
    trajectories_dir : str or Path
        Directory containing the raw DLR UT trajectory CSV files.
    output_csv : str or Path
        Path to the output CSV file.

    Returns
    -------
    pd.DataFrame
        Aggregated results (one row per time_range × lane).
    """
    trajectories_dir = Path(trajectories_dir)
    output_csv = Path(output_csv)

    # 1. Preprocess all files
    print("[run_all] Preprocessing all files...")
    preprocessed = preprocess_all(trajectories_dir)
    print(f"[run_all] Preprocessed {len(preprocessed)} files.")

    # 2. Analyze each (file, lane) combination
    print("[run_all] Analyzing all (file, lane) combinations...")
    results = []
    for i, (time_range, df_metrics) in enumerate(preprocessed.items(), start=1):
        if df_metrics.empty:
            continue
        for lane_id in sorted(df_metrics["laneID"].unique()):
            record = analyze_file_lane(df_metrics, time_range, int(lane_id))
            if record is not None:
                results.append(record)
        if i % 10 == 0 or i == len(preprocessed):
            print(f"[run_all] {i}/{len(preprocessed)} files analyzed")

    # 3. Build the results DataFrame
    if not results:
        print("[run_all] WARNING: no results were produced.")
        return pd.DataFrame()

    # Serialize psi_frame as a JSON string so it can be stored in CSV
    for rec in results:
        rec["psi_frame"] = json.dumps(rec["psi_frame"], ensure_ascii=False)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["time_range", "lane"]).reset_index(drop=True)

    # 4. Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[run_all] Saved {len(results_df)} rows to {output_csv}")

    return results_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full analysis pipeline on the DLR UT dataset."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download the dataset; assume it is already in data/.",
    )
    args = parser.parse_args()

    # 1. Optionally download the dataset
    if not args.skip_download:
        print("[run_all] Downloading dataset (if needed)...")
        try:
            from download_data import (
                download_archive,
                extract_archive,
                verify_dataset,
            )
            from config import DATA_DIR, DLRUT_ARCHIVE_NAME, DLRUT_ZENODO_URL

            archive_path = DATA_DIR / DLRUT_ARCHIVE_NAME
            download_archive(DLRUT_ZENODO_URL, archive_path)
            extract_archive(archive_path, DATA_DIR)
            verify_dataset()
        except Exception as exc:
            print(f"[run_all] ERROR while downloading dataset: {exc}", file=sys.stderr)
            return 1

    # 2. Check that the trajectory directory exists
    if not DLRUT_TRAJECTORIES_DIR.exists():
        print(
            f"[run_all] ERROR: trajectory directory not found: "
            f"{DLRUT_TRAJECTORIES_DIR}",
            file=sys.stderr,
        )
        return 1

    # 3. Run the analysis
    try:
        run_all(DLRUT_TRAJECTORIES_DIR, ALL_DAY_CSV)
    except Exception as exc:
        print(f"[run_all] ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

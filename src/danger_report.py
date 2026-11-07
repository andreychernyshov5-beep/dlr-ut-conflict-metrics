"""
Generate a human-readable danger report for a single 15-minute DLR UT file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conflict_metrics import build_pairs, calibrate_ttc, add_conflict_flags

from config import (
    DANGER_REPORT_TXT,
    DRAC_THRESHOLD,
    MIN_PAIR_FRAMES,
    TTC_PERCENTILE,
    WINDOW_FRAMES,
)
from preprocess import preprocess_file


PSI_HIGH = 0.30
DV_HIGH = 5.0
GAP_SMALL = 3.0
V_LEADER_SLOW = 5.0 / 3.6
N_MIN_PAIRS = 2


def sliding_windows(sorted_frames, win_frames, step_frac=0.5):
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


def find_longest_interval(df_lane, min_veh=3):
    counts = df_lane.groupby("frame")["agent_id"].nunique()
    valid = counts[counts >= min_veh].index.tolist()
    if not valid:
        return None, None, 0
    best_start, best_len = valid[0], 1
    cur_start, cur_len = valid[0], 1
    for i in range(1, len(valid)):
        if valid[i] == valid[i - 1] + 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = valid[i], 1
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    return best_start, best_start + best_len - 1, best_len


def calibrate_lane_ttc(df_lane):
    s, e, n = find_longest_interval(df_lane)
    if s is None:
        pairs = build_pairs(df_lane)
        if pairs.empty:
            return 3.0, 0, 0
        try:
            thr = calibrate_ttc(pairs, percentile=TTC_PERCENTILE)
        except TypeError:
            thr = calibrate_ttc(pairs)
        return thr, 0, 0
    frames = df_lane[(df_lane["frame"] >= s) & (df_lane["frame"] <= e)]["frame"].unique()
    df_interval = df_lane[df_lane["frame"].isin(frames)]
    pairs = build_pairs(df_interval)
    if pairs.empty:
        return 3.0, n, 0
    try:
        thr = calibrate_ttc(pairs, percentile=TTC_PERCENTILE)
    except TypeError:
        thr = calibrate_ttc(pairs)
    return thr, n, len(pairs)


def analyze_lane(df_lane, lane_id):
    df_lane = df_lane.sort_values(["frame", "x"]).reset_index(drop=True)
    ttc_thr, _, _ = calibrate_lane_ttc(df_lane)

    pairs = build_pairs(df_lane)
    if pairs.empty:
        return None

    pairs = add_conflict_flags(pairs, ttc_thr, DRAC_THRESHOLD)

    # Compatibility with both older and newer conflict_metrics:
    # newer build_pairs returns v_follower/v_leader but not leader_id/follower_id;
    # older returns leader_id/follower_id but not velocities.
    if "leader_id" not in pairs.columns or "follower_id" not in pairs.columns:
        pairs[["leader_id", "follower_id"]] = pairs["pair_id"].str.split("__", expand=True)

    if "v_follower" not in pairs.columns or "v_leader" not in pairs.columns:
        spd = df_lane[["frame", "agent_id", "vx"]].copy()
        spd["agent_id"] = spd["agent_id"].astype(str)
        if "v_follower" not in pairs.columns:
            pairs = pairs.merge(
                spd.rename(columns={"agent_id": "follower_id", "vx": "v_follower"}),
                on=["frame", "follower_id"], how="left",
            )
        if "v_leader" not in pairs.columns:
            pairs = pairs.merge(
                spd.rename(columns={"agent_id": "leader_id", "vx": "v_leader"}),
                on=["frame", "leader_id"], how="left",
            )

    if "acceleration" in df_lane.columns:
        acc = df_lane[["frame", "agent_id", "acceleration"]].copy()
        acc["agent_id"] = acc["agent_id"].astype(str)
        pairs = pairs.merge(
            acc.rename(columns={"agent_id": "follower_id", "acceleration": "a_follower"}),
            on=["frame", "follower_id"], how="left",
        ).merge(
            acc.rename(columns={"agent_id": "leader_id", "acceleration": "a_leader"}),
            on=["frame", "leader_id"], how="left",
        )
    else:
        pairs["a_follower"] = np.nan
        pairs["a_leader"] = np.nan

    pairs["lane"] = int(lane_id)
    pairs["TTC_thr_l"] = float(ttc_thr)

    g = pairs.groupby("pair_id")
    feat = pd.DataFrame({
        "lane": g["lane"].first(),
        "TTC_thr_l": g["TTC_thr_l"].first(),
        "n_frames": g.size(),
        "n_conflict": g["is_conflict"].sum(),
        "mean_dv": g["dv"].mean(),
        "mean_gap": g["s"].mean(),
        "mean_ttc": g["ttc"].apply(lambda s: s[np.isfinite(s)].mean()),
        "max_drac": g["drac"].max(),
        "a_leader_std": g["a_leader"].std(ddof=0),
        "a_leader_min": g["a_leader"].min(),
        "v_leader_mean": g["v_leader"].mean(),
        "v_follower_std": g["v_follower"].std(ddof=0),
    }).reset_index()
    feat["follower_id"] = feat["pair_id"].str.split("__").str[1]
    feat["leader_id"] = feat["pair_id"].str.split("__").str[0]
    feat["pair_id_display"] = feat["follower_id"] + "→" + feat["leader_id"]
    feat["psi_csv"] = feat["n_conflict"] / feat["n_frames"]
    for c in ["a_leader_std", "v_follower_std"]:
        feat[c] = feat[c].fillna(0.0)
    feat = feat[feat["n_frames"] >= MIN_PAIR_FRAMES].reset_index(drop=True)

    frame_list = np.sort(pairs["frame"].unique())
    Cw = []
    for f0, f1 in sliding_windows(frame_list, WINDOW_FRAMES):
        m = pairs[(pairs["frame"] >= f0) & (pairs["frame"] <= f1)]
        nfr = m["frame"].nunique()
        if nfr == 0:
            continue
        ncf = m.loc[m["is_conflict"], "frame"].nunique()
        Cw.append(100.0 * ncf / nfr)

    lane_record = {
        "laneID": int(lane_id),
        "TTC_thr_l": float(ttc_thr),
        "n_frames_total": int(pairs["frame"].nunique()),
        "n_frames_conflict": int(pairs.loc[pairs["is_conflict"], "frame"].nunique()),
        "n_pairs_unique": int(pairs["pair_id"].nunique()),
        "n_pairs_conflict": int(pairs.loc[pairs["is_conflict"], "pair_id"].nunique()),
        "C_lane_median": float(np.median(Cw)) if Cw else float("nan"),
        "C_lane_mean": float(np.mean(Cw)) if Cw else float("nan"),
        "C_lane_std": float(np.std(Cw)) if Cw else float("nan"),
    }
    return {"lane": lane_record, "pairs": feat}


def build_report(file_path: Path) -> str:
    df = preprocess_file(file_path)
    if df.empty:
        return f"DANGER REPORT\n{'=' * 60}\nNo motorized vehicles found in {file_path.name}.\n"

    lane_results = []
    for lane_id, df_lane in df.groupby("laneID"):
        res = analyze_lane(df_lane, int(lane_id))
        if res is not None:
            lane_results.append(res)

    if not lane_results:
        return f"DANGER REPORT\n{'=' * 60}\nNo pairs could be built for {file_path.name}.\n"

    lanes = pd.DataFrame([r["lane"] for r in lane_results]).sort_values("laneID")
    pairs_all = pd.concat([r["pairs"] for r in lane_results], ignore_index=True)

    af = (
        pairs_all.groupby("follower_id")
        .agg(
            n_pairs=("leader_id", "count"),
            lanes=("lane", lambda s: sorted(set(int(x) for x in s))),
            mean_psi=("psi_csv", "mean"),
            max_psi=("psi_csv", "max"),
            mean_dv=("mean_dv", "mean"),
            mean_gap=("mean_gap", "mean"),
        ).reset_index()
    )
    af = af[af["mean_psi"] > 0.1].sort_values(["mean_psi", "n_pairs"], ascending=False)
    af_top = af.head(10)

    lm = (
        pairs_all.groupby("leader_id")
        .agg(
            n_pairs=("follower_id", "count"),
            lanes=("lane", lambda s: sorted(set(int(x) for x in s))),
            mean_psi=("psi_csv", "mean"),
            max_psi=("psi_csv", "max"),
            mean_v_leader=("v_leader_mean", "mean"),
        )
        .sort_values(["n_pairs", "mean_psi"], ascending=False)
        .reset_index()
    )
    lm["v_leader_kmh"] = lm["mean_v_leader"] * 3.6
    lm_top = lm.head(10)

    top_pairs = pairs_all.sort_values("psi_csv", ascending=False).head(10)

    lines = []
    lines.append("REAL-TRAFFIC DANGER REPORT")
    lines.append("=" * 60)
    lines.append(f"File: {file_path.name}")
    lines.append("")
    lines.append("Per-lane TTC thresholds:")
    for _, r in lanes.iterrows():
        lines.append(f"  lane {int(r['laneID'])}: TTC_thr = {r['TTC_thr_l']:.3f} s")
    lines.append("")
    lines.append("Per-lane C_lane (median over 10-s windows):")
    for _, r in lanes.sort_values("C_lane_median", ascending=False).iterrows():
        lines.append(
            f"  lane {int(r['laneID'])}: C = {r['C_lane_median']:.2f} % "
            f"(mean {r['C_lane_mean']:.2f}, std {r['C_lane_std']:.2f})"
        )
    lines.append("")
    lines.append("Aggressive followers (top-10 by mean_psi):")
    for _, r in af_top.iterrows():
        lines.append(
            f"  follower {r['follower_id']} (lanes {r['lanes']}): "
            f"n_pairs={int(r['n_pairs'])}, mean_psi={r['mean_psi']:.3f}, "
            f"max_psi={r['max_psi']:.3f}, mean_dv={r['mean_dv']:.2f}, "
            f"mean_gap={r['mean_gap']:.2f}"
        )
    lines.append("")
    lines.append("Leader magnets (top-10 by n_pairs):")
    for _, r in lm_top.iterrows():
        lines.append(
            f"  leader {r['leader_id']} (lanes {r['lanes']}): "
            f"n_pairs={int(r['n_pairs'])}, mean_psi={r['mean_psi']:.3f}, "
            f"max_psi={r['max_psi']:.3f}, v_leader={r['v_leader_kmh']:.2f} km/h"
        )
    lines.append("")
    lines.append("Top-10 pairs by psi_csv (all lanes):")
    for _, r in top_pairs.iterrows():
        lines.append(
            f"  {r['pair_id_display']} (lane {int(r['lane'])}): "
            f"psi={r['psi_csv']:.3f}, frames={int(r['n_frames'])}, "
            f"conflict={int(r['n_conflict'])}, "
            f"mean_dv={r['mean_dv']:.2f}, mean_gap={r['mean_gap']:.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a danger report for a single DLR UT 15-minute file."
    )
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--out", default=DANGER_REPORT_TXT, type=Path)
    args = parser.parse_args()

    try:
        report = build_report(args.file)
    except Exception as exc:
        print(f"[danger_report] ERROR: {exc}", file=sys.stderr)
        return 1

    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"\n[danger_report] Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

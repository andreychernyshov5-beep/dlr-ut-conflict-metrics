"""
Generate the three figures presented in the paper.

Input:
    results/conflict_metrics_all_day.csv

Outputs:
    figures/C_vs_time.png
    figures/M_Delta_C_NE.png
    figures/C_vs_volume.png

Usage:
    python src/plots.py
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ALL_DAY_CSV,
    DIRECTION_COLORS,
    DIRECTION_NAMES,
    FIG_C_VS_TIME,
    FIG_C_VS_VOLUME,
    FIG_M_DELTA_C_NE,
    FIG_DPI,
)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load the aggregated results and add a parsed `start_time` column."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Aggregated results not found: {csv_path}. "
            "Run `python src/run_all.py` first."
        )
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["start_time"] = pd.to_datetime(
        df["time_range"].str.split("_").str[0],
        format="%y%m%d-%H%M%S",
    )
    return df


def plot_c_vs_time(df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 1: share of conflict frames C by direction over the day.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for i, lane in enumerate([1, 2, 3, 4]):
        ax = axes[i // 2, i % 2]
        sub = df[df["lane"] == lane].sort_values("start_time")
        ax.plot(
            sub["start_time"],
            sub["C"] * 100,
            "o-",
            markersize=2,
            color=DIRECTION_COLORS[lane],
            label=DIRECTION_NAMES[lane],
        )
        ax.set_title(f"{DIRECTION_NAMES[lane]}: C vs time")
        ax.set_ylabel("C, %")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper left")

    for ax in axes[1, :]:
        ax.set_xlabel("Time (UTC)")

    plt.suptitle(
        "Share of conflict frames C by direction, 24.09.2023", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {out_path}")


def plot_m_delta_c_ne(df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 2: inequality check M <= C <= M + Delta for the NE direction.
    """
    sub = df[df["lane"] == 1].sort_values("start_time")
    if sub.empty:
        print("[plots] WARNING: no data for lane 1 (NE); skipping M_Delta_C_NE.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(
        sub["start_time"],
        sub["M"] * 100,
        "o-",
        markersize=3,
        label="M (limiting conflict)",
        color="tab:blue",
    )
    ax.plot(
        sub["start_time"],
        sub["Delta"] * 100,
        "s-",
        markersize=3,
        label="Delta (boundary gap)",
        color="tab:orange",
    )
    ax.plot(
        sub["start_time"],
        sub["C"] * 100,
        "^-",
        markersize=3,
        label="C (macro conflict)",
        color="tab:red",
    )
    ax.fill_between(
        sub["start_time"],
        sub["M"] * 100,
        (sub["M"] + sub["Delta"]) * 100,
        alpha=0.15,
        color="gray",
        label="Allowed range [M, M+Δ]",
    )

    ax.set_title("Inequality check M ≤ C ≤ M + Δ for NE direction (Lane 1)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("%")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {out_path}")


def plot_c_vs_volume(df: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 3: conflict share C vs traffic volume (number of vehicles per
    15-min interval) by direction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for lane in [1, 2, 3, 4]:
        sub = df[df["lane"] == lane]
        ax.scatter(
            sub["n_vehicles"],
            sub["C"] * 100,
            label=DIRECTION_NAMES[lane],
            alpha=0.5,
            s=15,
            color=DIRECTION_COLORS[lane],
        )
    ax.set_xlabel("Number of vehicles per 15-min interval")
    ax.set_ylabel("C, %")
    ax.set_title("C vs traffic volume by direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {out_path}")


def main() -> int:
    try:
        df = load_results(ALL_DAY_CSV)
    except Exception as exc:
        print(f"[plots] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[plots] Loaded {len(df)} records from {ALL_DAY_CSV}")

    try:
        plot_c_vs_time(df, FIG_C_VS_TIME)
        plot_m_delta_c_ne(df, FIG_M_DELTA_C_NE)
        plot_c_vs_volume(df, FIG_C_VS_VOLUME)
    except Exception as exc:
        print(f"[plots] ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

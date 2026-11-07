"""
Configuration and constants for the DLR UT conflict metrics analysis.

This module centralizes all paths, file names, dataset parameters, and
metric-related constants used throughout the repository.
"""

from pathlib import Path

# ============================================================
# Repository paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# Create directories if they do not exist
for _dir in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# DLR UT dataset parameters
# ============================================================
# Zenodo record (public fragment, 24.09.2023)
DLRUT_ZENODO_DOI = "10.5281/zenodo.14773161"
DLRUT_ZENODO_URL = (
    "https://zenodo.org/records/14773161/files/"
    "DLR-Urban-Traffic-dataset_v1-2-0.zip"
)
DLRUT_ARCHIVE_NAME = "DLR-Urban-Traffic-dataset_v1-2-0.zip"
DLRUT_EXTRACTED_DIR_NAME = "DLR-Urban-Traffic-dataset_v1-2-0"

# Path to the extracted dataset
DLRUT_ROOT = DATA_DIR / DLRUT_EXTRACTED_DIR_NAME
DLRUT_TRAJECTORIES_DIR = DLRUT_ROOT / "raw_data" / "trajectories"

# Number of 15-minute files in the public fragment (24.09.2023)
N_FILES = 96

# Sampling rate of the interpolated trajectories (Hz)
FPS = 20
FRAME_DURATION_S = 1.0 / FPS

# Window size for computing C_median / C_mean (seconds)
WINDOW_SEC = 10.0
WINDOW_FRAMES = int(WINDOW_SEC * FPS)

# ============================================================
# Preprocessing parameters
# ============================================================
# Vehicle classes kept in the analysis (pedestrians and bicycles are excluded)
VEHICLE_TYPES = ["car", "van", "truck", "motorbike"]

# Classification columns in the raw DLR UT CSV
CLASS_COLS = [
    "classifications_car",
    "classifications_van",
    "classifications_truck",
    "classifications_motorbike",
    "classifications_bicycle",
    "classifications_pedestrian",
]

# Short names for the classification columns
CLASS_SHORT_NAMES = {c: c.replace("classifications_", "") for c in CLASS_COLS}

# Direction labels assigned from yaw (in degrees, normalized to [0, 360))
DIRECTION_LABELS = ["NE", "NW", "SW", "SE"]

# Mapping from direction label to lane ID (as used in the paper)
DIRECTION_TO_LANE = {"NE": 1, "NW": 2, "SW": 3, "SE": 4}

# Inverse mapping (for readability in plots)
LANE_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_LANE.items()}

# Minimum number of frames for a pair to be included in the ranking
MIN_PAIR_FRAMES = 10

# ============================================================
# Conflict metrics parameters (from conflict-metrics library)
# ============================================================
# DRAC threshold (m/s^2), fixed as in the library's real-traffic example
DRAC_THRESHOLD = 3.4

# Percentile of the empirical TTC distribution used to calibrate TTC_thr
TTC_PERCENTILE = 5

# ============================================================
# Output file names
# ============================================================
# Aggregated results for all 96 files and 4 directions
ALL_DAY_CSV = RESULTS_DIR / "conflict_metrics_all_day.csv"

# Tables
TABLE_DOMINANT_PAIR_TYPES = RESULTS_DIR / "dominant_pair_types.csv"
TABLE_MEAN_PSI = RESULTS_DIR / "mean_psi_by_pair_type.csv"
TABLE_C_ALL_VS_C_MEDIAN = RESULTS_DIR / "c_all_vs_c_median_by_direction.csv"

# Danger report
DANGER_REPORT_TXT = RESULTS_DIR / "danger_report.txt"

# Figures
FIG_C_VS_TIME = FIGURES_DIR / "C_vs_time.png"
FIG_M_DELTA_C_NE = FIGURES_DIR / "M_Delta_C_NE.png"
FIG_C_VS_VOLUME = FIGURES_DIR / "C_vs_volume.png"

# ============================================================
# Plotting parameters
# ============================================================
# Colors for the four directions
DIRECTION_COLORS = {
    1: "tab:blue",    # NE
    2: "tab:orange",  # NW
    3: "tab:green",   # SW
    4: "tab:red",     # SE
}

# Human-readable direction names for plot titles/legends
DIRECTION_NAMES = {
    1: "NE (Lane 1)",
    2: "NW (Lane 2)",
    3: "SW (Lane 3)",
    4: "SE (Lane 4)",
}

# Figure DPI
FIG_DPI = 150

# ============================================================
# Reproducibility
# ============================================================
# Random seed (not strictly needed here, but kept for consistency)
RANDOM_SEED = 42

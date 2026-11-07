# Conflict Metrics for Urban Intersections: DLR UT Dataset Analysis

This repository contains the code and analysis scripts for the paper:

**"Conflict Metrics for Urban Intersections: Application of a Two-Layer Model to the DLR Urban Traffic Dataset"**  
*Author: A.V. Chernyshov*

## Overview

This project applies a two-layer conflict model to the **DLR Urban Traffic (DLR UT) dataset**, recorded at a signalized intersection in Braunschweig, Germany. The analysis uses the open-source [`conflict-metrics`](https://github.com/andreychernyshov5-beep/conflict-metrics) library to compute conflict metrics (`ψ`, `Ψ_frame`, `C`, `M`, `Δ`, `r`) and analyze the contribution of different vehicle pair types (e.g., car→car, car→truck).

The repository provides a fully reproducible pipeline to:
1.  Download the public fragment of the DLR UT dataset (24.09.2023).
2.  Preprocess the raw data into the required schema.
3.  Process all 96 (15-minute) files and save the aggregated results.
4.  Generate tables and figures presented in the paper.
5.  Produce a detailed "danger report" for a single 15-minute interval.

## Data

The analysis is based on the **DLR Urban Traffic (DLR UT) dataset**.

- **Source:** Zenodo, DOI: [10.5281/zenodo.14773161](https://doi.org/10.5281/zenodo.14773161)
- **Documentation:** [DLR UT Dataset Documentation (v1.0.1)](https://doi.org/10.5281/zenodo.11396372)
- **Fragment Used:** One day of data (September 24, 2023), comprising 96 files of 15-minute trajectory data.
- **Sampling rate:** 20 Hz (interpolated trajectories).
- **Coordinate system:** UTM Zone 32N (Easting 604625–604900, Northing 5792650–5792875).
- **Object classes:** pedestrian, bicycle, motorbike, car, van, truck. In this analysis, only motorized vehicles (car, van, truck, motorbike) are kept.
- **License:** The DLR UT dataset is published under the **CC BY-NC-SA 4.0** license. See the Zenodo record for full terms.

The dataset is **not included** in this repository. The `download_data.py` script automates the download and extraction process.

**Known limitations of the dataset (from the official documentation):**
- Accuracy decreases with distance from the intersection center.
- Object tracking may be interrupted in shaded areas (e.g., the western arm).
- A single object may be split into multiple trajectories, or multiple objects may be merged into one trajectory.
- A trajectory ID may occasionally switch between different objects.

These limitations are inherited from the source data and are not corrected by this repository.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andreychernyshov5-beep/dlr-ut-conflict-metrics.git
    cd dlr-ut-conflict-metrics

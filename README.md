# Drag and Trajectory Analysis under Neutral Wind

This repo is a Python-based framework for retrieving, interpolating, visualizing, and propagating satellites through **neutral wind**. The neutral wind and neutral density comes from WAM-IPE. The propagator used is the python wrapper for Orekit, which includes the following perturbations: atmospheric drag, high-resolution gravity field, solar radiation pressure, third-body attraction to the sun and moon, and solid tides.

---

## 🚀 Overview

Neutral_Wind provides tools to:

* Query atmospheric models for density, temperature, and wind
* Interpolate neutral wind fields across altitude, latitude, longitude, and time
* Propagate orbits and trajectories under wind and drag influences
* Visualize atmospheric and orbital results with custom plots

---

## 📁 Repository Structure

```
Neutral_Wind/
├── wind_analysis.py                      # Trajectory analysis due to wind
├── atmospheres.py                        # Atmospheric model interface definitions
├── query_msis.py                         # Query MSIS-like models for atmospheric parameters
├── wam_vertical_interpolate_neutral_wind.py  # Vertical interpolation of wind profiles
├── propagator.py                         # Orbit/trajectory propagation with wind effects
├── plot.py                               # Plotting and visualization utilities
├── orekit_helpers.py                     # Orekit-related helper functions
├── orekit-data.zip                       # Orekit data bundle (if using Orekit)
├── figures/                              # Example figures and visualizations
├── results/                              # Example results (profiles, trajectories)
└── Space-Weather-Data/                   # Space weather and geomagnetic input data
```

---

## ⚙️ Getting Started


### 🐍 Instructions to Create a Miniconda Virtual Environment with Orekit Python Wrapper

#### 1. Create a virtual environment

```bash
conda create -n orekit_venv python anaconda
```

#### 2. Activate the virtual environment

```bash
conda activate orekit_venv
```

#### 3. Install packages using conda

```bash
conda install -c conda-forge orekit
conda install -c conda-forge pandas
conda install -c anaconda numpy
conda install -c anaconda python-dateutil
conda install -c conda-forge matplotlib
```

#### 4. Check the installed package version

```bash
pip show orekit
```

---

## ✉️ Contact

**Maintainer:** Mia Tian
GitHub: [@mia-tian](https://github.com/mia-tian)
Email: miatian@mit.edu or miatian2002@gmail.com

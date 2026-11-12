# Reproducibility of the Experiments

This repository contains the data and scripts required to reproduce the experiments reported in the paper. The scripts cover model fitting and prediction.

The repository is organized as follows:

```text
.
├── Data/
│   └── ...
├── Experiments/
│   ├── fit_models_v1.py
│   ├── fit_models_v2.py
│   ├── predictions_v1.py
│   ├── predictions_v2.py
│   └── spbn.py
├── Metrics/
│   └── ...
├── Models/
│   └── ...
└── README.md
```

- `Data/` contains the datasets used in the experiments. Data is stored after preprocessing. Please contact the authors for raw data and preprocessing scripts.
- `Experiments/` contains the scripts for fitting the models and generating predictions.
- `Models/` is the output directory for fitted models. It is created and populated when the corresponding scripts are executed. Within this directory, a separate subdirectory is used for each dataset (e.g., `Models/BRCA/`).
- `Metrics/` is the output directory for prediction metrics. It is created and populated when the corresponding scripts are executed.
  
The fitted models and prediction results contain the information required to reproduce the figures reported in the paper.

## ⚙️ Requirements

The experiments require two versions of RNACOREX:

- **RNACOREX 0.1.5** for `fit_models_v1.py` and `predictions_v1.py`.
- **RNACOREX 0.2** for `fit_models_v2.py`, `predictions_v2.py`, and `spbn.py`.

The two versions must be used in separate Python environments to avoid dependency conflicts and ensure that each script runs with the intended implementation.

## 📦 Installation

We recommend using separate Conda environments for the two RNACOREX versions.

Install the previous RNACOREX release from PyPi:

```bash
python -m pip install rnacorex==0.1.5
```

Install RNACOREX 0.2 from the appropriate distribution source. If the release is available on PyPI, install the corresponding published version. Alternatively, if the package is being used directly from its source repository, install the local project.

```bash
python -m pip install -e /path/to/RNACOREX
```

Replace `/path/to/RNACOREX` with the actual path to the RNACOREX 0.2 source directory.

## 🧪 Experiments

Run the scripts in the following order, using the appropriate Conda environment:

1. `fit_models_v1.py` — Fits LGBN models with RNACOREX 0.1.5 (`rnacorex-015`).
2. `fit_models_v2.py` — Fits LGBN and SLGBN models with RNACOREX 0.2 (`rnacorex-020`).
3. `spbn.py` — Fits SPBN models and generates predictions with RNACOREX 0.2 (`rnacorex-020`).
4. `predictions_v1.py` — Generates predictions using LGBN models fitted with RNACOREX 0.1.5 (`rnacorex-015`).
5. `predictions_v2.py` — Generates predictions using LGBN and SLGBN models fitted with RNACOREX 0.2 (`rnacorex-020`).

The fitting scripts must be run before the corresponding prediction scripts.

## 🗂️ Model Organization and Outputs

The Model output directory is organized by dataset:

```text
Models/
├── BRCA/
│   └── ...
├── ...
Metrics/
├── lgbn_v1_results.csv
├── ...
```

In Models/ a separate directory is used for each dataset. The directory contents depend on the model-fitting workflow.

- **RNACOREX 0.1.5:** a separate model is generated for each network size.
- **RNACOREX 0.2 (LGBN and SLGBN):** models fitted with different network sizes are stored together in a single object.
- **SPBN:** model fitting and prediction are performed in the same script to avoid issues associated with saving and reloading PyBNesian models.

The scripts also generate prediction metrics as individual CSV files in `Metrics/`.

## 🔬 Reproducibility Notes

For reliable reproduction of the reported results:

1. Use RNACOREX 0.1.5 for `fit_models_v1.py` and `predictions_v1.py`.
2. Use RNACOREX 0.2 for `fit_models_v2.py`, `predictions_v2.py` and `spbn.py`.
3. Execute the scripts in the prescribed order.

The scripts, datasets, fitted models, and prediction outputs together define the computational workflow used to reproduce the experiments described in the paper.

Execution-time comparisons were obtained by measuring individual stages of the model-fitting process, which required modifications to the RNACOREX source code. These modifications are not included in this repository but can be requested from the authors.

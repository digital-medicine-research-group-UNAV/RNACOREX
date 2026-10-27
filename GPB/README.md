# Reproducibility of the Experiments

This repository contains the data and scripts required to reproduce the experiments reported in the paper. The scripts cover model fitting and prediction using two versions of RNACOREX, with separate workflows for the models supported by the previous version and those supported by RNACOREX 0.2.

The repository is organized as follows:

```text
.
├── Data/
│   └── ...
├── Experiments/
│   ├── fit_models_v1.py
│   ├── fit_models_v2.py
│   ├── spbn.py
│   └── predict.py
├── Models/
│   └── ...
└── README.md
```

- `Data/` contains the datasets used in the experiments.
- `Experiments/` contains the scripts for fitting the models and generating predictions.
- `Models/` is the output directory for fitted models. It is created and populated when the corresponding scripts are executed. Within this directory, a separate subdirectory is used for each dataset (e.g., `Models/BRCA/`).

The fitted models and prediction results contain the information required to reproduce the figures reported in the paper.

## Requirements

The experiments require two versions of RNACOREX:

- **RNACOREX 0.1.5** for `fit_models_v1.py`.
- **RNACOREX 0.2** for `fit_models_v2.py`, `spbn.py`, and `predict.py`.

The two versions must be used in separate Python environments to avoid dependency conflicts and ensure that each script runs with the intended implementation.

Python and the additional dependencies required by the scripts must also be installed. The required Python version and dependency versions should be selected according to the compatibility requirements of RNACOREX and the experimental environment.

## Installation

We recommend using separate Conda environments for the two RNACOREX versions.

### 1. Environment for RNACOREX 0.1.5

Create and activate a Conda environment:

```bash
conda create -n rnacorex-015 python=3.11
conda activate rnacorex-015
```

Install the previous RNACOREX release:

```bash
python -m pip install rnacorex==0.1.5
```

Before running `fit_models_v1.py`, check whether the required engines are available:

```python
import rnacorex

rnacorex.check_engines()
```

If the required engines are missing, download them using:

```python
rnacorex.download()
```

Run `check_engines()` again to verify their availability before starting the experiments.

### 2. Environment for RNACOREX 0.2

Create and activate a separate Conda environment:

```bash
conda create -n rnacorex-020 python=3.11
conda activate rnacorex-020
```

Install RNACOREX 0.2 from the appropriate distribution source. If the release is available on PyPI, install the corresponding published version. Alternatively, if the package is being used directly from its source repository, install the local project.

For a local development checkout:

```bash
python -m pip install -e /path/to/RNACOREX-git
```

Replace `/path/to/RNACOREX-git` with the actual path to the RNACOREX 0.2 source directory.

The editable installation allows the scripts to use the source code in that directory. For strict reproducibility, record the exact RNACOREX commit and dependency versions used to generate the results.

## Experimental Workflow

The scripts must be executed in the order described below. In particular, `predict.py` must be run only after the three model-fitting or fitting-and-prediction scripts have completed successfully.

### Step 1. Fit models using RNACOREX 0.1.5

**Script:** `Experiments/fit_models_v1.py`

**Required version:** RNACOREX 0.1.5

Activate the corresponding Conda environment:

```bash
conda activate rnacorex-015
```

Then run:

```bash
python Experiments/fit_models_v1.py
```

This script fits the models using the previous RNACOREX implementation. A separate fitted model is generated for each network size. The resulting models are stored under the corresponding dataset directory in `Models/`.

For example, models associated with the BRCA dataset are stored in:

```text
Models/
└── BRCA/
    └── ...
```

Ensure that the required RNACOREX engines are available before execution.

### Step 2. Fit LGBN and SLGBN models using RNACOREX 0.2

**Script:** `Experiments/fit_models_v2.py`

**Required version:** RNACOREX 0.2

Activate the environment containing RNACOREX 0.2:

```bash
conda activate rnacorex-020
```

Run:

```bash
python Experiments/fit_models_v2.py
```

This script fits the LGBN and SLGBN models using the updated RNACOREX implementation.

Unlike the workflow used in `fit_models_v1.py`, models fitted with different network sizes are stored together in a single object. This organization reflects the model-handling approach used by the RNACOREX 0.2 workflow.

The resulting fitted models are stored in the appropriate dataset-specific directory under `Models/`.

### Step 3. Fit SPBN models and generate predictions

**Script:** `Experiments/spbn.py`

**Required version:** RNACOREX 0.2

Run:

```bash
python Experiments/spbn.py
```

This script fits the SPBN models and generates predictions.

SPBN is handled separately because saving a fitted model and subsequently loading it can cause problems when using PyBNesian. To avoid this issue, model fitting and prediction are performed within the same script, without saving and reloading the fitted model between these operations.

This workflow is intended to avoid serialization and reloading issues associated with the SPBN implementation.

### Step 4. Generate predictions using the fitted SPBN models

**Script:** `Experiments/predict.py`

**Required version:** RNACOREX 0.2

Run this script only after the previous three scripts have completed successfully:

```bash
python Experiments/predict.py
```

This script generates predictions using the fitted SPBN models and the corresponding model artifacts produced by the preceding workflow.

The required model files and intermediate outputs must be available before execution. The script should not be run before the preceding model-fitting and prediction stages have completed.

## Model Organization and Outputs

The output directory is organized by dataset:

```text
Models/
├── BRCA/
│   └── ...
├── ...
```

A separate directory is used for each dataset. The directory contents depend on the model-fitting workflow.

- **RNACOREX 0.1.5:** a separate model is generated for each network size.
- **RNACOREX 0.2 (LGBN and SLGBN):** models fitted with different network sizes are stored together in a single object.
- **SPBN:** model fitting and prediction are performed in the same script to avoid issues associated with saving and reloading PyBNesian models.

The scripts also generate prediction results and other experimental outputs. Together, the fitted models and prediction results retain the information needed to reproduce the figures reported in the paper.

## Reproducibility Notes

For reliable reproduction of the reported results:

1. Use RNACOREX 0.1.5 exclusively for `fit_models_v1.py`.
2. Use RNACOREX 0.2 for `fit_models_v2.py`, `spbn.py`, and `predict.py`.
3. Keep the two RNACOREX environments separate.
4. Verify the availability of the required RNACOREX engines before running `fit_models_v1.py`.
5. Execute the scripts in the prescribed order.
6. Ensure that the datasets in `Data/` are available at the paths expected by the scripts.
7. Preserve the generated model files and prediction results required to reproduce the paper figures.
8. Record the Python version, dependency versions, RNACOREX version or commit, and relevant execution settings used for each run.

The scripts, datasets, fitted models, and prediction outputs together define the computational workflow used to reproduce the experiments described in the paper.

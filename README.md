# BBOB Data Generation Pipeline

This repository implements a **three-step pipeline** for generating, processing, and using datasets derived from **BBOB (Black-Box Optimization Benchmark) functions**.

This README currently documents **Step 1**, which is responsible for **data generation via Design of Experiments (DoE)**.  
Steps 2 and 3 will build on the data generated here.

---

## Step 1 — Data Generation (`sampler.py`)

The first step of the pipeline samples BBOB benchmark functions using different DoE strategies and logs the resulting datasets to disk.

This step is implemented in `sampler.py`


---

## Purpose

`sampler.py` generates datasets by:

1. Selecting a BBOB function (problem ID, dimension, instance),
2. Sampling points in the decision space using a chosen DoE method,
3. Scaling samples to the true BBOB bounds,
4. Evaluating the BBOB function at those points,
5. Logging inputs and function values using IOH’s `Analyzer`.

The script is designed to be executed from the command line and produces structured, reproducible datasets.

---

## Dependencies

The following Python packages are required:

- `numpy`
- `scipy`
- `ioh`
- `pflacco`
- `scikit-learn`

Install them with:
`pip install numpy scipy ioh pflacco scikit-learn`


---

## Basic Usage

Run the sampler from the repository root:
`python sampler.py [OPTIONS]`


Minimal example:
`python sampler.py --problem-id 1 --dimension 5 --sampler lhs`

This command:
- Selects **BBOB problem 1**
- Uses **5 dimensions**
- Applies **Latin Hypercube Sampling**
- Generates `dimension × multiplier` samples

---

## Command-Line Arguments

### BBOB Problem Configuration

| Argument | Description | Default |
|--------|-------------|---------|
| `--problem-id` | BBOB function ID (1–24) | `1` |
| `--dimension` | Problem dimensionality | `2` |
| `--instance` | BBOB instance ID (1–15) | `1` |

---

### Sampling Configuration

| Argument | Description | Default |
|--------|-------------|---------|
| `--sampler` | Sampling method: `monte-carlo`, `lhs`, `sobol`, `halton` | `lhs` |
| `--multiplier` | Number of samples = `dimension × multiplier` | `25` |
| `--random-seed` | Random seed for reproducibility | `42` |

---

### Sampler-Specific Options

These options affect LHS, Sobol, and Halton samplers.

| Argument | Description | Default |
|--------|-------------|---------|
| `--quasi-random-criterion` | Optimization criterion (`random-cd`, `lloyd`) | `random-cd` |
| `--lhs-strength` | Strength of LHS design (1 or 2) | `1` |

**Note:**  
Sobol sampling automatically rounds the number of samples **up to the nearest power of two**, as required by the Sobol sequence construction.

---

## Supported Sampling Methods

- **Monte Carlo**  
  Uniform random sampling over the unit hypercube.

- **Latin Hypercube Sampling (LHS)**  
  Stratified sampling using `scipy.stats.qmc.LatinHypercube`, with optional optimization.

- **Sobol Sequences**  
  Low-discrepancy quasi-random sequences with scrambling and optimization.

- **Halton Sequences**  
  Deterministic low-discrepancy sequences with optional scrambling.

All sampling methods initially generate points in `[0, 1]^d`, which are then scaled to the BBOB problem bounds.

---

## Output Structure

Results are logged automatically using IOH’s `Analyzer`.

The output directory structure is:
data/
└── <sampler>/
└── <problem_id><problem_name>/
└── Dim<dimension>/
└── Instance_<instance>/
└── <multiplier>_<random_seed>/


Each directory contains:
- Sampled decision vectors,
- Function evaluations,
- Logged metadata from the IOH framework.

---

## Example
python sampler.py
--problem-id 5
--dimension 10
--instance 3
--sampler sobol
--multiplier 50
--random-seed 123


This command:
- Generates Sobol samples (rounded to a power of two),
- Evaluates BBOB function 5,
- Stores results under:

`data/sobol/5_<problem_name>/Dim_10/Instance_3/50_123/`


---
## Step 2 — Objective Evaluation (`y_sampling.py`)

Step 2 evaluates previously generated **X samples** on **all BBOB benchmark functions and instances**, producing corresponding **Y (objective value) datasets**.

This step consumes the output of **Step 1** and is implemented in:
`y_sampling.py`


---

## Purpose

`y_sampling.py` performs the following tasks:

1. Reads pre-generated **X samples** from CSV files,
2. Automatically infers:
   - problem dimension,
   - random seed,
   - number of samples,
   - objective type (e.g. `ELA_extraction`, `reduction`),
3. Evaluates **all BBOB functions (1–24)** and **all instances (0–14)** on the given samples,
4. Saves function values (`fX`) to disk using a structured directory layout.

This step decouples **input sampling (X)** from **objective evaluation (Y)**, allowing reuse of the same samples across multiple benchmark functions.

---

## Expected Input Structure

`y_sampling.py` expects X samples stored as CSV files under a directory structure similar to:

x_samples/
└── <objective_type>/
└── Dimension_<D>/
└── seed_<SEED>/
└── Samples_<N>/
└── *_samples.csv


Key assumptions:
- Each CSV file contains only **X values** (shape: `n_samples × dimension`),
- Directory names encode metadata:
  - `Dimension_<D>`
  - `seed_<SEED>`
  - `Samples_<N>`
- `<objective_type>` is a folder name such as `ELA_extraction` or `reduction`.

---

## What the Script Does

For **each X-sample file**, the script:

1. Loads the samples into a pandas DataFrame,
2. Loops over:
   - BBOB function IDs: `1 → 24`,
   - BBOB instances: `0 → 14`,
3. Evaluates the BBOB function at every sampled point,
4. Stores the resulting objective values in a CSV file.

Each evaluation produces a single-column file:


Key assumptions:
- Each CSV file contains only **X values** (shape: `n_samples × dimension`),
- Directory names encode metadata:
  - `Dimension_<D>`
  - `seed_<SEED>`
  - `Samples_<N>`
- `<objective_type>` is a folder name such as `ELA_extraction` or `reduction`.

---

## What the Script Does

For **each X-sample file**, the script:

1. Loads the samples into a pandas DataFrame,
2. Loops over:
   - BBOB function IDs: `1 → 24`,
   - Instances: `0 → 14`,
3. Evaluates the BBOB function at every sampled point,
4. Stores the resulting objective values in a CSV file.

Each evaluation produces a single-column file:
`*evaluations.csv`


containing:
fX
---

## Output Structure

The computed objective values are saved under:
bbob_evaluations/
└── <objective_type>/
└── Dimension_<D>/
└── seed_<SEED>/
└── Samples_<N>/
└── f_<FUNCTION_ID>/
└── id_<INSTANCE>/
└── evaluations.csv


Each `evaluations.csv` file corresponds to:
- one BBOB function,
- one BBOB instance,
- one fixed set of X samples.

If an output directory already exists, the script **skips recomputation** for that function–instance pair.

---

## How to Run

At present, the input directory is defined directly inside the script:

```python
directory = Path("x_samples/ELA_extraction/Dimension_20")
```

To run Step 2:

1. Edit the directory path in y_sampling.py to point to your X-sample root,
2. Execute:
  `python y_sampling.py`

## Step 3 — Exploratory Landscape Analysis (ELA) Feature Extraction

Step 3 converts the raw `(X, fX)` datasets generated in Steps 1 and 2 into **high-level Exploratory Landscape Analysis (ELA) features**.

This step is responsible for transforming sampled optimization landscapes into **fixed-dimensional feature vectors** suitable for:
- algorithm selection,
- performance prediction,
- surrogate modeling,
- or landscape classification.

Two complementary implementations are provided:

- `ela_sampling.py` — ELA feature extraction in the **original search space**
- `ela_embedding_sampling_fixed.py` — ELA feature extraction after **random dimensionality reduction**

---

## Purpose of Step 3

Given:
- input samples `X` (Step 1),
- objective evaluations `fX` (Step 2),

Step 3:
1. Matches corresponding `X` and `fX` files using directory metadata,
2. Computes a suite of **ELA features** using the `pflacco` library,
3. Saves one feature vector per:
   - BBOB function,
   - BBOB instance,
   - sampling configuration.

The result is a structured collection of CSV files, each representing a **numerical fingerprint of a landscape**.

---

## Variant A — Direct ELA (`ela_sampling.py`)

This script extracts ELA features **directly in the original problem dimension**.

### What it does

For each `(X, fX)` pair:
- Computes classical ELA feature groups:
  - meta-model features,
  - distribution features,
  - level-set features,
  - nearest-better clustering (NBC),
  - dispersion features,
  - information content,
  - PCA-based features,
- Computes **fitness–distance correlation (FDC)** using the known BBOB optimum,
- Produces **one feature vector per function–instance pair**.

### Characteristics

- No dimensionality reduction
- One-shot feature extraction (no bootstrapping)
- Best suited for **low-to-moderate dimensions**
- Fast and deterministic given fixed inputs

### Output (conceptual)
ela_features_2/
└── <objective_type>/
└── Dimension_<D>/
└── seed_<SEED>/
└── Samples_<N>/
└── f_<FUNCTION_ID>/
└── id_<INSTANCE>/
└── ela_features.csv


---

## Variant B — Embedded & Bootstrapped ELA (`ela_embedding_sampling_fixed.py`)

This script extends classical ELA by applying **Gaussian random embeddings** and **bootstrap resampling** before feature extraction.

### What it does

For each `(X, fX)` pair:
1. Reduces dimensionality using **Gaussian Random Embeddings**:
   - reduced dimension = `reduction_ratio × original dimension`,
2. Repeats the embedding with multiple random seeds,
3. For each embedding:
   - draws multiple bootstrap subsamples,
   - extracts ELA features on the reduced space,
4. Stores both:
   - learned embedding models,
   - multiple ELA feature realizations.

### Why this exists

- Classical ELA features degrade in high dimensions
- Random embeddings:
  - stabilize feature estimates,
  - reduce sample complexity,
  - enable ELA in higher-dimensional settings
- Bootstrapping captures **feature uncertainty and variability**

### Characteristics

- Dimensionality reduction + stochasticity
- Multiple feature vectors per function–instance
- Significantly more computationally expensive
- Designed for **robust analysis in higher dimensions**

### Output (conceptual)
ela_features_reduced_2/
└── <objective_type>/
└── D_<D>/
└── s_<SEED>/
└── N_<N>/
└── f_<FUNCTION_ID>/
└── id_<INSTANCE>/
└── r_ratio_<R>/
├── models/
│ └── random_embedding_seed_.pkl
└── features/
└── random_embedding_seed_/
└── round_*.csv


---

## Relationship Between the Two Scripts

| Aspect | `ela_sampling.py` | `ela_embedding_sampling_fixed.py` |
|-----|------------------|-----------------------------------|
| Feature space | Original dimension | Reduced dimension |
| Randomness | None | Embeddings + bootstraps |
| Output size | One feature vector | Many feature vectors |
| Best for | Low dimensions | Medium–high dimensions |
| Purpose | Baseline ELA | Robust / stabilized ELA |

Both scripts **consume exactly the same `(X, fX)` data** produced by Steps 1 and 2.

---

## Final Output of the Pipeline

After Step 3, the pipeline produces:
- structured `(X, fX)` datasets,
- one or more ELA feature representations per landscape,
- ready-to-use inputs for downstream or analysis tasks.


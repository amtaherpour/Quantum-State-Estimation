# Multi-Region Quantum State and Readout Estimation Framework

This repository implements a modular simulation and optimization framework for **multi-region quantum state estimation with readout-noise estimation**, based on the formulation in the associated paper.

The code is written to be:

- **fully modular**
- **easy to extend**
- **easy to run in Google Colab or Jupyter**
- **independent of specialized quantum libraries**

Even though the paper is quantum-inspired, the implementation is built using only standard numerical Python tools such as **NumPy** and **Matplotlib**. The core problem is treated as a structured optimization problem over:

- regional density matrices
- regional confusion matrices
- overlap consistency constraints

---

## Repository goal

This codebase lets you:

1. define a multi-region estimation problem,
2. simulate synthetic data,
3. run the alternating optimization algorithm,
4. evaluate the recovered regional states and confusion matrices,
5. run parameter sweeps and generate figures.

This is intended for:

- simulation studies,
- ablation studies,
- debugging theoretical ideas,
- producing paper figures,
- handing off to collaborators, assistants, or students.

---

## Main idea

The framework implements the following workflow:

1. Build an experiment configuration.
2. Generate a synthetic ground-truth state and readout-noise model.
3. Build regional POVMs.
4. Simulate regional ideal and noisy outcome distributions.
5. Sample counts / empirical frequencies.
6. Solve the alternating optimization problem:
   - inner ADMM for the regional state block
   - local projected-gradient updates for confusion matrices
7. Evaluate reconstruction quality.
8. Repeat across parameter settings to generate figures.

---

# Installation

## Minimal requirements

The code uses only standard Python scientific libraries:

- Python 3.10+
- NumPy
- Matplotlib

Install them with:

```bash
pip install numpy matplotlib
```

If you are using **Google Colab**, these are usually already available.

---

# Repository structure

This project has **14 modules**:

```text
core_ops.py
config.py
states.py
regions.py
measurements.py
noise.py
objectives.py
simulator.py
state_admm.py
confusion_solver.py
alternating_solver.py
metrics.py
experiments.py
main.py
```

Each module has a very specific role.

---

# Module-by-module overview

## 1. `core_ops.py`

Low-level numerical and matrix utilities.

### Purpose

This is the mathematical backbone of the codebase.

### It provides

- Hermitian conjugation (`dagger`)
- Hermitian / anti-Hermitian decomposition
- density-matrix validation
- projection onto the density-matrix set
- projection onto simplex / column-stochastic matrices
- Kronecker products
- subsystem permutation
- partial trace
- small matrix utilities

### Use this when

You need general-purpose linear algebra utilities or subsystem operations.

---

## 2. `config.py`

Experiment configuration system.

### Purpose

This is the central place where experiment structure and solver settings are defined.

### It provides

- `RegionConfig`
- `LossConfig`
- `ADMMConfig`
- `SimulationConfig`
- `ExperimentConfig`

### It also provides convenience builders

- `build_sliding_window_regions(...)`
- `build_pairwise_chain_regions(...)`
- `make_default_experiment_config()`

### Use this when

You want to define:

- number of sites,
- qubits per site,
- region layout,
- loss function,
- simulation seed,
- solver hyperparameters.

---

## 3. `states.py`

State generation, initialization, and overlap-consistency utilities.

### Purpose

This module creates quantum states and manages regional state objects.

### It provides

- random pure states
- random mixed states
- product states
- global-to-regional reductions
- overlap-consistent regional truth generation
- state initialization for optimization
- overlap consistency checks

### Use this when

You need:

- synthetic true states,
- initialized regional density matrices,
- overlap residual checks.

---

## 4. `regions.py`

Region and overlap graph structure.

### Purpose

This module encodes the structural relationships between regions.

### It provides

- `RegionGraph`
- overlap metadata
- neighbor relationships
- local/global site-index mappings
- overlap subsystem dimensions
- canonical keys for consensus and dual variables

### Use this when

You need clean access to:

- which regions overlap,
- which subsystems to keep in partial traces,
- overlap dimensions for ADMM.

---

## 5. `measurements.py`

POVM construction and measurement maps.

### Purpose

This module defines measurement models and computes Born-rule probabilities.

### It provides

POVM families:

- `computational`
- `random_ic`
- `pauli6_single_qubit`

It also provides:

- POVM validation
- measurement map
- adjoint measurement map
- expected counts
- config-based POVM builders

### Use this when

You need:

- POVMs for each region,
- predicted measurement probabilities from a state.

---

## 6. `noise.py`

Confusion-matrix construction and readout-noise handling.

### Purpose

This module manages regional readout-noise models.

### It provides

- identity confusion matrices
- uniform confusion matrices
- noisy-identity confusion matrices
- random column-stochastic confusion matrices
- confusion projection
- confusion validation
- application of confusion matrices to probability vectors
- config-based true/init/reference confusion builders

### Use this when

You need:

- synthetic readout noise,
- initialization of confusion matrices,
- regularization references.

---

## 7. `objectives.py`

Loss functions, gradients, and optimization diagnostics.

### Purpose

This module contains the scalar objectives and their gradients.

### It provides

Losses:

- `l2`
- `nll`

It also provides:

- gradients with respect to predicted probabilities
- gradients with respect to regional states
- gradients with respect to confusion matrices
- total objective evaluation
- overlap residual norms
- relative-change diagnostics

### Use this when

You need:

- objective values,
- gradients,
- convergence diagnostics.

---

## 8. `simulator.py`

Synthetic data generation.

### Purpose

This module generates complete simulated experiments.

### It provides

- overlap-consistent truth generation
- POVM construction
- true confusion generation
- ideal probabilities
- noisy probabilities
- multinomial counts
- empirical frequencies
- `SimulationResult`

### Use this when

You want to simulate the full pipeline before solving the optimization problem.

---

## 9. `state_admm.py`

Inner ADMM solver for the proximal state subproblem.

### Purpose

This module solves the state-update block of the alternating method.

### It provides

- partial-trace adjoint
- initialization of overlap consensus variables
- initialization of dual variables
- projected-gradient local regional state updates
- full inner ADMM loop
- `StateADMMResult`

### Use this when

You want to solve the state block for fixed confusion matrices.

---

## 10. `confusion_solver.py`

Local solver for confusion-matrix updates.

### Purpose

This module updates regional confusion matrices for fixed regional states.

### It provides

- local confusion objective
- confusion gradients
- projected-gradient solver
- batched updates across regions
- `ConfusionUpdateResult`

### Use this when

You want to solve the confusion block after the state block is updated.

---

## 11. `alternating_solver.py`

Outer alternating optimization loop.

### Purpose

This is the top-level optimizer that alternates between:

- state update via ADMM
- confusion update via local projected gradient

### It provides

- iterate initialization
- outer loop orchestration
- history tracking
- `AlternatingSolverResult`

### Use this when

You want to solve the full estimation problem.

---

## 12. `metrics.py`

Evaluation and diagnostics.

### Purpose

This module contains analysis-only functionality.

### It provides

- state errors
- confusion-matrix errors
- probability-fit errors
- overlap consistency summaries
- objective wrappers
- history summaries
- end-to-end solution summaries

### Use this when

You want to compare estimates to truth or summarize solver behavior.

---

## 13. `experiments.py`

Preset experiments and end-to-end runners.

### Purpose

This module defines reusable experiment presets and a simple end-to-end runner.

### It provides

Preset builders:

- `make_pairwise_chain_experiment(...)`
- `make_sliding_window_experiment(...)`
- `make_single_qubit_local_experiment(...)`
- `make_fast_debug_experiment()`
- `make_named_experiment(...)`

It also provides:

- `run_configured_experiment(...)`
- `run_named_experiment(...)`
- `ExperimentRunResult`

### Use this when

You want a clean preset or a one-call experiment execution.

---

## 14. `main.py`

Driver script for single runs, parameter sweeps, plotting, and saving.

### Purpose

This is the top-level user interface.

### It provides

- single experiment runs
- parameter sweeps
- history plotting
- sweep plotting
- saving JSON and CSV outputs

### Main entry points

- `run_single_experiment(...)`
- `run_parameter_sweep(...)`
- `plot_single_run_histories(...)`
- `plot_sweep_metric(...)`
- `save_single_run_summary(...)`
- `save_sweep_records_json(...)`
- `save_sweep_records_csv(...)`

### Use this when

You want to actually run studies and generate figures.

---

# Recommended workflow

If you are new to the repository, use this order:

## Step 1 — choose a preset experiment

Start with:

- `"fast_debug"`
- `"default"`
- `"pairwise_chain_small"`

## Step 2 — run one single experiment

Use `main.run_single_experiment(...)`.

## Step 3 — inspect output and plots

Check:

- objective history
- state residuals
- overlap residuals
- state and confusion errors

## Step 4 — run a parameter sweep

Use `main.run_parameter_sweep(...)`.

## Step 5 — save tables and figures

Use:

- `save_single_run_summary(...)`
- `save_sweep_records_json(...)`
- `save_sweep_records_csv(...)`

---

# Quick start

## Option A: Run from Python / Jupyter / Colab

```python
import importlib
import main
importlib.reload(main)

result = main.run_single_experiment(
    "fast_debug",
    make_plots=True,
    verbose=False,
)
```

This will:

- build the config,
- simulate data,
- solve the estimation problem,
- print a summary,
- generate plots.

---

## Option B: Run the demo from the command line

```bash
python main.py
```

This will:

- print available named experiments,
- run the demo single experiment.

---

# Available named experiments

You can list them with:

```python
import experiments
experiments.list_available_experiments()
```

Current named presets:

- `"default"`
- `"fast_debug"`
- `"pairwise_chain_small"`
- `"sliding_window_small"`
- `"single_qubit_local"`

---

# How to run a single experiment

## Minimal example

```python
import importlib
import main
importlib.reload(main)

result = main.run_single_experiment("fast_debug")
```

## Example with overrides

```python
result = main.run_single_experiment(
    "pairwise_chain_small",
    shots=2000,
    confusion_strength=0.05,
    lambda_confusion=1e-2,
    beta=1.0,
    gamma_rho=1.0,
    gamma_c=1.0,
    outer_max_iters=10,
    inner_max_iters=20,
    use_shot_noise=True,
    loss_name="nll",
    make_plots=True,
    verbose=False,
)
```

## What you get back

The return value is an `ExperimentRunResult` object containing:

- the full config,
- the simulation result,
- the solver result,
- a final metrics summary,
- summarized solver histories.

Useful fields:

```python
result.config
result.simulation
result.solver_result
result.summary
result.history_summary
```

---

# How to run a parameter sweep

## Example: sweep over number of shots

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="shots",
    values=[200, 400, 800, 1600],
    num_trials=3,
    verbose=True,
)
```

## Example: sweep over true readout-noise strength

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="confusion_strength",
    values=[0.0, 0.02, 0.05, 0.1],
    num_trials=3,
    verbose=True,
)
```

## Example: sweep over regularization strength

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="lambda_confusion",
    values=[0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    num_trials=3,
    verbose=True,
)
```

## Example: sweep over ADMM penalty

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="beta",
    values=[0.1, 0.5, 1.0, 2.0, 5.0],
    num_trials=3,
    verbose=True,
)
```

---

# Supported sweep parameters

Current supported parameter names in `main.py`:

- `"shots"`
- `"confusion_strength"`
- `"lambda_confusion"`
- `"beta"`
- `"gamma_rho"`
- `"gamma_c"`
- `"outer_max_iters"`
- `"inner_max_iters"`
- `"state_gd_max_iters"`
- `"confusion_gd_max_iters"`
- `"state_step_size"`
- `"confusion_step_size"`
- `"outer_tol"`
- `"inner_primal_tol"`
- `"inner_dual_tol"`
- `"state_gd_tol"`
- `"confusion_gd_tol"`
- `"use_shot_noise"`
- `"loss_name"`
- `"prob_floor"`

---

# How to make figures

## 1. Plot histories from one single run

```python
main.plot_single_run_histories(result)
```

This generates separate figures for:

- objective history
- state/confusion relative changes
- ADMM residuals
- max overlap residual
- confusion PG iterations

## 2. Plot a metric from a sweep

```python
main.plot_sweep_metric(
    records,
    metric_key="state_error_mean",
    x_key="parameter_value",
    xlabel="Shots",
    ylabel="Mean state error",
    title="Mean state error vs shots",
)
```

## Common sweep metrics to plot

You can plot any of these if present in the sweep records:

- `"runtime_sec"`
- `"final_objective"`
- `"fit_objective"`
- `"regularized_objective"`
- `"state_error_mean"`
- `"state_error_max"`
- `"confusion_error_mean"`
- `"confusion_error_max"`
- `"prob_l2_mean"`
- `"prob_l2_max"`
- `"prob_tv_mean"`
- `"prob_tv_max"`
- `"overlap_mean"`
- `"overlap_max"`
- `"final_state_primal_residual"`
- `"final_state_dual_residual"`
- `"final_state_max_overlap_residual"`
- `"outer_iterations"`

---

# How to save results

## Save a single run summary

```python
main.save_single_run_summary(
    result,
    save_dir="results",
    filename="my_run_summary.json",
)
```

This saves:

- config
- final metrics summary
- solver history summary
- raw solver history
- important scalar diagnostics

---

## Save sweep records to JSON

```python
main.save_sweep_records_json(
    records,
    save_dir="results",
    filename="sweep_records.json",
)
```

---

## Save sweep records to CSV

```python
main.save_sweep_records_csv(
    records,
    save_dir="results",
    filename="sweep_records.csv",
)
```

This is useful if you want to:

- open the results in Excel,
- send them to collaborators,
- make plots elsewhere.

---

# Typical study recipes

## Recipe 1 — State error vs number of shots

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="shots",
    values=[200, 400, 800, 1600, 3200],
    num_trials=5,
    verbose=True,
)

main.plot_sweep_metric(
    records,
    metric_key="state_error_mean",
    x_key="parameter_value",
    xlabel="Shots",
    ylabel="Mean state error",
    title="State error vs number of shots",
)
```

---

## Recipe 2 — Confusion estimation error vs readout-noise strength

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="confusion_strength",
    values=[0.0, 0.01, 0.03, 0.05, 0.1],
    num_trials=5,
    verbose=True,
)

main.plot_sweep_metric(
    records,
    metric_key="confusion_error_mean",
    x_key="parameter_value",
    xlabel="True confusion strength",
    ylabel="Mean confusion error",
    title="Confusion estimation error vs true readout noise",
)
```

---

## Recipe 3 — Effect of regularization parameter

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="lambda_confusion",
    values=[0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    num_trials=5,
    verbose=True,
)

main.plot_sweep_metric(
    records,
    metric_key="state_error_mean",
    x_key="parameter_value",
    xlabel="lambda_confusion",
    ylabel="Mean state error",
    title="State error vs confusion regularization",
)
```

---

## Recipe 4 — Runtime vs parameter

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="shots",
    values=[200, 400, 800, 1600],
    num_trials=3,
    verbose=True,
)

main.plot_sweep_metric(
    records,
    metric_key="runtime_sec",
    x_key="parameter_value",
    xlabel="Shots",
    ylabel="Runtime (sec)",
    title="Runtime vs shots",
)
```

---

# How to use custom configurations

You do not have to use only named presets.

You can build your own config using `experiments.py` or `config.py`.

## Example: build a custom sliding-window experiment

```python
import experiments
import main

cfg = experiments.make_sliding_window_experiment(
    num_sites=6,
    window_size=3,
    qubits_per_site=(1, 1, 1, 1, 1, 1),
    shots=1000,
    povm_type="random_ic",
    povm_num_outcomes=64,
    true_state_model="random_mixed",
    init_state_method="maximally_mixed",
    true_confusion_model="noisy_identity",
    init_confusion_method="identity",
    confusion_strength=0.05,
    loss_name="nll",
    seed=12345,
    experiment_name="my_custom_experiment",
)

result = main.run_single_experiment(
    cfg,
    make_plots=True,
    verbose=False,
)
```

---

# Recommended presets for different purposes

## For quick debugging

Use:

- `"fast_debug"`

Why:

- tiny problem
- fast runtime
- ideal for checking whether code works

## For small realistic tests

Use:

- `"pairwise_chain_small"`

Why:

- overlapping regions
- still manageable
- good first real benchmark

## For more overlap structure

Use:

- `"sliding_window_small"`

Why:

- more overlap relationships
- better for studying consistency effects

## For local one-qubit sanity checks

Use:

- `"single_qubit_local"`

Why:

- simplest local behavior
- useful for debugging POVMs and readout noise

---

# Interpretation of outputs

## Important solver quantities

### `initial_objective`

Objective at the beginning of the outer optimization.

### `final_objective`

Objective at the end of optimization.

### `final_state_primal_residual`

Final ADMM primal residual for overlap constraints.

### `final_state_dual_residual`

Final ADMM dual residual.

### `final_state_max_overlap_residual`

Largest overlap mismatch across all overlap constraints.

### `state_error_mean`

Average Frobenius error between estimated and true regional states.

### `confusion_error_mean`

Average Frobenius error between estimated and true regional confusion matrices.

### `prob_l2_mean`

Average probability mismatch between empirical and predicted probabilities.

---

# Self-tests

Most modules contain lightweight built-in self-tests.

Examples:

```python
import core_ops
core_ops.run_self_tests()
```

```python
import simulator
simulator.run_self_tests()
```

```python
import alternating_solver
alternating_solver.run_self_tests()
```

This is useful after editing the code.

---

# Suggested first-use checklist for a collaborator or assistant

If you are handing this repository to someone else, ask them to follow this exact order:

## Step 1

Install dependencies:

```bash
pip install numpy matplotlib
```

## Step 2

Open Python, Jupyter, or Colab and import the main driver:

```python
import main
```

## Step 3

Run the smallest demo:

```python
result = main.run_single_experiment("fast_debug")
```

## Step 4

Run a small sweep:

```python
records = main.run_parameter_sweep(
    "fast_debug",
    parameter_name="shots",
    values=[200, 400, 800],
    num_trials=2,
)
```

## Step 5

Plot and save the results:

```python
main.plot_sweep_metric(
    records,
    metric_key="state_error_mean",
    x_key="parameter_value",
    xlabel="Shots",
    ylabel="Mean state error",
    title="State error vs shots",
)

main.save_sweep_records_csv(records, save_dir="results", filename="shots_sweep.csv")
```

If those 5 steps work, the repository is ready for studies.

---

# Troubleshooting

## 1. “Old code is still running”

If you update a module in Jupyter or Colab, reload it:

```python
import importlib
import main
importlib.reload(main)
```

If several modules changed, restart the runtime/kernel.

---

## 2. A self-test fails after editing

Run the self-tests for the edited module and any dependent modules.

Example:

```python
import measurements
measurements.run_self_tests()
```

Then test a higher-level module afterward.

---

## 3. Runtime is slow

Reduce:

- number of regions
- shots
- outer iterations
- inner ADMM iterations
- projected-gradient iterations

Or use:

- `"fast_debug"`

---

## 4. Overlap residuals remain large

Possible causes:

- insufficient inner ADMM iterations
- very loose tolerances
- hard optimization settings
- large noise / poor initialization

Try:

- increasing `inner_max_iters`
- increasing `beta`
- increasing `outer_max_iters`

---

## 5. Confusion estimation looks poor

Try:

- increasing `lambda_confusion`
- increasing shots
- using `nll` loss
- starting from identity confusion
- reducing true confusion strength for debugging

---

# Notes and limitations

- This framework is designed for **small-to-moderate synthetic experiments**.
- It is not intended as a large-scale production tomography package.
- The current implementation prioritizes:

  - clarity,
  - modularity,
  - reproducibility,
  - figure generation,

  over heavy optimization or HPC scaling.

---

# Recommended citation / project note

If this code is used in connection with the associated paper, please cite the paper and mention that this repository implements the modular simulation and alternating optimization framework described there.

---

# Final recommendation

If you only remember one file, remember this:

## `main.py`

That is the top-level driver for:

- single experiments,
- parameter sweeps,
- plots,
- saving outputs.

Everything else supports it.


---

# Stabilized paper baseline

A tuned baseline preset is available for current paper-style studies:

- `"paper_pairwise_chain_baseline"`

This preset was added after a diagnostic tuning sequence on the realistic **NLL + shot-noise** regime.

## Baseline geometry

The current stabilized baseline uses:

- 4 sites
- 1 qubit per site
- pairwise chain regions:
  - `(0,1)`
  - `(1,2)`
  - `(2,3)`

This is the first geometry that was taken through a full debugging and stabilization workflow.

## Baseline measurement / truth setup

The preset uses:

- `povm_type = "random_ic"`
- `povm_num_outcomes = 16`
- `true_state_model = "random_mixed"`
- `init_state_method = "maximally_mixed"`
- `true_confusion_model = "noisy_identity"`
- `init_confusion_method = "identity"`
- `confusion_strength = 0.05`
- `loss_name = "nll"`
- `seed = 12347`
- shot noise enabled

## Stabilized solver settings

The following solver settings are currently the recommended baseline for this geometry:

- `beta = 4.0`
- `gamma_rho = 1.0`
- `gamma_c = 40.0`
- `lambda_confusion = 0.5`
- `outer_max_iters = 8`
- `inner_max_iters = 60`
- `state_gd_max_iters = 60`
- `confusion_gd_max_iters = 400`
- `confusion_step_size = 0.03`
- `confusion_gd_tol = 1e-6`

These settings were selected because they gave substantially better behavior in the realistic regime:

- cleaner monotone objective decrease,
- much smaller overlap residuals,
- much smaller state primal residuals,
- lower confusion error,
- reduced confusion projected-gradient workload relative to earlier tuned variants.

## Why this baseline exists

The purpose of this preset is to provide a **frozen, reproducible starting point** for current study generation.

It is meant to answer:

- state recovery vs shots,
- confusion recovery vs true readout-noise strength,
- recovery vs confusion regularization,
- solver-behavior summaries for the baseline geometry.

It is **not** claimed to be universally optimal for all geometries.

---

# Updated available named experiments

Current named presets now include:

- `"default"`
- `"fast_debug"`
- `"pairwise_chain_small"`
- `"sliding_window_small"`
- `"single_qubit_local"`
- `"paper_pairwise_chain_baseline"`

You can verify this with:

```python
import experiments
experiments.list_available_experiments()

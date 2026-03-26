# experiments.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np

from alternating_solver import AlternatingSolverResult, solve_alternating
from config import (
    ADMMConfig,
    ExperimentConfig,
    LossConfig,
    RegionConfig,
    SimulationConfig,
    build_pairwise_chain_regions,
    build_sliding_window_regions,
    make_default_experiment_config,
)
from metrics import summarize_history, summarize_solution
from noise import build_all_reference_confusions
from regions import RegionGraph
from simulator import SimulationResult, simulate_experiment


# ============================================================
# Small helpers
# ============================================================

def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _ensure_nonempty_string(value: str, name: str) -> str:
    value = str(value).strip()
    if len(value) == 0:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _default_qubits_per_site(num_sites: int) -> Tuple[int, ...]:
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    return tuple(1 for _ in range(num_sites))


def _normalize_qubits_per_site(
    num_sites: int,
    qubits_per_site: Optional[Tuple[int, ...]],
) -> Tuple[int, ...]:
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    if qubits_per_site is None:
        return _default_qubits_per_site(num_sites)

    q = tuple(int(v) for v in qubits_per_site)
    if len(q) != num_sites:
        raise ValueError(
            f"qubits_per_site must have length {num_sites}, got {len(q)}."
        )
    if any(v <= 0 for v in q):
        raise ValueError(f"All entries of qubits_per_site must be positive, got {q}.")
    return q


# ============================================================
# Preset config builders
# ============================================================

def make_pairwise_chain_experiment(
    *,
    num_sites: int = 3,
    qubits_per_site: Optional[Tuple[int, ...]] = None,
    shots: int = 1000,
    povm_type: str = "random_ic",
    povm_num_outcomes: Optional[int] = None,
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "noisy_identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.03,
    loss_name: str = "nll",
    prob_floor: float = 1e-12,
    seed: int = 12345,
    experiment_name: str = "pairwise_chain_experiment",
) -> ExperimentConfig:
    """
    Build a pairwise-chain experiment with regions
        (0,1), (1,2), ..., (n-2,n-1).

    This is the main recommended family for debugging and early experiments.
    """
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    qubits_per_site = _normalize_qubits_per_site(num_sites, qubits_per_site)
    shots = _ensure_positive_int(shots, "shots")
    experiment_name = _ensure_nonempty_string(experiment_name, "experiment_name")

    regions = build_pairwise_chain_regions(
        num_sites=num_sites,
        shots=shots,
        povm_type=povm_type,
        povm_num_outcomes=povm_num_outcomes,
        true_state_model=true_state_model,
        init_state_method=init_state_method,
        true_confusion_model=true_confusion_model,
        init_confusion_method=init_confusion_method,
        confusion_strength=confusion_strength,
        name_prefix="R",
    )

    return ExperimentConfig(
        qubits_per_site=qubits_per_site,
        regions=regions,
        loss=LossConfig(name=loss_name, prob_floor=prob_floor),
        admm=ADMMConfig(),
        simulation=SimulationConfig(seed=seed),
        experiment_name=experiment_name,
    )


def make_sliding_window_experiment(
    *,
    num_sites: int = 5,
    window_size: int = 3,
    qubits_per_site: Optional[Tuple[int, ...]] = None,
    shots: int = 1000,
    povm_type: str = "random_ic",
    povm_num_outcomes: Optional[int] = None,
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "noisy_identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.03,
    loss_name: str = "nll",
    prob_floor: float = 1e-12,
    seed: int = 12345,
    experiment_name: str = "sliding_window_experiment",
) -> ExperimentConfig:
    """
    Build a sliding-window experiment with regions
        (0,...,w-1), (1,...,w), ..., (n-w,...,n-1).
    """
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    window_size = _ensure_positive_int(window_size, "window_size")
    qubits_per_site = _normalize_qubits_per_site(num_sites, qubits_per_site)
    shots = _ensure_positive_int(shots, "shots")
    experiment_name = _ensure_nonempty_string(experiment_name, "experiment_name")

    regions = build_sliding_window_regions(
        num_sites=num_sites,
        window_size=window_size,
        shots=shots,
        povm_type=povm_type,
        povm_num_outcomes=povm_num_outcomes,
        true_state_model=true_state_model,
        init_state_method=init_state_method,
        true_confusion_model=true_confusion_model,
        init_confusion_method=init_confusion_method,
        confusion_strength=confusion_strength,
        name_prefix="R",
    )

    return ExperimentConfig(
        qubits_per_site=qubits_per_site,
        regions=regions,
        loss=LossConfig(name=loss_name, prob_floor=prob_floor),
        admm=ADMMConfig(),
        simulation=SimulationConfig(seed=seed),
        experiment_name=experiment_name,
    )


def make_single_qubit_local_experiment(
    *,
    num_sites: int = 4,
    shots: int = 1000,
    povm_type: str = "pauli6_single_qubit",
    true_state_model: str = "random_mixed",
    init_state_method: str = "maximally_mixed",
    true_confusion_model: str = "noisy_identity",
    init_confusion_method: str = "identity",
    confusion_strength: float = 0.03,
    loss_name: str = "nll",
    prob_floor: float = 1e-12,
    seed: int = 12345,
    experiment_name: str = "single_qubit_local_experiment",
) -> ExperimentConfig:
    """
    Build a non-overlapping single-qubit regional experiment with one site per region.

    This is useful for very small sanity checks and local-noise debugging.
    """
    num_sites = _ensure_positive_int(num_sites, "num_sites")
    shots = _ensure_positive_int(shots, "shots")
    experiment_name = _ensure_nonempty_string(experiment_name, "experiment_name")

    regions = tuple(
        RegionConfig(
            name=f"R{i}",
            sites=(i,),
            shots=shots,
            povm_type=povm_type,
            povm_num_outcomes=None,
            true_state_model=true_state_model,
            init_state_method=init_state_method,
            true_confusion_model=true_confusion_model,
            init_confusion_method=init_confusion_method,
            confusion_strength=confusion_strength,
        )
        for i in range(num_sites)
    )

    return ExperimentConfig(
        qubits_per_site=tuple(1 for _ in range(num_sites)),
        regions=regions,
        loss=LossConfig(name=loss_name, prob_floor=prob_floor),
        admm=ADMMConfig(),
        simulation=SimulationConfig(seed=seed),
        experiment_name=experiment_name,
    )


def make_fast_debug_experiment() -> ExperimentConfig:
    """
    Create a very small experiment suitable for quick smoke tests.

    Design
    ------
    - 3 sites, 1 qubit each
    - pairwise chain regions
    - L2 loss
    - shot noise disabled by default
    - small solver iteration counts for fast turnaround
    """
    cfg = make_pairwise_chain_experiment(
        num_sites=3,
        qubits_per_site=(1, 1, 1),
        shots=400,
        povm_type="random_ic",
        povm_num_outcomes=16,
        true_state_model="random_mixed",
        init_state_method="maximally_mixed",
        true_confusion_model="noisy_identity",
        init_confusion_method="identity",
        confusion_strength=0.03,
        loss_name="l2",
        seed=12345,
        experiment_name="fast_debug_experiment",
    )

    cfg.simulation.use_shot_noise = False
    cfg.admm.outer_max_iters = 2
    cfg.admm.inner_max_iters = 5
    cfg.admm.state_gd_max_iters = 20
    cfg.admm.confusion_gd_max_iters = 20
    cfg.admm.verbose = False
    return cfg


def make_named_experiment(name: str) -> ExperimentConfig:
    """
    Build one of the named preset experiments.

    Supported names
    ---------------
    - "default"
    - "fast_debug"
    - "pairwise_chain_small"
    - "sliding_window_small"
    - "single_qubit_local"
    """
    name = _ensure_nonempty_string(name, "name").lower()

    if name == "default":
        return make_default_experiment_config()

    if name == "fast_debug":
        return make_fast_debug_experiment()

    if name == "pairwise_chain_small":
        return make_pairwise_chain_experiment(
            num_sites=4,
            qubits_per_site=(1, 1, 1, 1),
            shots=800,
            povm_type="random_ic",
            povm_num_outcomes=16,
            true_state_model="random_mixed",
            init_state_method="maximally_mixed",
            true_confusion_model="noisy_identity",
            init_confusion_method="identity",
            confusion_strength=0.03,
            loss_name="nll",
            seed=12345,
            experiment_name="pairwise_chain_small",
        )

    if name == "sliding_window_small":
        return make_sliding_window_experiment(
            num_sites=5,
            window_size=3,
            qubits_per_site=(1, 1, 1, 1, 1),
            shots=800,
            povm_type="random_ic",
            povm_num_outcomes=64,  # 3 qubits -> dim=8 -> dim^2=64
            true_state_model="random_mixed",
            init_state_method="maximally_mixed",
            true_confusion_model="noisy_identity",
            init_confusion_method="identity",
            confusion_strength=0.03,
            loss_name="nll",
            seed=12345,
            experiment_name="sliding_window_small",
        )

    if name == "single_qubit_local":
        return make_single_qubit_local_experiment(
            num_sites=4,
            shots=1000,
            povm_type="pauli6_single_qubit",
            true_state_model="random_mixed",
            init_state_method="maximally_mixed",
            true_confusion_model="noisy_identity",
            init_confusion_method="identity",
            confusion_strength=0.03,
            loss_name="nll",
            seed=12345,
            experiment_name="single_qubit_local",
        )

    supported = (
        "default",
        "fast_debug",
        "pairwise_chain_small",
        "sliding_window_small",
        "single_qubit_local",
    )
    raise ValueError(f"Unknown experiment name '{name}'. Supported names: {supported}.")


def list_available_experiments() -> Tuple[str, ...]:
    """
    Return the available named preset experiments.
    """
    return (
        "default",
        "fast_debug",
        "pairwise_chain_small",
        "sliding_window_small",
        "single_qubit_local",
    )


# ============================================================
# End-to-end runner
# ============================================================

@dataclass
class ExperimentRunResult:
    """
    Bundle for a full experiment execution.

    Attributes
    ----------
    config :
        Experiment configuration used.
    simulation :
        Synthetic-data simulation bundle.
    solver_result :
        Outer alternating-solver result.
    summary :
        Final metrics summary.
    history_summary :
        Summary statistics for solver history arrays.
    """
    config: ExperimentConfig
    simulation: SimulationResult
    solver_result: AlternatingSolverResult
    summary: Dict[str, object]
    history_summary: Dict[str, Dict[str, float]]

    def validate(self) -> None:
        """
        Validate the combined run result.
        """
        graph = RegionGraph(self.config)
        self.simulation.validate()
        self.solver_result.validate(self.config, graph)

    def pretty_print(self) -> None:
        """
        Print a compact end-to-end run summary.
        """
        print("=" * 72)
        print("ExperimentRunResult")
        print("-" * 72)
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Solver converged: {self.solver_result.converged}")
        print(f"Outer iterations: {self.solver_result.num_outer_iterations}")
        print(f"Initial objective: {self.solver_result.initial_objective:.6e}")
        print(f"Final objective: {self.solver_result.final_objective:.6e}")

        if "state_error" in self.summary:
            print(
                f"Mean state error: "
                f"{self.summary['state_error']['aggregate']['mean']:.6e}"
            )
        if "confusion_error" in self.summary:
            print(
                f"Mean confusion error: "
                f"{self.summary['confusion_error']['aggregate']['mean']:.6e}"
            )

        print(
            f"Max overlap residual: "
            f"{self.solver_result.final_state_max_overlap_residual:.6e}"
        )
        print("=" * 72)


def run_configured_experiment(
    cfg: ExperimentConfig,
    *,
    truth_mode: str = "global_consistent",
    site_model_override: Optional[str] = None,
    initial_region_states: Optional[Mapping[str, np.ndarray]] = None,
    initial_region_confusions: Optional[Mapping[str, np.ndarray]] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    prob_floor: Optional[float] = None,
    verbose: Optional[bool] = None,
) -> ExperimentRunResult:
    """
    Run a full experiment end to end.

    Steps
    -----
    1) Simulate synthetic data from the config.
    2) Solve the alternating optimization problem.
    3) Summarize the estimated solution against the truth.
    """
    if verbose is None:
        verbose = cfg.admm.verbose

    simulation = simulate_experiment(
        cfg,
        truth_mode=truth_mode,
        site_model_override=site_model_override,
    )

    reference_confusions = build_all_reference_confusions(cfg)

    solver_result = solve_alternating(
        cfg=cfg,
        empirical_probabilities=simulation.empirical_probabilities,
        region_povms=simulation.region_povms,
        initial_region_states=initial_region_states,
        initial_region_confusions=initial_region_confusions,
        reference_confusions=reference_confusions,
        loss=loss,
        region_shots=simulation.region_shots,
        prob_floor=prob_floor,
        verbose=verbose,
    )

    summary = summarize_solution(
        cfg=cfg,
        empirical_probabilities=simulation.empirical_probabilities,
        region_states=solver_result.region_states,
        region_povms=simulation.region_povms,
        region_confusions=solver_result.region_confusions,
        reference_confusions=reference_confusions,
        true_region_states=simulation.region_states,
        true_region_confusions=simulation.region_confusions,
        loss=loss,
        region_shots=simulation.region_shots,
        prob_floor=prob_floor,
    )

    history_summary = summarize_history(solver_result.history)

    result = ExperimentRunResult(
        config=cfg,
        simulation=simulation,
        solver_result=solver_result,
        summary=summary,
        history_summary=history_summary,
    )
    result.validate()
    return result


def run_named_experiment(
    name: str,
    *,
    truth_mode: str = "global_consistent",
    site_model_override: Optional[str] = None,
    loss: Optional[Union[str, LossConfig]] = None,
    prob_floor: Optional[float] = None,
    verbose: Optional[bool] = None,
) -> ExperimentRunResult:
    """
    Convenience wrapper:
        cfg = make_named_experiment(name)
        run_configured_experiment(cfg, ...)
    """
    cfg = make_named_experiment(name)
    return run_configured_experiment(
        cfg,
        truth_mode=truth_mode,
        site_model_override=site_model_override,
        loss=loss,
        prob_floor=prob_floor,
        verbose=verbose,
    )


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_pairwise_builder() -> None:
    cfg = make_pairwise_chain_experiment(
        num_sites=4,
        qubits_per_site=(1, 1, 1, 1),
        shots=500,
        povm_type="random_ic",
        povm_num_outcomes=16,
        experiment_name="pairwise_test",
    )
    graph = RegionGraph(cfg)

    assert cfg.num_sites == 4
    assert cfg.num_regions == 3
    assert graph.overlap_pairs == ((0, 1), (1, 2))


def _self_test_sliding_builder() -> None:
    cfg = make_sliding_window_experiment(
        num_sites=5,
        window_size=3,
        qubits_per_site=(1, 1, 1, 1, 1),
        shots=500,
        povm_type="random_ic",
        povm_num_outcomes=64,
        experiment_name="sliding_test",
    )
    graph = RegionGraph(cfg)

    assert cfg.num_regions == 3
    # Because overlap means any nonempty intersection, R0 and R2 also overlap at site 2.
    assert graph.overlap_pairs == ((0, 1), (0, 2), (1, 2))


def _self_test_named_builder() -> None:
    cfg = make_named_experiment("fast_debug")
    assert cfg.experiment_name == "fast_debug_experiment"
    assert cfg.num_regions >= 1


def _self_test_end_to_end_run() -> None:
    cfg = make_fast_debug_experiment()
    result = run_configured_experiment(cfg, truth_mode="global_consistent", verbose=False)

    result.validate()
    assert result.solver_result.num_outer_iterations >= 1
    assert "fit_objective" in result.summary
    assert "objective" in result.history_summary


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the experiments module.
    """
    tests = [
        ("pairwise builder", _self_test_pairwise_builder),
        ("sliding-window builder", _self_test_sliding_builder),
        ("named preset builder", _self_test_named_builder),
        ("end-to-end experiment run", _self_test_end_to_end_run),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All experiments self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

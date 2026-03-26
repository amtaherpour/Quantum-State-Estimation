# simulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from config import ExperimentConfig
from measurements import POVM, build_all_region_povms, measurement_map
from noise import (
    apply_confusion_matrix,
    build_all_true_confusions,
    validate_region_confusion_collection,
)
from states import (
    generate_consistent_regional_truth_from_global_product,
    generate_independent_regional_truth,
    validate_region_state_collection,
)


# ============================================================
# Small helpers
# ============================================================

def _as_numpy_array(x, dtype=None) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _coerce_rng(rng=None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _normalize_probability_vector(p: np.ndarray, name: str) -> np.ndarray:
    p = _as_numpy_array(p, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(~np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(p < -1e-12):
        raise ValueError(f"{name} contains negative entries.")
    p = np.maximum(p, 0.0)
    s = float(np.sum(p))
    if s <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return p / s


def _resolve_global_site_model(cfg: ExperimentConfig, site_model_override: Optional[str]) -> str:
    if site_model_override is not None:
        return str(site_model_override)

    models = {region.true_state_model for region in cfg.regions}
    if len(models) == 1:
        return next(iter(models))

    raise ValueError(
        "The regions do not share a common true_state_model, so a consistent global truth "
        "cannot be inferred automatically. Pass site_model_override explicitly."
    )


# ============================================================
# Data containers
# ============================================================

@dataclass
class SimulationResult:
    """
    Bundle containing a complete synthetic experiment.

    Attributes
    ----------
    config :
        Experiment configuration used for simulation.
    global_state :
        Global truth density matrix, if generated.
    site_states :
        Site-local truth density matrices, if generated.
    region_states :
        True regional density matrices.
    region_povms :
        POVMs used in each region.
    region_confusions :
        True confusion matrices used in each region.
    ideal_probabilities :
        Ideal Born probability vectors per region.
    noisy_probabilities :
        Noisy probability vectors after confusion matrices.
    counts :
        Observed multinomial counts per region.
    empirical_probabilities :
        Empirical frequency vectors per region.
    region_shots :
        Number of shots per region.
    metadata :
        Extra bookkeeping information.
    """
    config: ExperimentConfig
    global_state: Optional[np.ndarray]
    site_states: Optional[Tuple[np.ndarray, ...]]
    region_states: Dict[str, np.ndarray]
    region_povms: Dict[str, POVM]
    region_confusions: Dict[str, np.ndarray]
    ideal_probabilities: Dict[str, np.ndarray]
    noisy_probabilities: Dict[str, np.ndarray]
    counts: Dict[str, np.ndarray]
    empirical_probabilities: Dict[str, np.ndarray]
    region_shots: Dict[str, int]
    metadata: Dict[str, object]

    def validate(self) -> None:
        """
        Validate internal consistency of the simulation bundle.
        """
        validate_region_state_collection(
            self.config,
            self.region_states,
            check_overlap_consistency=self.global_state is not None,
        )
        validate_region_confusion_collection(self.config, self.region_confusions)

        expected_names = {region.name for region in self.config.regions}

        collections = {
            "region_povms": self.region_povms,
            "ideal_probabilities": self.ideal_probabilities,
            "noisy_probabilities": self.noisy_probabilities,
            "counts": self.counts,
            "empirical_probabilities": self.empirical_probabilities,
            "region_shots": self.region_shots,
        }

        for label, mapping in collections.items():
            names = set(mapping.keys())
            if names != expected_names:
                missing = expected_names - names
                extra = names - expected_names
                raise ValueError(
                    f"{label} keys do not match configured region names. "
                    f"Missing={sorted(missing)}, extra={sorted(extra)}."
                )

        for region in self.config.regions:
            name = region.name
            m = self.region_povms[name].num_outcomes
            shots = int(self.region_shots[name])

            if self.ideal_probabilities[name].shape != (m,):
                raise ValueError(
                    f"ideal_probabilities['{name}'] has shape {self.ideal_probabilities[name].shape}, expected {(m,)}."
                )
            if self.noisy_probabilities[name].shape != (m,):
                raise ValueError(
                    f"noisy_probabilities['{name}'] has shape {self.noisy_probabilities[name].shape}, expected {(m,)}."
                )
            if self.empirical_probabilities[name].shape != (m,):
                raise ValueError(
                    f"empirical_probabilities['{name}'] has shape {self.empirical_probabilities[name].shape}, expected {(m,)}."
                )
            if self.counts[name].shape != (m,):
                raise ValueError(
                    f"counts['{name}'] has shape {self.counts[name].shape}, expected {(m,)}."
                )

            if int(np.sum(self.counts[name])) != shots:
                raise ValueError(
                    f"counts['{name}'] sums to {int(np.sum(self.counts[name]))}, expected {shots}."
                )

            if not np.isclose(np.sum(self.ideal_probabilities[name]), 1.0, atol=1e-10):
                raise ValueError(f"ideal_probabilities['{name}'] does not sum to 1.")
            if not np.isclose(np.sum(self.noisy_probabilities[name]), 1.0, atol=1e-10):
                raise ValueError(f"noisy_probabilities['{name}'] does not sum to 1.")
            if not np.isclose(np.sum(self.empirical_probabilities[name]), 1.0, atol=1e-10):
                raise ValueError(f"empirical_probabilities['{name}'] does not sum to 1.")

    def summary(self) -> Dict[str, object]:
        """
        Return a compact summary dictionary.
        """
        return {
            "experiment_name": self.config.experiment_name,
            "num_regions": self.config.num_regions,
            "region_names": tuple(region.name for region in self.config.regions),
            "has_global_state": self.global_state is not None,
            "seed": self.config.simulation.seed,
            "use_shot_noise": self.config.simulation.use_shot_noise,
        }

    def pretty_print(self) -> None:
        """
        Print a readable summary of the simulation result.
        """
        print("=" * 72)
        print("SimulationResult summary")
        print("-" * 72)
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Has global state: {self.global_state is not None}")
        print(f"Seed: {self.config.simulation.seed}")
        print(f"Use shot noise: {self.config.simulation.use_shot_noise}")
        print("-" * 72)
        for region in self.config.regions:
            name = region.name
            m = self.region_povms[name].num_outcomes
            shots = self.region_shots[name]
            print(
                f"[{name}] shots={shots}, dim={self.region_povms[name].dim}, "
                f"outcomes={m}, counts_sum={int(np.sum(self.counts[name]))}"
            )
        print("=" * 72)


# ============================================================
# Sampling helpers
# ============================================================

def sample_counts_from_probabilities(
    probabilities: np.ndarray,
    shots: int,
    rng=None,
) -> np.ndarray:
    """
    Sample multinomial counts from a probability vector.

    Parameters
    ----------
    probabilities :
        Probability vector.
    shots :
        Number of shots.
    rng :
        None, seed, or NumPy Generator.

    Returns
    -------
    np.ndarray
        Integer count vector.
    """
    p = _normalize_probability_vector(probabilities, "probabilities")
    shots = _ensure_positive_int(shots, "shots")
    rng = _coerce_rng(rng)

    counts = rng.multinomial(shots, pvals=p)
    return counts.astype(int)


def counts_to_empirical_probabilities(counts: np.ndarray) -> np.ndarray:
    """
    Convert a count vector to empirical frequencies.
    """
    counts = _as_numpy_array(counts, dtype=int).reshape(-1)
    if counts.size == 0:
        raise ValueError("counts must be non-empty.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    total = int(np.sum(counts))
    if total <= 0:
        raise ValueError("counts must sum to a positive integer.")
    return counts.astype(float) / float(total)


# ============================================================
# Region-level simulation
# ============================================================

def simulate_region_observation(
    rho: np.ndarray,
    povm: POVM,
    confusion: np.ndarray,
    shots: int,
    use_shot_noise: bool = True,
    rng=None,
) -> Dict[str, np.ndarray]:
    """
    Simulate one region's ideal probabilities, noisy probabilities, counts,
    and empirical frequencies.
    """
    shots = _ensure_positive_int(shots, "shots")
    rng = _coerce_rng(rng)

    ideal = measurement_map(rho, povm)
    noisy = apply_confusion_matrix(confusion, ideal, prob_floor=0.0)

    if use_shot_noise:
        counts = sample_counts_from_probabilities(noisy, shots=shots, rng=rng)
        empirical = counts_to_empirical_probabilities(counts)
    else:
        empirical = noisy.copy()
        counts = np.rint(shots * empirical).astype(int)

        # Fix rounding mismatch so counts sum exactly to shots.
        diff = shots - int(np.sum(counts))
        if diff != 0:
            idx = int(np.argmax(empirical))
            counts[idx] += diff

    return {
        "ideal_probabilities": ideal,
        "noisy_probabilities": noisy,
        "counts": counts,
        "empirical_probabilities": empirical,
    }


# ============================================================
# Full experiment simulation
# ============================================================

def simulate_experiment(
    cfg: ExperimentConfig,
    *,
    site_model_override: Optional[str] = None,
    truth_mode: str = "global_consistent",
    independent_rank: Optional[int] = None,
    confusion_dirichlet_concentration: float = 1.0,
) -> SimulationResult:
    """
    Simulate a complete synthetic experiment.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    site_model_override :
        Optional override for the site-level truth model when using
        truth_mode='global_consistent'.
    truth_mode :
        One of:
        - 'global_consistent'
        - 'independent_regions'
    independent_rank :
        Optional rank for independently generated random mixed states.
    confusion_dirichlet_concentration :
        Concentration parameter used when a true confusion model is
        random_column_stochastic.

    Returns
    -------
    SimulationResult
        Complete synthetic experiment bundle.

    Notes
    -----
    The recommended mode for the full project is 'global_consistent', since it
    guarantees overlap-consistent regional truth states.
    """
    truth_mode = str(truth_mode)
    if truth_mode not in {"global_consistent", "independent_regions"}:
        raise ValueError(
            f"truth_mode must be one of {{'global_consistent', 'independent_regions'}}, got '{truth_mode}'."
        )

    rng = cfg.simulation.make_rng()

    # --------------------------------------------------------
    # Truth states
    # --------------------------------------------------------
    if truth_mode == "global_consistent":
        site_model = _resolve_global_site_model(cfg, site_model_override)
        global_state, site_states, region_states = generate_consistent_regional_truth_from_global_product(
            cfg=cfg,
            site_model=site_model,
            rng=rng,
            rank=cfg.simulation.state_rank,
        )
        validate_region_state_collection(cfg, region_states, check_overlap_consistency=True)
    else:
        global_state = None
        site_states = None
        region_states = generate_independent_regional_truth(
            cfg=cfg,
            rng=rng,
            rank=independent_rank if independent_rank is not None else cfg.simulation.state_rank,
        )
        validate_region_state_collection(cfg, region_states, check_overlap_consistency=False)

    # --------------------------------------------------------
    # POVMs and true confusions
    # --------------------------------------------------------
    region_povms = build_all_region_povms(cfg, rng=rng)
    region_confusions = build_all_true_confusions(
        cfg,
        rng=rng,
        concentration=confusion_dirichlet_concentration,
    )

    validate_region_confusion_collection(cfg, region_confusions)

    # --------------------------------------------------------
    # Region-wise observation generation
    # --------------------------------------------------------
    ideal_probabilities: Dict[str, np.ndarray] = {}
    noisy_probabilities: Dict[str, np.ndarray] = {}
    counts: Dict[str, np.ndarray] = {}
    empirical_probabilities: Dict[str, np.ndarray] = {}
    region_shots: Dict[str, int] = {}

    for region in cfg.regions:
        name = region.name
        shots = int(region.shots)
        region_shots[name] = shots

        record = simulate_region_observation(
            rho=region_states[name],
            povm=region_povms[name],
            confusion=region_confusions[name],
            shots=shots,
            use_shot_noise=cfg.simulation.use_shot_noise,
            rng=rng,
        )

        ideal_probabilities[name] = record["ideal_probabilities"]
        noisy_probabilities[name] = record["noisy_probabilities"]
        counts[name] = record["counts"]
        empirical_probabilities[name] = record["empirical_probabilities"]

    result = SimulationResult(
        config=cfg,
        global_state=global_state,
        site_states=site_states,
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
        ideal_probabilities=ideal_probabilities,
        noisy_probabilities=noisy_probabilities,
        counts=counts,
        empirical_probabilities=empirical_probabilities,
        region_shots=region_shots,
        metadata={
            "truth_mode": truth_mode,
            "site_model_override": site_model_override,
            "confusion_dirichlet_concentration": float(confusion_dirichlet_concentration),
        },
    )
    result.validate()
    return result


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_count_sampling() -> None:
    p = np.array([0.1, 0.2, 0.7], dtype=float)
    counts = sample_counts_from_probabilities(p, shots=1000, rng=123)
    empirical = counts_to_empirical_probabilities(counts)

    assert counts.shape == (3,)
    assert int(np.sum(counts)) == 1000
    assert empirical.shape == (3,)
    assert np.isclose(np.sum(empirical), 1.0, atol=1e-12)


def _self_test_region_simulation() -> None:
    from measurements import make_computational_povm
    from noise import make_noisy_identity_confusion

    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    povm = make_computational_povm(2)
    confusion = make_noisy_identity_confusion(2, strength=0.2)

    record = simulate_region_observation(
        rho=rho,
        povm=povm,
        confusion=confusion,
        shots=500,
        use_shot_noise=True,
        rng=123,
    )

    assert record["ideal_probabilities"].shape == (2,)
    assert record["noisy_probabilities"].shape == (2,)
    assert record["counts"].shape == (2,)
    assert record["empirical_probabilities"].shape == (2,)
    assert int(np.sum(record["counts"])) == 500
    assert np.isclose(np.sum(record["ideal_probabilities"]), 1.0, atol=1e-12)
    assert np.isclose(np.sum(record["noisy_probabilities"]), 1.0, atol=1e-12)
    assert np.isclose(np.sum(record["empirical_probabilities"]), 1.0, atol=1e-12)


def _self_test_full_simulation_global_consistent() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    result = simulate_experiment(cfg, truth_mode="global_consistent")

    result.validate()
    assert result.global_state is not None
    assert result.site_states is not None
    assert set(result.region_states.keys()) == {region.name for region in cfg.regions}
    assert set(result.region_povms.keys()) == {region.name for region in cfg.regions}
    assert set(result.region_confusions.keys()) == {region.name for region in cfg.regions}


def _self_test_full_simulation_independent_regions() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    result = simulate_experiment(cfg, truth_mode="independent_regions")

    result.validate()
    assert result.global_state is None
    assert result.site_states is None
    assert set(result.region_states.keys()) == {region.name for region in cfg.regions}


def _self_test_no_shot_noise() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    cfg.simulation.use_shot_noise = False  # dataclass is mutable

    result = simulate_experiment(cfg, truth_mode="global_consistent")
    result.validate()

    for name in result.region_states.keys():
        assert np.allclose(
            result.empirical_probabilities[name],
            result.noisy_probabilities[name],
            atol=1e-12,
        )


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the simulator module.
    """
    tests = [
        ("count sampling", _self_test_count_sampling),
        ("single-region simulation", _self_test_region_simulation),
        ("full simulation: global consistent", _self_test_full_simulation_global_consistent),
        ("full simulation: independent regions", _self_test_full_simulation_independent_regions),
        ("full simulation without shot noise", _self_test_no_shot_noise),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All simulator self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    result = simulate_experiment(cfg, truth_mode="global_consistent")
    result.pretty_print()

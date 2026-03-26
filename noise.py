# noise.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np

from config import ExperimentConfig, RegionConfig
from core_ops import (
    frobenius_norm,
    is_column_stochastic,
    normalize_probability_vector,
    project_to_column_stochastic,
)


# ============================================================
# Numerical defaults
# ============================================================

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8
DEFAULT_PROB_FLOOR = 1e-12


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


def _ensure_nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _ensure_probability(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    return value


def _coerce_rng(rng=None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _region_obj(cfg: ExperimentConfig, region: Union[RegionConfig, str]) -> RegionConfig:
    if isinstance(region, RegionConfig):
        return region
    return cfg.region_by_name(region)


# ============================================================
# Outcome-count resolution
# ============================================================

def resolve_region_num_outcomes(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
) -> int:
    """
    Resolve the number of measurement outcomes for a region from the config.

    This mirrors the conventions used by the POVM builder logic:
    - computational: number of outcomes = region dimension
    - pauli6_single_qubit: 6
    - random_ic: region.povm_num_outcomes if given, else dim^2
    """
    region_obj = _region_obj(cfg, region)
    dim = cfg.region_dimension(region_obj)

    if region_obj.povm_type == "computational":
        return dim
    if region_obj.povm_type == "pauli6_single_qubit":
        return 6
    if region_obj.povm_type == "random_ic":
        return region_obj.povm_num_outcomes if region_obj.povm_num_outcomes is not None else dim * dim

    raise ValueError(
        f"Unsupported povm_type '{region_obj.povm_type}' for region '{region_obj.name}'."
    )


# ============================================================
# Confusion-matrix validation and projection
# ============================================================

def validate_confusion_matrix(
    c: np.ndarray,
    num_outcomes: Optional[int] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """
    Validate a column-stochastic confusion matrix.

    A valid confusion matrix C satisfies:
    - C is square
    - C_{ij} >= 0
    - each column sums to 1
    """
    c = _as_numpy_array(c, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"Confusion matrix must be 2D, got shape {c.shape}.")
    if c.shape[0] != c.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape {c.shape}.")

    if num_outcomes is not None:
        num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
        if c.shape != (num_outcomes, num_outcomes):
            raise ValueError(
                f"Confusion matrix has shape {c.shape}, expected {(num_outcomes, num_outcomes)}."
            )

    if np.any(c < -atol):
        min_entry = float(np.min(c))
        raise ValueError(
            f"Confusion matrix contains negative entries below tolerance. "
            f"Minimum entry = {min_entry:.6e}."
        )

    col_sums = np.sum(c, axis=0)
    if not np.allclose(col_sums, 1.0, atol=atol, rtol=rtol):
        raise ValueError(
            "Confusion matrix columns do not sum to 1 within tolerance. "
            f"Column sums = {col_sums}."
        )


def project_confusion_matrix(c: np.ndarray) -> np.ndarray:
    """
    Project a real matrix onto the set of column-stochastic matrices
    by simplex-projecting each column.
    """
    c = _as_numpy_array(c, dtype=float)
    return project_to_column_stochastic(c)


def is_valid_confusion_matrix(
    c: np.ndarray,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """
    Return True if the matrix is column-stochastic.
    """
    return is_column_stochastic(_as_numpy_array(c, dtype=float), atol=atol, rtol=rtol)


# ============================================================
# Standard confusion-matrix constructors
# ============================================================

def make_identity_confusion(num_outcomes: int) -> np.ndarray:
    """
    Identity confusion matrix, corresponding to ideal readout.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    return np.eye(num_outcomes, dtype=float)


def make_uniform_confusion(num_outcomes: int) -> np.ndarray:
    """
    Uniform confusion matrix: every recorded outcome is equally likely
    regardless of the ideal outcome.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    return np.full((num_outcomes, num_outcomes), 1.0 / num_outcomes, dtype=float)


def make_noisy_identity_confusion(
    num_outcomes: int,
    strength: float = 0.05,
) -> np.ndarray:
    """
    Noisy-identity confusion matrix:
        C = (1 - strength) I + strength U,
    where U is the uniform column-stochastic matrix.

    Parameters
    ----------
    num_outcomes :
        Number of outcomes.
    strength :
        Mixing weight in [0, 1].

    Returns
    -------
    np.ndarray
        Column-stochastic confusion matrix.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    strength = _ensure_probability(strength, "strength")

    eye = make_identity_confusion(num_outcomes)
    uni = make_uniform_confusion(num_outcomes)
    c = (1.0 - strength) * eye + strength * uni
    c = project_confusion_matrix(c)
    validate_confusion_matrix(c, num_outcomes=num_outcomes)
    return c


def make_random_column_stochastic_confusion(
    num_outcomes: int,
    rng=None,
    concentration: float = 1.0,
) -> np.ndarray:
    """
    Sample a random column-stochastic confusion matrix by drawing each column
    independently from a Dirichlet distribution.

    Parameters
    ----------
    num_outcomes :
        Number of outcomes.
    rng :
        None, integer seed, or NumPy Generator.
    concentration :
        Dirichlet concentration parameter. Must be positive.
        Smaller values produce spikier columns.

    Returns
    -------
    np.ndarray
        Random column-stochastic matrix.
    """
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    concentration = float(concentration)
    if concentration <= 0.0:
        raise ValueError(f"concentration must be positive, got {concentration}.")

    rng = _coerce_rng(rng)
    alpha = np.full(num_outcomes, concentration, dtype=float)

    c = np.zeros((num_outcomes, num_outcomes), dtype=float)
    for j in range(num_outcomes):
        c[:, j] = rng.dirichlet(alpha)

    validate_confusion_matrix(c, num_outcomes=num_outcomes)
    return c


# ============================================================
# Applying readout noise
# ============================================================

def apply_confusion_matrix(
    c: np.ndarray,
    ideal_probabilities: np.ndarray,
    prob_floor: float = 0.0,
) -> np.ndarray:
    """
    Apply a confusion matrix to an ideal probability vector:
        p_noisy = C p_ideal.

    Parameters
    ----------
    c :
        Column-stochastic confusion matrix.
    ideal_probabilities :
        Probability vector over ideal outcomes.
    prob_floor :
        Optional nonnegative floor after application and before renormalization.

    Returns
    -------
    np.ndarray
        Noisy probability vector.
    """
    c = _as_numpy_array(c, dtype=float)
    validate_confusion_matrix(c)

    p = _as_numpy_array(ideal_probabilities, dtype=float).reshape(-1)
    if p.size != c.shape[1]:
        raise ValueError(
            f"ideal_probabilities has length {p.size}, but confusion matrix expects length {c.shape[1]}."
        )

    p = normalize_probability_vector(p, floor=0.0)
    q = c @ p
    q = np.real(q)

    if prob_floor < 0.0:
        raise ValueError(f"prob_floor must be non-negative, got {prob_floor}.")

    if prob_floor > 0.0:
        q = np.maximum(q, prob_floor)
        q = q / np.sum(q)
    else:
        q = normalize_probability_vector(q, floor=0.0)

    return q


def apply_confusion_to_region_probabilities(
    region_probabilities: Dict[str, np.ndarray],
    region_confusions: Dict[str, np.ndarray],
    prob_floor: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Apply region-specific confusion matrices to region-specific probability vectors.
    """
    prob_names = set(region_probabilities.keys())
    conf_names = set(region_confusions.keys())

    missing = prob_names - conf_names
    extra = conf_names - prob_names

    if missing:
        raise ValueError(f"Missing confusion matrices for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected confusion matrices for regions: {sorted(extra)}.")

    out: Dict[str, np.ndarray] = {}
    for name, p in region_probabilities.items():
        out[name] = apply_confusion_matrix(
            region_confusions[name],
            p,
            prob_floor=prob_floor,
        )
    return out


# ============================================================
# Regularization helpers
# ============================================================

def confusion_frobenius_regularizer(
    c: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Squared Frobenius regularizer:
        ||C - C_ref||_F^2
    """
    c = _as_numpy_array(c, dtype=float)
    reference = _as_numpy_array(reference, dtype=float)

    if c.shape != reference.shape:
        raise ValueError(
            f"c and reference must have the same shape, got {c.shape} and {reference.shape}."
        )

    return float(np.sum((c - reference) ** 2))


def confusion_identity_distance(c: np.ndarray) -> float:
    """
    Frobenius distance from the identity confusion matrix.
    """
    c = _as_numpy_array(c, dtype=float)
    validate_confusion_matrix(c)
    return frobenius_norm(c - np.eye(c.shape[0], dtype=float))


# ============================================================
# Config-based builders
# ============================================================

def build_true_region_confusion(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng=None,
    concentration: float = 1.0,
) -> np.ndarray:
    """
    Build the ground-truth confusion matrix for one region using
    region.true_confusion_model.
    """
    region_obj = _region_obj(cfg, region)
    m = resolve_region_num_outcomes(cfg, region_obj)
    model = region_obj.true_confusion_model

    if model == "identity":
        return make_identity_confusion(m)

    if model == "noisy_identity":
        return make_noisy_identity_confusion(
            num_outcomes=m,
            strength=region_obj.confusion_strength,
        )

    if model == "random_column_stochastic":
        return make_random_column_stochastic_confusion(
            num_outcomes=m,
            rng=rng,
            concentration=concentration,
        )

    raise ValueError(
        f"Unsupported true_confusion_model '{model}' for region '{region_obj.name}'."
    )


def build_initial_region_confusion(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng=None,
) -> np.ndarray:
    """
    Build the initial confusion matrix for one region using
    region.init_confusion_method.
    """
    region_obj = _region_obj(cfg, region)
    m = resolve_region_num_outcomes(cfg, region_obj)
    method = region_obj.init_confusion_method

    if method == "identity":
        return make_identity_confusion(m)

    if method == "uniform":
        return make_uniform_confusion(m)

    if method == "noisy_identity":
        return make_noisy_identity_confusion(
            num_outcomes=m,
            strength=region_obj.confusion_strength,
        )

    raise ValueError(
        f"Unsupported init_confusion_method '{method}' for region '{region_obj.name}'."
    )


def build_reference_region_confusion(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
) -> np.ndarray:
    """
    Build the reference confusion matrix used in regularization.

    At the current config stage, only 'identity' is supported.
    """
    region_obj = _region_obj(cfg, region)
    m = resolve_region_num_outcomes(cfg, region_obj)

    if region_obj.reference_confusion_type == "identity":
        return make_identity_confusion(m)

    raise ValueError(
        f"Unsupported reference_confusion_type '{region_obj.reference_confusion_type}' "
        f"for region '{region_obj.name}'."
    )


def build_all_true_confusions(
    cfg: ExperimentConfig,
    rng=None,
    concentration: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Build ground-truth confusion matrices for all regions.
    """
    rng = _coerce_rng(rng)
    out: Dict[str, np.ndarray] = {}
    for region in cfg.regions:
        out[region.name] = build_true_region_confusion(
            cfg=cfg,
            region=region,
            rng=rng,
            concentration=concentration,
        )
    return out


def build_all_initial_confusions(
    cfg: ExperimentConfig,
    rng=None,
) -> Dict[str, np.ndarray]:
    """
    Build initial confusion matrices for all regions.
    """
    rng = _coerce_rng(rng)
    out: Dict[str, np.ndarray] = {}
    for region in cfg.regions:
        out[region.name] = build_initial_region_confusion(
            cfg=cfg,
            region=region,
            rng=rng,
        )
    return out


def build_all_reference_confusions(
    cfg: ExperimentConfig,
) -> Dict[str, np.ndarray]:
    """
    Build reference confusion matrices for all regions.
    """
    out: Dict[str, np.ndarray] = {}
    for region in cfg.regions:
        out[region.name] = build_reference_region_confusion(cfg, region)
    return out


# ============================================================
# Validation for collections
# ============================================================

def validate_region_confusion_collection(
    cfg: ExperimentConfig,
    confusions: Dict[str, np.ndarray],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """
    Validate a region-name -> confusion-matrix mapping against the config.
    """
    expected_names = {region.name for region in cfg.regions}
    provided_names = set(confusions.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names

    if missing:
        raise ValueError(f"Missing confusion matrices for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected region names in confusion collection: {sorted(extra)}.")

    for region in cfg.regions:
        m = resolve_region_num_outcomes(cfg, region)
        validate_confusion_matrix(
            confusions[region.name],
            num_outcomes=m,
            atol=atol,
            rtol=rtol,
        )


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_basic_constructors() -> None:
    m = 5
    eye = make_identity_confusion(m)
    uni = make_uniform_confusion(m)
    noisy = make_noisy_identity_confusion(m, strength=0.2)

    validate_confusion_matrix(eye, num_outcomes=m)
    validate_confusion_matrix(uni, num_outcomes=m)
    validate_confusion_matrix(noisy, num_outcomes=m)

    assert np.allclose(eye, np.eye(m))
    assert np.allclose(np.sum(uni, axis=0), 1.0)
    assert np.allclose(np.sum(noisy, axis=0), 1.0)


def _self_test_random_column_stochastic() -> None:
    m = 7
    c = make_random_column_stochastic_confusion(m, rng=123, concentration=0.7)
    validate_confusion_matrix(c, num_outcomes=m)
    assert c.shape == (m, m)
    assert np.all(c >= 0.0)
    assert np.allclose(np.sum(c, axis=0), 1.0, atol=1e-10)


def _self_test_apply_confusion() -> None:
    c = make_noisy_identity_confusion(3, strength=0.3)
    p = np.array([0.0, 1.0, 0.0], dtype=float)
    q = apply_confusion_matrix(c, p)

    assert q.shape == (3,)
    assert np.isclose(np.sum(q), 1.0, atol=1e-10)
    assert np.all(q >= 0.0)
    assert np.allclose(q, c[:, 1], atol=1e-10)


def _self_test_regularizer() -> None:
    c = make_noisy_identity_confusion(4, strength=0.1)
    ref = make_identity_confusion(4)

    val = confusion_frobenius_regularizer(c, ref)
    dist = confusion_identity_distance(c)

    assert val >= 0.0
    assert np.isclose(val, dist ** 2, atol=1e-10)


def _self_test_config_builders() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()

    true_confusions = build_all_true_confusions(cfg, rng=2024)
    init_confusions = build_all_initial_confusions(cfg, rng=2024)
    ref_confusions = build_all_reference_confusions(cfg)

    validate_region_confusion_collection(cfg, true_confusions)
    validate_region_confusion_collection(cfg, init_confusions)
    validate_region_confusion_collection(cfg, ref_confusions)

    assert set(true_confusions.keys()) == {region.name for region in cfg.regions}
    assert set(init_confusions.keys()) == {region.name for region in cfg.regions}
    assert set(ref_confusions.keys()) == {region.name for region in cfg.regions}

    for region in cfg.regions:
        m = resolve_region_num_outcomes(cfg, region)
        assert true_confusions[region.name].shape == (m, m)
        assert init_confusions[region.name].shape == (m, m)
        assert ref_confusions[region.name].shape == (m, m)


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the noise module.
    """
    tests = [
        ("basic confusion constructors", _self_test_basic_constructors),
        ("random column-stochastic confusion", _self_test_random_column_stochastic),
        ("apply confusion matrix", _self_test_apply_confusion),
        ("confusion regularizer", _self_test_regularizer),
        ("config-based confusion builders", _self_test_config_builders),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All noise self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

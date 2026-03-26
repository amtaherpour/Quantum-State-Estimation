# states.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from config import ExperimentConfig, RegionConfig
from core_ops import (
    frobenius_norm,
    is_density_matrix,
    kron_all,
    maximally_mixed,
    partial_trace,
    project_to_density_matrix,
    subsystem_dimensions_from_qubits,
)


# ============================================================
# Type aliases
# ============================================================

RNGInput = Optional[Union[int, np.random.Generator]]


# ============================================================
# Internal helpers
# ============================================================

def _coerce_rng(rng: RNGInput = None) -> np.random.Generator:
    """
    Convert `rng` into a NumPy Generator.

    Parameters
    ----------
    rng :
        None, integer seed, or an existing NumPy Generator.

    Returns
    -------
    np.random.Generator
    """
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _ensure_valid_rank(rank: Optional[int], dim: int) -> Optional[int]:
    if rank is None:
        return None
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive when provided, got {rank}.")
    if rank > dim:
        raise ValueError(f"rank={rank} cannot exceed dimension={dim}.")
    return rank


def _region_obj(region: Union[RegionConfig, str], cfg: ExperimentConfig) -> RegionConfig:
    if isinstance(region, RegionConfig):
        return region
    return cfg.region_by_name(region)


# ============================================================
# State-vector and density-matrix primitives
# ============================================================

def normalize_state_vector(psi: np.ndarray, atol: float = 1e-14) -> np.ndarray:
    """
    Normalize a complex state vector to unit Euclidean norm.

    Parameters
    ----------
    psi :
        Input complex vector.

    Returns
    -------
    np.ndarray
        Normalized complex vector.
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    if psi.size == 0:
        raise ValueError("psi must be non-empty.")
    norm = np.linalg.norm(psi)
    if norm <= atol:
        raise ValueError("Cannot normalize a numerically zero vector.")
    return psi / norm


def ket_to_density(psi: np.ndarray) -> np.ndarray:
    """
    Convert a state vector into a rank-1 density matrix.

    Parameters
    ----------
    psi :
        Complex state vector.

    Returns
    -------
    np.ndarray
        Density matrix |psi><psi|.
    """
    psi = normalize_state_vector(psi)
    return np.outer(psi, np.conjugate(psi))


def computational_basis_ket(index: int, dim: int) -> np.ndarray:
    """
    Return the computational-basis vector e_index in C^dim.
    """
    index = int(index)
    dim = _ensure_positive_int(dim, "dim")
    if index < 0 or index >= dim:
        raise ValueError(f"index must lie in [0, {dim - 1}], got {index}.")
    psi = np.zeros(dim, dtype=np.complex128)
    psi[index] = 1.0
    return psi


def random_complex_vector(dim: int, rng: RNGInput = None) -> np.ndarray:
    """
    Sample a complex Gaussian vector in C^dim.

    Real and imaginary parts are sampled independently from N(0,1).
    """
    dim = _ensure_positive_int(dim, "dim")
    rng = _coerce_rng(rng)
    real = rng.normal(size=dim)
    imag = rng.normal(size=dim)
    return real + 1j * imag


def random_pure_state_ket(dim: int, rng: RNGInput = None) -> np.ndarray:
    """
    Sample a random pure-state ket in C^dim using normalized complex Gaussian entries.
    """
    dim = _ensure_positive_int(dim, "dim")
    rng = _coerce_rng(rng)
    psi = random_complex_vector(dim, rng=rng)
    return normalize_state_vector(psi)


def random_pure_density_matrix(dim: int, rng: RNGInput = None) -> np.ndarray:
    """
    Sample a random pure-state density matrix in dimension `dim`.
    """
    psi = random_pure_state_ket(dim, rng=rng)
    return ket_to_density(psi)


def random_mixed_density_matrix(
    dim: int,
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> np.ndarray:
    """
    Sample a random mixed density matrix in dimension `dim`.

    Construction
    ------------
    Let G be a complex Gaussian matrix of shape (dim, rank). Then
        rho = G G^dagger / Tr(G G^dagger).

    If rank is None, a full-rank draw with rank=dim is used.

    Parameters
    ----------
    dim :
        Hilbert-space dimension.
    rng :
        None, seed, or NumPy Generator.
    rank :
        Optional target rank, with 1 <= rank <= dim.

    Returns
    -------
    np.ndarray
        A valid density matrix.
    """
    dim = _ensure_positive_int(dim, "dim")
    rng = _coerce_rng(rng)
    rank = _ensure_valid_rank(rank, dim)
    if rank is None:
        rank = dim

    real = rng.normal(size=(dim, rank))
    imag = rng.normal(size=(dim, rank))
    g = real + 1j * imag
    rho = g @ np.conjugate(g.T)
    rho = rho / np.trace(rho)
    rho = project_to_density_matrix(rho)
    return rho


def sample_density_matrix(
    dim: int,
    model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> np.ndarray:
    """
    Sample a density matrix according to the requested model.

    Supported models
    ----------------
    - "random_mixed"
    - "random_pure"
    - "maximally_mixed"
    """
    dim = _ensure_positive_int(dim, "dim")
    model = str(model)

    if model == "random_mixed":
        return random_mixed_density_matrix(dim, rng=rng, rank=rank)
    if model == "random_pure":
        return random_pure_density_matrix(dim, rng=rng)
    if model == "maximally_mixed":
        return maximally_mixed(dim)

    raise ValueError(
        f"Unsupported state model '{model}'. Supported: "
        f"{{'random_mixed', 'random_pure', 'maximally_mixed'}}."
    )


# ============================================================
# Product states
# ============================================================

def build_product_pure_ket(local_kets: Sequence[np.ndarray]) -> np.ndarray:
    """
    Build the tensor-product ket from a sequence of local kets.

    Parameters
    ----------
    local_kets :
        Sequence of state vectors.

    Returns
    -------
    np.ndarray
        Tensor-product state vector.
    """
    if len(local_kets) == 0:
        raise ValueError("local_kets must be a non-empty sequence.")
    normalized = [normalize_state_vector(psi) for psi in local_kets]
    return kron_all(normalized)


def build_product_density(local_states: Sequence[np.ndarray]) -> np.ndarray:
    """
    Build the tensor-product density matrix from a sequence of local density matrices.

    Parameters
    ----------
    local_states :
        Sequence of valid density matrices.

    Returns
    -------
    np.ndarray
        Tensor-product density matrix.
    """
    if len(local_states) == 0:
        raise ValueError("local_states must be a non-empty sequence.")
    for idx, rho in enumerate(local_states):
        if not is_density_matrix(rho):
            raise ValueError(f"local_states[{idx}] is not a valid density matrix.")
    return kron_all(local_states)


def generate_site_density_matrices(
    qubits_per_site: Sequence[int],
    model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Generate one density matrix per site.

    Parameters
    ----------
    qubits_per_site :
        Number of qubits at each site.
    model :
        State model used independently at each site.
    rng :
        None, seed, or NumPy Generator.
    rank :
        Optional rank for random mixed-state generation.

    Returns
    -------
    tuple[np.ndarray, ...]
        Site-local density matrices.
    """
    rng = _coerce_rng(rng)
    site_dims = subsystem_dimensions_from_qubits(qubits_per_site)
    states: List[np.ndarray] = []
    for dim in site_dims:
        states.append(sample_density_matrix(dim, model=model, rng=rng, rank=rank))
    return tuple(states)


def generate_global_product_state(
    qubits_per_site: Sequence[int],
    site_model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a global product density matrix over sites.

    Each site gets an independently sampled local density matrix, and the
    global state is their tensor product.

    Parameters
    ----------
    qubits_per_site :
        Number of qubits at each site.
    site_model :
        Local site-state model.
    rng :
        None, seed, or NumPy Generator.
    rank :
        Optional rank for random mixed states.

    Returns
    -------
    np.ndarray
        Global product density matrix.
    """
    local_states = generate_site_density_matrices(
        qubits_per_site=qubits_per_site,
        model=site_model,
        rng=rng,
        rank=rank,
    )
    return build_product_density(local_states)


# ============================================================
# Global-to-regional reductions
# ============================================================

def reduce_global_state_to_region(
    global_rho: np.ndarray,
    qubits_per_site: Sequence[int],
    region_sites: Sequence[int],
) -> np.ndarray:
    """
    Reduce a global site-factorized density operator to a chosen region.

    Parameters
    ----------
    global_rho :
        Global density matrix defined over all sites.
    qubits_per_site :
        Number of qubits at each site.
    region_sites :
        Site indices to keep.

    Returns
    -------
    np.ndarray
        Reduced regional density matrix.
    """
    site_dims = subsystem_dimensions_from_qubits(qubits_per_site)
    keep = tuple(int(s) for s in region_sites)
    return partial_trace(global_rho, dims=site_dims, keep=keep)


def reduce_global_state_to_all_regions(
    global_rho: np.ndarray,
    cfg: ExperimentConfig,
) -> Dict[str, np.ndarray]:
    """
    Reduce a global density matrix to all configured regions.

    Parameters
    ----------
    global_rho :
        Global density matrix over the full site tensor product.
    cfg :
        Experiment configuration.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping region name -> reduced density matrix.
    """
    out: Dict[str, np.ndarray] = {}
    for region in cfg.regions:
        out[region.name] = reduce_global_state_to_region(
            global_rho=global_rho,
            qubits_per_site=cfg.qubits_per_site,
            region_sites=region.sites,
        )
    return out


def generate_consistent_regional_truth_from_global_product(
    cfg: ExperimentConfig,
    site_model: str = "random_mixed",
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Dict[str, np.ndarray]]:
    """
    Generate an overlap-consistent family of regional states from a global product state.

    Returns
    -------
    tuple
        (global_rho, site_states, regional_states_dict)

    Notes
    -----
    This is the safest synthetic truth generator for the first full version
    because it guarantees that all regional reduced states are mutually
    overlap-consistent.
    """
    rng = _coerce_rng(rng)
    site_states = generate_site_density_matrices(
        qubits_per_site=cfg.qubits_per_site,
        model=site_model,
        rng=rng,
        rank=rank,
    )
    global_rho = build_product_density(site_states)
    region_states = reduce_global_state_to_all_regions(global_rho, cfg)
    return global_rho, site_states, region_states


# ============================================================
# Region-state initialization
# ============================================================

def initialize_region_state(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng: RNGInput = None,
    method: Optional[str] = None,
    rank: Optional[int] = None,
) -> np.ndarray:
    """
    Initialize one regional density matrix for optimization.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    region :
        Region object or region name.
    rng :
        None, integer seed, or Generator.
    method :
        Optional override of the region's configured init_state_method.
    rank :
        Optional rank used if the method is "random_mixed".

    Returns
    -------
    np.ndarray
        Valid density matrix for the region.
    """
    rng = _coerce_rng(rng)
    region_obj = _region_obj(region, cfg)
    dim = cfg.region_dimension(region_obj)
    chosen_method = region_obj.init_state_method if method is None else str(method)
    rho = sample_density_matrix(dim=dim, model=chosen_method, rng=rng, rank=rank)
    return project_to_density_matrix(rho)


def initialize_all_region_states(
    cfg: ExperimentConfig,
    rng: RNGInput = None,
    method_override: Optional[str] = None,
    rank: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Initialize all regional density matrices.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    rng :
        None, integer seed, or Generator.
    method_override :
        If provided, use this method for every region instead of each region's
        own configured init_state_method.
    rank :
        Optional rank for random mixed-state initialization.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping region name -> initialized density matrix.
    """
    rng = _coerce_rng(rng)
    out: Dict[str, np.ndarray] = {}
    for region in cfg.regions:
        out[region.name] = initialize_region_state(
            cfg=cfg,
            region=region,
            rng=rng,
            method=method_override,
            rank=rank,
        )
    return out


# ============================================================
# Truth generation from region-local configs
# ============================================================

def generate_independent_regional_truth(
    cfg: ExperimentConfig,
    rng: RNGInput = None,
    rank: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate regional truth states independently from each region's configured model.

    Warning
    -------
    This function does NOT guarantee overlap consistency across regions.
    It is intended only for debugging or deliberately inconsistent tests.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping region name -> regional density matrix.
    """
    rng = _coerce_rng(rng)
    out: Dict[str, np.ndarray] = {}
    for region in cfg.regions:
        dim = cfg.region_dimension(region)
        rho = sample_density_matrix(
            dim=dim,
            model=region.true_state_model,
            rng=rng,
            rank=rank,
        )
        out[region.name] = project_to_density_matrix(rho)
    return out


# ============================================================
# Overlap consistency checks
# ============================================================

def overlap_reduction_for_pair(
    cfg: ExperimentConfig,
    region_states: Dict[str, np.ndarray],
    region_a: Union[str, RegionConfig],
    region_b: Union[str, RegionConfig],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the two overlap reductions for a pair of regions.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    region_states :
        Mapping region name -> density matrix.
    region_a, region_b :
        Regions or region names.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reduced states on the shared overlap subsystem, ordered according to
        the global site order restricted to the overlap.
    """
    a = _region_obj(region_a, cfg)
    b = _region_obj(region_b, cfg)

    overlap_sites = cfg.region_overlap_sites(a, b)
    if len(overlap_sites) == 0:
        raise ValueError(
            f"Regions '{a.name}' and '{b.name}' do not overlap, so no overlap reduction exists."
        )

    rho_a = region_states[a.name]
    rho_b = region_states[b.name]

    local_keep_a = [a.sites.index(s) for s in overlap_sites]
    local_keep_b = [b.sites.index(s) for s in overlap_sites]

    dims_a = cfg.region_site_dimensions(a)
    dims_b = cfg.region_site_dimensions(b)

    red_a = partial_trace(rho_a, dims=dims_a, keep=local_keep_a)
    red_b = partial_trace(rho_b, dims=dims_b, keep=local_keep_b)
    return red_a, red_b


def pairwise_overlap_residual(
    cfg: ExperimentConfig,
    region_states: Dict[str, np.ndarray],
    region_a: Union[str, RegionConfig],
    region_b: Union[str, RegionConfig],
) -> float:
    """
    Frobenius norm of the overlap mismatch between two regions.
    """
    red_a, red_b = overlap_reduction_for_pair(cfg, region_states, region_a, region_b)
    return frobenius_norm(red_a - red_b)


def all_overlap_residuals(
    cfg: ExperimentConfig,
    region_states: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], float]:
    """
    Compute overlap residuals for all overlapping region pairs.

    Returns
    -------
    dict[(str, str), float]
        Pairwise Frobenius mismatch on every overlap.
    """
    out: Dict[Tuple[str, str], float] = {}
    for i, j in cfg.overlap_pairs():
        name_i = cfg.regions[i].name
        name_j = cfg.regions[j].name
        out[(name_i, name_j)] = pairwise_overlap_residual(
            cfg=cfg,
            region_states=region_states,
            region_a=name_i,
            region_b=name_j,
        )
    return out


def are_region_states_overlap_consistent(
    cfg: ExperimentConfig,
    region_states: Dict[str, np.ndarray],
    atol: float = 1e-8,
) -> bool:
    """
    Check whether all overlapping region pairs agree on their shared reductions.
    """
    residuals = all_overlap_residuals(cfg, region_states)
    return all(val <= atol for val in residuals.values())


# ============================================================
# Validation helpers for collections of states
# ============================================================

def validate_region_state_collection(
    cfg: ExperimentConfig,
    region_states: Dict[str, np.ndarray],
    check_overlap_consistency: bool = False,
    overlap_atol: float = 1e-8,
) -> None:
    """
    Validate a mapping region name -> density matrix.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    region_states :
        Mapping from region names to density matrices.
    check_overlap_consistency :
        If True, also validate pairwise overlap consistency.
    overlap_atol :
        Tolerance used for overlap-consistency checks.

    Raises
    ------
    ValueError
        If any state is missing, malformed, or inconsistent.
    """
    expected_names = {region.name for region in cfg.regions}
    provided_names = set(region_states.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names

    if missing:
        raise ValueError(f"Missing regional states for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected region names in state collection: {sorted(extra)}.")

    for region in cfg.regions:
        rho = np.asarray(region_states[region.name], dtype=np.complex128)
        dim = cfg.region_dimension(region)
        if rho.shape != (dim, dim):
            raise ValueError(
                f"State for region '{region.name}' has shape {rho.shape}, expected {(dim, dim)}."
            )
        if not is_density_matrix(rho):
            raise ValueError(f"State for region '{region.name}' is not a valid density matrix.")

    if check_overlap_consistency:
        residuals = all_overlap_residuals(cfg, region_states)
        bad = {k: v for k, v in residuals.items() if v > overlap_atol}
        if bad:
            raise ValueError(
                f"Regional states are not overlap-consistent within tolerance {overlap_atol}. "
                f"Residuals: {bad}"
            )


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_random_states() -> None:
    rng = np.random.default_rng(123)

    rho_pure = random_pure_density_matrix(dim=4, rng=rng)
    rho_mixed = random_mixed_density_matrix(dim=4, rng=rng, rank=3)

    assert is_density_matrix(rho_pure)
    assert is_density_matrix(rho_mixed)
    assert np.isclose(np.trace(rho_pure), 1.0, atol=1e-10)
    assert np.isclose(np.trace(rho_mixed), 1.0, atol=1e-10)


def _self_test_product_state() -> None:
    rho0 = random_pure_density_matrix(dim=2, rng=11)
    rho1 = random_mixed_density_matrix(dim=2, rng=22)
    rho = build_product_density([rho0, rho1])

    assert rho.shape == (4, 4)
    assert is_density_matrix(rho)

    red0 = partial_trace(rho, dims=[2, 2], keep=[0])
    red1 = partial_trace(rho, dims=[2, 2], keep=[1])

    assert np.allclose(red0, rho0, atol=1e-10)
    assert np.allclose(red1, rho1, atol=1e-10)


def _self_test_global_to_regions() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    global_rho, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=123,
    )

    assert is_density_matrix(global_rho)
    validate_region_state_collection(cfg, region_states, check_overlap_consistency=True)


def _self_test_initialization() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    init_states = initialize_all_region_states(cfg, rng=7)

    validate_region_state_collection(cfg, init_states, check_overlap_consistency=False)
    assert set(init_states.keys()) == {region.name for region in cfg.regions}


def _self_test_overlap_residuals() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=999,
    )

    residuals = all_overlap_residuals(cfg, region_states)
    assert len(residuals) == 1
    for val in residuals.values():
        assert val <= 1e-10


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the states module.
    """
    tests = [
        ("random state generation", _self_test_random_states),
        ("product state construction", _self_test_product_state),
        ("global-to-regional reductions", _self_test_global_to_regions),
        ("regional initialization", _self_test_initialization),
        ("overlap residuals", _self_test_overlap_residuals),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All states self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

# measurements.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from config import ExperimentConfig, RegionConfig
from core_ops import (
    dagger,
    frobenius_norm,
    hermitian_part,
    identity,
    is_psd,
)


# ============================================================
# Numerical defaults
# ============================================================

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8
DEFAULT_PROB_FLOOR = 1e-12


# ============================================================
# Small internal helpers
# ============================================================

def _as_numpy_array(x, dtype=None) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _check_square_matrix(a: np.ndarray, name: str = "matrix") -> None:
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {a.shape}.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square, got shape {a.shape}.")


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


def _ket_to_density(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm <= 1e-14:
        raise ValueError("Cannot build a density matrix from a numerically zero ket.")
    psi = psi / norm
    return np.outer(psi, np.conjugate(psi))


def _projector(psi: np.ndarray) -> np.ndarray:
    return _ket_to_density(psi)


def _eigh_inverse_sqrt_psd(a: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Return A^{-1/2} for a Hermitian positive definite matrix A.

    Raises
    ------
    ValueError
        If A is numerically singular or not positive definite.
    """
    a = hermitian_part(_as_numpy_array(a, dtype=np.complex128))
    _check_square_matrix(a, "a")

    evals, evecs = np.linalg.eigh(a)
    min_eval = float(np.min(evals))
    if min_eval <= atol:
        raise ValueError(
            f"Cannot compute inverse square root because the matrix is numerically singular "
            f"or not strictly positive definite. Minimum eigenvalue = {min_eval}."
        )
    inv_sqrt_evals = 1.0 / np.sqrt(evals)
    return (evecs * inv_sqrt_evals) @ dagger(evecs)


def _effects_to_tuple(effects: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    if len(effects) == 0:
        raise ValueError("effects must be a non-empty sequence.")
    out: List[np.ndarray] = []
    for idx, e in enumerate(effects):
        e = _as_numpy_array(e, dtype=np.complex128)
        _check_square_matrix(e, f"effects[{idx}]")
        out.append(e)
    dim = out[0].shape[0]
    for idx, e in enumerate(out):
        if e.shape != (dim, dim):
            raise ValueError(
                f"All POVM effects must have the same shape. "
                f"effects[0] has shape {(dim, dim)} but effects[{idx}] has shape {e.shape}."
            )
    return tuple(out)


# ============================================================
# POVM data structure
# ============================================================

@dataclass(frozen=True)
class POVM:
    """
    Positive operator-valued measure (POVM).

    Attributes
    ----------
    name :
        Human-readable POVM label.
    effects :
        Tuple of effect operators E_m satisfying
            E_m >= 0,  sum_m E_m = I.
    dim :
        Hilbert-space dimension.
    num_outcomes :
        Number of outcomes.
    metadata :
        Optional dictionary for construction metadata.
    """
    name: str
    effects: Tuple[np.ndarray, ...]
    dim: int
    num_outcomes: int
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        effects = _effects_to_tuple(self.effects)
        object.__setattr__(self, "effects", effects)

        dim = _ensure_positive_int(self.dim, "dim")
        num_outcomes = _ensure_positive_int(self.num_outcomes, "num_outcomes")

        if len(effects) != num_outcomes:
            raise ValueError(
                f"num_outcomes={num_outcomes} but {len(effects)} effects were provided."
            )
        if any(e.shape != (dim, dim) for e in effects):
            raise ValueError(
                f"All effects must have shape {(dim, dim)}."
            )

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "num_outcomes", num_outcomes)

    def validate(
        self,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        check_psd: bool = True,
    ) -> None:
        """
        Validate the POVM constraints.

        Raises
        ------
        ValueError
            If any POVM condition fails.
        """
        validate_povm(
            self.effects,
            dim=self.dim,
            atol=atol,
            rtol=rtol,
            check_psd=check_psd,
        )


# ============================================================
# POVM validation
# ============================================================

def povm_identity_residual(
    effects: Sequence[np.ndarray],
    dim: Optional[int] = None,
) -> float:
    """
    Return the Frobenius residual of sum_m E_m - I.
    """
    effects = _effects_to_tuple(effects)
    if dim is None:
        dim = effects[0].shape[0]
    dim = int(dim)
    s = np.zeros((dim, dim), dtype=np.complex128)
    for e in effects:
        s = s + e
    return frobenius_norm(s - identity(dim))


def validate_povm(
    effects: Sequence[np.ndarray],
    dim: Optional[int] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    check_psd: bool = True,
) -> None:
    r"""
    Validate the POVM constraints:
        E_m \succeq 0,    \sum_m E_m = I.

    Parameters
    ----------
    effects :
        POVM effect operators.
    dim :
        Hilbert-space dimension. If None, inferred from the effects.
    atol, rtol :
        Numerical tolerances.
    check_psd :
        If True, each effect is checked for positive semidefiniteness.

    Raises
    ------
    ValueError
        If the POVM is invalid.
    """
    effects = _effects_to_tuple(effects)
    if dim is None:
        dim = effects[0].shape[0]
    dim = _ensure_positive_int(dim, "dim")

    for idx, e in enumerate(effects):
        if e.shape != (dim, dim):
            raise ValueError(
                f"effects[{idx}] has shape {e.shape}, expected {(dim, dim)}."
            )
        if not np.allclose(e, dagger(e), atol=atol, rtol=rtol):
            raise ValueError(f"effects[{idx}] is not Hermitian within tolerance.")
        if check_psd and not is_psd(e, atol=atol):
            raise ValueError(f"effects[{idx}] is not PSD within tolerance.")

    s = np.zeros((dim, dim), dtype=np.complex128)
    for e in effects:
        s = s + e

    if not np.allclose(s, identity(dim), atol=atol, rtol=rtol):
        residual = frobenius_norm(s - identity(dim))
        raise ValueError(
            f"POVM effects do not sum to the identity within tolerance. "
            f"Residual Frobenius norm = {residual:.6e}."
        )


# ============================================================
# Standard POVM constructors
# ============================================================

def make_computational_povm(dim: int) -> POVM:
    """
    Construct the computational-basis projective measurement in dimension `dim`.

    Outcomes
    --------
    One outcome for each basis vector e_k, with effect
        E_k = |k><k|.
    """
    dim = _ensure_positive_int(dim, "dim")
    effects: List[np.ndarray] = []
    for k in range(dim):
        e = np.zeros((dim, dim), dtype=np.complex128)
        e[k, k] = 1.0
        effects.append(e)

    povm = POVM(
        name=f"computational_dim_{dim}",
        effects=tuple(effects),
        dim=dim,
        num_outcomes=dim,
        metadata={"type": "computational"},
    )
    povm.validate()
    return povm


def make_pauli6_single_qubit_povm() -> POVM:
    r"""
    Construct the 6-outcome single-qubit Pauli POVM:
        (1/3)|0><0|, (1/3)|1><1|,
        (1/3)|+><+|, (1/3)|-><-|,
        (1/3)|+i><+i|, (1/3)|-i><-i|.

    Since each Pauli basis pair sums to I, the total sum is
        (1/3)(I + I + I) = I.
    """
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    one = np.array([0.0, 1.0], dtype=np.complex128)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    minus = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    plus_i = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    minus_i = np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2.0)

    effects = tuple(
        (1.0 / 3.0) * _projector(psi)
        for psi in (zero, one, plus, minus, plus_i, minus_i)
    )

    povm = POVM(
        name="pauli6_single_qubit",
        effects=effects,
        dim=2,
        num_outcomes=6,
        metadata={"type": "pauli6_single_qubit"},
    )
    povm.validate()
    return povm


def make_random_ic_povm(
    dim: int,
    num_outcomes: Optional[int] = None,
    rng=None,
    max_tries: int = 20,
) -> POVM:
    r"""
    Construct a random informationally complete rank-1 POVM.

    Construction
    ------------
    Sample random vectors psi_m in C^d and define rank-1 operators
        F_m = |psi_m><psi_m|.
    Let
        A = sum_m F_m.
    If A is positive definite, define
        E_m = A^{-1/2} F_m A^{-1/2}.
    Then each E_m is PSD and
        sum_m E_m = I.

    Parameters
    ----------
    dim :
        Hilbert-space dimension.
    num_outcomes :
        Number of POVM outcomes. If None, use dim^2.
    rng :
        None, seed, or NumPy Generator.
    max_tries :
        Maximum number of attempts if the frame operator is numerically singular.

    Returns
    -------
    POVM
        Random POVM with `num_outcomes` outcomes.

    Notes
    -----
    For the intended use in this project, we require num_outcomes >= dim^2.
    """
    dim = _ensure_positive_int(dim, "dim")
    if num_outcomes is None:
        num_outcomes = dim * dim
    num_outcomes = _ensure_positive_int(num_outcomes, "num_outcomes")
    if num_outcomes < dim * dim:
        raise ValueError(
            f"Random informationally complete POVM requires num_outcomes >= dim^2 = {dim * dim}, "
            f"got {num_outcomes}."
        )

    rng = _coerce_rng(rng)

    for attempt in range(max_tries):
        vectors = []
        frame = np.zeros((dim, dim), dtype=np.complex128)

        for _ in range(num_outcomes):
            v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            f = _projector(v)
            vectors.append(f)
            frame = frame + f

        try:
            frame_inv_sqrt = _eigh_inverse_sqrt_psd(frame, atol=1e-12)
        except ValueError:
            continue

        effects = tuple(
            hermitian_part(frame_inv_sqrt @ f @ frame_inv_sqrt)
            for f in vectors
        )

        povm = POVM(
            name=f"random_ic_dim_{dim}_M_{num_outcomes}",
            effects=effects,
            dim=dim,
            num_outcomes=num_outcomes,
            metadata={
                "type": "random_ic",
                "attempt": attempt + 1,
            },
        )
        povm.validate(atol=1e-8, rtol=1e-7)
        return povm

    raise RuntimeError(
        f"Failed to construct a numerically stable random IC POVM after {max_tries} attempts."
    )


# ============================================================
# Measurement maps and Born-rule probabilities
# ============================================================

def born_probability_vector(
    rho: np.ndarray,
    effects: Sequence[np.ndarray],
    prob_floor: float = 0.0,
) -> np.ndarray:
    r"""
    Compute the Born probability vector
        p_m = Tr(E_m rho).

    Parameters
    ----------
    rho :
        Density matrix or Hermitian operator.
    effects :
        POVM effects.
    prob_floor :
        Optional nonnegative floor applied entrywise before renormalization.

    Returns
    -------
    np.ndarray
        Real probability vector.
    """
    rho = _as_numpy_array(rho, dtype=np.complex128)
    _check_square_matrix(rho, "rho")
    effects = _effects_to_tuple(effects)

    dim = effects[0].shape[0]
    if rho.shape != (dim, dim):
        raise ValueError(
            f"rho has shape {rho.shape}, but effects act on dimension {dim}."
        )

    probs = np.array(
        [np.real_if_close(np.trace(e @ rho)) for e in effects],
        dtype=float,
    )

    probs = np.real(probs)
    if prob_floor < 0.0:
        raise ValueError(f"prob_floor must be nonnegative, got {prob_floor}.")
    if prob_floor > 0.0:
        probs = np.maximum(probs, prob_floor)
        s = np.sum(probs)
        if s <= 0.0:
            raise ValueError("Probability vector has non-positive total mass after flooring.")
        probs = probs / s

    return probs


def measurement_map(rho: np.ndarray, povm: Union[POVM, Sequence[np.ndarray]]) -> np.ndarray:
    """
    Apply the linear POVM measurement map to an operator rho.

    This is simply the vector of Born expectations.
    """
    effects = povm.effects if isinstance(povm, POVM) else povm
    return born_probability_vector(rho, effects, prob_floor=0.0)


def measurement_map_adjoint(
    weights: np.ndarray,
    povm: Union[POVM, Sequence[np.ndarray]],
) -> np.ndarray:
    r"""
    Apply the adjoint measurement map:
        M^*(w) = sum_m w_m E_m.

    This is useful for gradients with respect to rho.
    """
    effects = povm.effects if isinstance(povm, POVM) else _effects_to_tuple(povm)
    weights = _as_numpy_array(weights, dtype=float).reshape(-1)

    if len(weights) != len(effects):
        raise ValueError(
            f"weights has length {len(weights)} but POVM has {len(effects)} outcomes."
        )

    dim = effects[0].shape[0]
    out = np.zeros((dim, dim), dtype=np.complex128)
    for w, e in zip(weights, effects):
        out = out + float(w) * e
    return hermitian_part(out)


def expected_counts(
    rho: np.ndarray,
    povm: Union[POVM, Sequence[np.ndarray]],
    shots: int,
) -> np.ndarray:
    """
    Return the expected measurement counts for a given number of shots.
    """
    shots = _ensure_positive_int(shots, "shots")
    probs = measurement_map(rho, povm)
    return shots * probs


def povm_effect_traces(povm: Union[POVM, Sequence[np.ndarray]]) -> np.ndarray:
    """
    Return the traces of the POVM effects.
    """
    effects = povm.effects if isinstance(povm, POVM) else _effects_to_tuple(povm)
    return np.array([np.real(np.trace(e)) for e in effects], dtype=float)


# ============================================================
# Config-based builders
# ============================================================

def build_region_povm(
    cfg: ExperimentConfig,
    region: Union[RegionConfig, str],
    rng=None,
) -> POVM:
    """
    Build the POVM for one region from the experiment configuration.
    """
    region_obj = cfg.region_by_name(region) if isinstance(region, str) else region
    dim = cfg.region_dimension(region_obj)
    povm_type = region_obj.povm_type
    num_outcomes = region_obj.povm_num_outcomes

    if povm_type == "computational":
        povm = make_computational_povm(dim)
        if num_outcomes is not None and num_outcomes != povm.num_outcomes:
            raise ValueError(
                f"Region '{region_obj.name}' requested povm_num_outcomes={num_outcomes}, "
                f"but the computational POVM has exactly {povm.num_outcomes} outcomes."
            )
        return povm

    if povm_type == "pauli6_single_qubit":
        if dim != 2:
            raise ValueError(
                f"Region '{region_obj.name}' uses pauli6_single_qubit but has dimension {dim}."
            )
        povm = make_pauli6_single_qubit_povm()
        if num_outcomes is not None and num_outcomes != povm.num_outcomes:
            raise ValueError(
                f"Region '{region_obj.name}' requested povm_num_outcomes={num_outcomes}, "
                f"but pauli6_single_qubit has exactly 6 outcomes."
            )
        return povm

    if povm_type == "random_ic":
        if num_outcomes is None:
            num_outcomes = dim * dim
        return make_random_ic_povm(
            dim=dim,
            num_outcomes=num_outcomes,
            rng=rng,
        )

    raise ValueError(
        f"Unsupported povm_type '{povm_type}' for region '{region_obj.name}'."
    )


def build_all_region_povms(
    cfg: ExperimentConfig,
    rng=None,
) -> Dict[str, POVM]:
    """
    Build POVMs for all regions in the experiment configuration.

    Returns
    -------
    dict[str, POVM]
        Mapping region name -> POVM object.
    """
    rng = _coerce_rng(rng)
    out: Dict[str, POVM] = {}
    for region in cfg.regions:
        out[region.name] = build_region_povm(cfg, region, rng=rng)
    return out


# ============================================================
# Validation of POVM collections
# ============================================================

def validate_region_povm_collection(
    cfg: ExperimentConfig,
    povms: Dict[str, POVM],
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """
    Validate a region-name -> POVM mapping against the experiment config.
    """
    expected_names = {region.name for region in cfg.regions}
    provided_names = set(povms.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names

    if missing:
        raise ValueError(f"Missing POVMs for regions: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected region names in POVM collection: {sorted(extra)}.")

    for region in cfg.regions:
        povm = povms[region.name]
        expected_dim = cfg.region_dimension(region)

        if povm.dim != expected_dim:
            raise ValueError(
                f"POVM for region '{region.name}' has dim={povm.dim}, expected {expected_dim}."
            )
        povm.validate(atol=atol, rtol=rtol, check_psd=True)


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_computational_povm() -> None:
    povm = make_computational_povm(4)
    povm.validate()

    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[2, 2] = 1.0

    p = measurement_map(rho, povm)
    target = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    assert np.allclose(p, target, atol=1e-10)
    assert np.isclose(np.sum(p), 1.0, atol=1e-10)


def _self_test_pauli6_povm() -> None:
    povm = make_pauli6_single_qubit_povm()
    povm.validate()

    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p = measurement_map(rho0, povm)

    assert len(p) == 6
    assert np.isclose(np.sum(p), 1.0, atol=1e-10)

    target = np.array([1/3, 0.0, 1/6, 1/6, 1/6, 1/6], dtype=float)
    assert np.allclose(p, target, atol=1e-10)


def _self_test_random_ic_povm() -> None:
    povm = make_random_ic_povm(dim=3, num_outcomes=9, rng=123)
    povm.validate()

    residual = povm_identity_residual(povm.effects, dim=3)
    assert residual <= 1e-8

    rho = np.eye(3, dtype=np.complex128) / 3.0
    p = measurement_map(rho, povm)
    assert np.isclose(np.sum(p), 1.0, atol=1e-8)
    assert np.all(p >= -1e-10)


def _self_test_measurement_adjoint() -> None:
    povm = make_random_ic_povm(dim=2, num_outcomes=4, rng=7)
    rho = np.array([[0.7, 0.1 - 0.05j], [0.1 + 0.05j, 0.3]], dtype=np.complex128)
    rho = hermitian_part(rho)
    rho = rho / np.trace(rho)

    w = np.array([0.2, -0.4, 0.7, 0.1], dtype=float)
    lhs = float(np.dot(w, measurement_map(rho, povm)))
    rhs = float(np.real(np.trace(measurement_map_adjoint(w, povm) @ rho)))

    assert np.isclose(lhs, rhs, atol=1e-10)


def _self_test_config_builders() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    povms = build_all_region_povms(cfg, rng=2024)
    validate_region_povm_collection(cfg, povms)

    assert set(povms.keys()) == {region.name for region in cfg.regions}
    for region in cfg.regions:
        assert povms[region.name].dim == cfg.region_dimension(region)


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the measurements module.
    """
    tests = [
        ("computational POVM", _self_test_computational_povm),
        ("pauli-6 single-qubit POVM", _self_test_pauli6_povm),
        ("random IC POVM", _self_test_random_ic_povm),
        ("measurement adjoint identity", _self_test_measurement_adjoint),
        ("config-based POVM builders", _self_test_config_builders),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All measurements self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

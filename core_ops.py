# core_ops.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


# ============================================================
# Numerical defaults
# ============================================================

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8
DEFAULT_PROB_FLOOR = 1e-12


# ============================================================
# Basic validation helpers
# ============================================================

def _as_numpy_array(x, dtype=None) -> np.ndarray:
    """Convert input to a NumPy array."""
    return np.asarray(x, dtype=dtype)


def _check_square_matrix(a: np.ndarray, name: str = "matrix") -> None:
    """Raise ValueError if `a` is not a square 2D array."""
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {a.shape}.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square, got shape {a.shape}.")


def _check_same_shape(a: np.ndarray, b: np.ndarray, name_a: str = "a", name_b: str = "b") -> None:
    """Raise ValueError if two arrays do not have the same shape."""
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got {a.shape} and {b.shape}."
        )


def _prod_int(values: Sequence[int]) -> int:
    """Integer product of a sequence of positive integers."""
    out = 1
    for v in values:
        out *= int(v)
    return out


def _validate_dims(dims: Sequence[int], name: str = "dims") -> List[int]:
    """Validate subsystem dimensions and return them as a list of ints."""
    if len(dims) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of positive integers.")
    dims_list = [int(d) for d in dims]
    if any(d <= 0 for d in dims_list):
        raise ValueError(f"All entries of {name} must be positive. Got {dims_list}.")
    return dims_list


def _validate_indices(indices: Sequence[int], n: int, name: str = "indices") -> List[int]:
    """Validate that indices are unique integers in range [0, n)."""
    idx = [int(i) for i in indices]
    if len(set(idx)) != len(idx):
        raise ValueError(f"{name} must not contain duplicates. Got {idx}.")
    if any(i < 0 or i >= n for i in idx):
        raise ValueError(f"{name} must lie in [0, {n - 1}]. Got {idx}.")
    return idx


# ============================================================
# Complex / Hermitian helpers
# ============================================================

def dagger(a: np.ndarray) -> np.ndarray:
    """
    Conjugate transpose of a matrix.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Conjugate transpose of `a`.
    """
    a = _as_numpy_array(a)
    return np.conjugate(a.T)


def hermitian_part(a: np.ndarray) -> np.ndarray:
    r"""
    Return the Hermitian part of a square matrix:
        (A + A^\dagger) / 2.
    """
    a = _as_numpy_array(a, dtype=np.complex128)
    _check_square_matrix(a, "a")
    return 0.5 * (a + dagger(a))


def antihermitian_part(a: np.ndarray) -> np.ndarray:
    r"""
    Return the anti-Hermitian part of a square matrix:
        (A - A^\dagger) / 2.
    """
    a = _as_numpy_array(a, dtype=np.complex128)
    _check_square_matrix(a, "a")
    return 0.5 * (a - dagger(a))


def real_if_close_scalar(x, atol: float = DEFAULT_ATOL) -> float | complex:
    """
    Return the real part if the imaginary part is numerically negligible.
    """
    x = np.asarray(x).item()
    if abs(np.imag(x)) <= atol:
        return float(np.real(x))
    return x


def real_if_close_array(x: np.ndarray, atol: float = DEFAULT_ATOL) -> np.ndarray:
    """
    Return a real array if all imaginary parts are numerically negligible.
    Otherwise return the original complex-valued array.
    """
    x = _as_numpy_array(x)
    if np.max(np.abs(np.imag(x))) <= atol:
        return np.real(x)
    return x


def hs_inner(a: np.ndarray, b: np.ndarray) -> float:
    r"""
    Hilbert-Schmidt inner product Re[Tr(A^\dagger B)].

    This is the natural real inner product for complex matrices when using
    Frobenius geometry.

    Returns
    -------
    float
    """
    a = _as_numpy_array(a, dtype=np.complex128)
    b = _as_numpy_array(b, dtype=np.complex128)
    _check_same_shape(a, b, "a", "b")
    return float(np.real(np.trace(dagger(a) @ b)))


def frobenius_norm(a: np.ndarray) -> float:
    """
    Frobenius norm of an array.
    """
    a = _as_numpy_array(a)
    return float(np.linalg.norm(a, ord="fro"))


# ============================================================
# Density matrix checks and projections
# ============================================================

def is_hermitian(a: np.ndarray, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    """
    Check whether a matrix is Hermitian up to numerical tolerances.
    """
    a = _as_numpy_array(a, dtype=np.complex128)
    _check_square_matrix(a, "a")
    return np.allclose(a, dagger(a), atol=atol, rtol=rtol)


def is_psd(a: np.ndarray, atol: float = DEFAULT_ATOL) -> bool:
    """
    Check whether a Hermitian matrix is positive semidefinite up to tolerance.

    A small negative eigenvalue above numerical noise is allowed.
    """
    a = hermitian_part(a)
    evals = np.linalg.eigvalsh(a)
    return bool(np.min(evals) >= -atol)


def is_density_matrix(
    rho: np.ndarray,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """
    Check whether a matrix is a valid density matrix:
    Hermitian, PSD, and unit trace.
    """
    rho = _as_numpy_array(rho, dtype=np.complex128)
    _check_square_matrix(rho, "rho")
    if not is_hermitian(rho, atol=atol, rtol=rtol):
        return False
    if not is_psd(rho, atol=atol):
        return False
    tr = np.trace(rho)
    return np.isclose(np.real(tr), 1.0, atol=atol, rtol=rtol) and abs(np.imag(tr)) <= atol


def normalize_trace(a: np.ndarray, target_trace: float = 1.0) -> np.ndarray:
    """
    Rescale a matrix so its trace equals `target_trace`.

    This does NOT enforce Hermiticity or PSD. It is only a trace rescaling.
    """
    a = _as_numpy_array(a, dtype=np.complex128)
    _check_square_matrix(a, "a")
    tr = np.trace(a)
    tr_real = np.real_if_close(tr)
    if abs(tr_real) < DEFAULT_ATOL:
        raise ValueError("Cannot normalize trace because the matrix trace is numerically zero.")
    return a * (target_trace / tr_real)


def project_vector_to_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Euclidean projection of a real vector onto the probability simplex:
        {x >= 0, sum(x) = z}

    Uses the standard sorting-based algorithm.

    Parameters
    ----------
    v : np.ndarray
        Real input vector.
    z : float
        Desired simplex sum. Must be positive.

    Returns
    -------
    np.ndarray
        Projected vector.
    """
    v = _as_numpy_array(v, dtype=float).reshape(-1)
    if z <= 0:
        raise ValueError(f"Simplex radius z must be positive, got {z}.")
    n = v.size
    if n == 0:
        raise ValueError("Input vector must be non-empty.")

    if np.all(v >= 0.0) and np.isclose(np.sum(v), z, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
        return v.copy()

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0

    if not np.any(cond):
        return np.full(n, z / n, dtype=float)

    rho = ind[cond][-1]
    theta = cssv[rho - 1] / rho
    w = np.maximum(v - theta, 0.0)

    s = np.sum(w)
    if s <= 0:
        return np.full(n, z / n, dtype=float)
    return w * (z / s)


def project_columns_to_simplex(a: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Project each column of a real matrix onto the simplex
        {x >= 0, sum(x) = z}.
    """
    a = _as_numpy_array(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got shape {a.shape}.")
    out = np.empty_like(a, dtype=float)
    for j in range(a.shape[1]):
        out[:, j] = project_vector_to_simplex(a[:, j], z=z)
    return out


def project_rows_to_simplex(a: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Project each row of a real matrix onto the simplex
        {x >= 0, sum(x) = z}.
    """
    a = _as_numpy_array(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got shape {a.shape}.")
    out = np.empty_like(a, dtype=float)
    for i in range(a.shape[0]):
        out[i, :] = project_vector_to_simplex(a[i, :], z=z)
    return out


def project_to_density_matrix(a: np.ndarray, target_trace: float = 1.0) -> np.ndarray:
    """
    Project a square complex matrix onto the set of density matrices
    (Hermitian PSD matrices with specified trace) in Frobenius geometry.

    Procedure:
    1) Hermitian symmetrization
    2) Eigenvalue decomposition
    3) Project eigenvalues onto simplex with sum = target_trace
    4) Reconstruct matrix

    Parameters
    ----------
    a : np.ndarray
        Square complex matrix.
    target_trace : float
        Desired trace, typically 1.0.

    Returns
    -------
    np.ndarray
        Projected density matrix.
    """
    a = _as_numpy_array(a, dtype=np.complex128)
    _check_square_matrix(a, "a")
    if target_trace <= 0:
        raise ValueError(f"target_trace must be positive, got {target_trace}.")

    h = hermitian_part(a)
    evals, evecs = np.linalg.eigh(h)
    evals_proj = project_vector_to_simplex(evals, z=target_trace)
    rho = (evecs * evals_proj) @ dagger(evecs)
    rho = hermitian_part(rho)
    rho = normalize_trace(rho, target_trace=target_trace)

    evals2, evecs2 = np.linalg.eigh(rho)
    evals2 = np.maximum(evals2, 0.0)
    evals2 = evals2 / np.sum(evals2) * target_trace
    rho = (evecs2 * evals2) @ dagger(evecs2)
    rho = hermitian_part(rho)
    return rho


def closest_psd(a: np.ndarray) -> np.ndarray:
    """
    Project a square matrix onto the PSD cone by Hermitian symmetrization
    followed by clipping negative eigenvalues to zero.

    This does NOT enforce unit trace.
    """
    a = _as_numpy_array(a, dtype=np.complex128)
    _check_square_matrix(a, "a")
    h = hermitian_part(a)
    evals, evecs = np.linalg.eigh(h)
    evals = np.maximum(evals, 0.0)
    return hermitian_part((evecs * evals) @ dagger(evecs))


# ============================================================
# Probability helpers
# ============================================================

def clip_probabilities(
    p: np.ndarray,
    floor: float = DEFAULT_PROB_FLOOR,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Clip a probability vector away from zero and optionally renormalize.

    Useful before log-likelihood evaluations.
    """
    p = _as_numpy_array(p, dtype=float).reshape(-1)
    if floor <= 0:
        raise ValueError(f"floor must be positive, got {floor}.")
    q = np.maximum(p, floor)
    if renormalize:
        s = np.sum(q)
        if s <= 0:
            raise ValueError("Cannot renormalize probabilities with non-positive total mass.")
        q = q / s
    return q


def normalize_probability_vector(p: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """
    Project a real vector onto the simplex after optional lower clipping.
    """
    p = _as_numpy_array(p, dtype=float).reshape(-1)
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}.")
    q = np.maximum(p, floor)
    return project_vector_to_simplex(q, z=1.0)


# ============================================================
# Stochastic-matrix helpers
# ============================================================

def is_column_stochastic(
    c: np.ndarray,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> bool:
    """
    Check whether a matrix is column-stochastic:
    nonnegative and each column sums to 1.
    """
    c = _as_numpy_array(c, dtype=float)
    if c.ndim != 2:
        return False
    if np.any(c < -atol):
        return False
    col_sums = np.sum(c, axis=0)
    return np.allclose(col_sums, 1.0, atol=atol, rtol=rtol)


def project_to_column_stochastic(c: np.ndarray) -> np.ndarray:
    """
    Project a real matrix onto the set of column-stochastic matrices
    by projecting each column onto the simplex.
    """
    c = _as_numpy_array(c, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"c must be 2D, got shape {c.shape}.")
    return project_columns_to_simplex(c, z=1.0)


# ============================================================
# Tensor / subsystem operations
# ============================================================

def kron_all(operators: Sequence[np.ndarray]) -> np.ndarray:
    """
    Kronecker product of a sequence of arrays.

    Parameters
    ----------
    operators : Sequence[np.ndarray]
        Non-empty sequence of matrices or vectors.

    Returns
    -------
    np.ndarray
        Kronecker product in left-to-right order.
    """
    if len(operators) == 0:
        raise ValueError("operators must be a non-empty sequence.")
    out = _as_numpy_array(operators[0], dtype=np.complex128)
    for op in operators[1:]:
        out = np.kron(out, _as_numpy_array(op, dtype=np.complex128))
    return out


def permute_subsystems(rho: np.ndarray, dims: Sequence[int], perm: Sequence[int]) -> np.ndarray:
    """
    Permute subsystem order of a density matrix.

    If rho acts on H_1 ⊗ ... ⊗ H_n with subsystem dimensions `dims`,
    then `perm` gives the new subsystem order.

    Example
    -------
    dims = [2, 3, 2], perm = [2, 0, 1]
    means:
        H_1 ⊗ H_2 ⊗ H_3  ->  H_3 ⊗ H_1 ⊗ H_2

    Parameters
    ----------
    rho : np.ndarray
        Square matrix of shape (prod(dims), prod(dims)).
    dims : Sequence[int]
        Subsystem dimensions.
    perm : Sequence[int]
        A permutation of range(len(dims)).

    Returns
    -------
    np.ndarray
        Permuted density matrix.
    """
    rho = _as_numpy_array(rho, dtype=np.complex128)
    _check_square_matrix(rho, "rho")
    dims = _validate_dims(dims, "dims")
    n = len(dims)
    perm = _validate_indices(perm, n, "perm")
    if sorted(perm) != list(range(n)):
        raise ValueError(f"perm must be a permutation of 0..{n - 1}. Got {perm}.")

    d_total = _prod_int(dims)
    if rho.shape != (d_total, d_total):
        raise ValueError(
            f"rho shape {rho.shape} is incompatible with subsystem dimensions {dims} "
            f"(total dimension {d_total})."
        )

    if n == 1:
        return rho.copy()

    tensor = rho.reshape(dims + dims)
    axes = list(perm) + [p + n for p in perm]
    permuted = np.transpose(tensor, axes=axes)
    new_dims = [dims[p] for p in perm]
    d_new = _prod_int(new_dims)
    return permuted.reshape(d_new, d_new)


def partial_trace(
    rho: np.ndarray,
    dims: Sequence[int],
    keep: Optional[Sequence[int]] = None,
    trace_out: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Partial trace over selected subsystems.

    Exactly one of `keep` or `trace_out` must be provided.

    Parameters
    ----------
    rho : np.ndarray
        Square density matrix or operator on the tensor-product space with
        subsystem dimensions `dims`.
    dims : Sequence[int]
        Subsystem dimensions.
    keep : Optional[Sequence[int]]
        Indices of subsystems to keep, in the desired output order.
    trace_out : Optional[Sequence[int]]
        Indices of subsystems to trace out.

    Returns
    -------
    np.ndarray
        Reduced operator on the kept subsystems.

    Notes
    -----
    - If `keep=[]`, the result is a 1x1 matrix containing Tr(rho).
    - The output subsystem order follows the order specified in `keep`.
    """
    rho = _as_numpy_array(rho, dtype=np.complex128)
    _check_square_matrix(rho, "rho")
    dims = _validate_dims(dims, "dims")
    n = len(dims)

    if (keep is None) == (trace_out is None):
        raise ValueError("Exactly one of `keep` or `trace_out` must be provided.")

    d_total = _prod_int(dims)
    if rho.shape != (d_total, d_total):
        raise ValueError(
            f"rho shape {rho.shape} is incompatible with dims {dims} "
            f"(total dimension {d_total})."
        )

    if keep is not None:
        keep = _validate_indices(keep, n, "keep")
        trace_out = [i for i in range(n) if i not in keep]
        keep_desired_order = list(keep)
    else:
        trace_out = _validate_indices(trace_out, n, "trace_out")
        keep_desired_order = [i for i in range(n) if i not in trace_out]

    if len(keep_desired_order) == 0:
        return np.array([[np.trace(rho)]], dtype=np.complex128)

    keep_canonical_order = [i for i in range(n) if i not in trace_out]
    current_dims = dims.copy()
    tensor = rho.reshape(dims + dims)

    for ax in sorted(trace_out, reverse=True):
        n_cur = len(current_dims)
        tensor = np.trace(tensor, axis1=ax, axis2=ax + n_cur)
        current_dims.pop(ax)

    reduced = tensor.reshape(_prod_int(current_dims), _prod_int(current_dims))

    if keep_desired_order != keep_canonical_order and len(current_dims) > 1:
        perm = [keep_canonical_order.index(i) for i in keep_desired_order]
        reduced = permute_subsystems(reduced, current_dims, perm)

    return reduced


def subsystem_dimensions_from_qubits(qubits_per_site: Sequence[int]) -> List[int]:
    """
    Convert a list of qubit counts per site to subsystem Hilbert-space dimensions.

    Example
    -------
    [1, 2, 1] -> [2, 4, 2]
    """
    q = [int(v) for v in qubits_per_site]
    if any(v < 0 for v in q):
        raise ValueError(f"qubits_per_site must be non-negative, got {q}.")
    return [2 ** v for v in q]


# ============================================================
# Small utility constructors
# ============================================================

def identity(dim: int) -> np.ndarray:
    """Complex identity matrix of size `dim`."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    return np.eye(dim, dtype=np.complex128)


def maximally_mixed(dim: int) -> np.ndarray:
    """Maximally mixed state I / dim."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    return np.eye(dim, dtype=np.complex128) / dim


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_simplex() -> None:
    v = np.array([0.2, -0.3, 2.0, 0.1], dtype=float)
    w = project_vector_to_simplex(v, z=1.0)
    assert np.all(w >= -1e-12)
    assert np.isclose(np.sum(w), 1.0, atol=1e-10)


def _self_test_density_projection() -> None:
    a = np.array([[0.7, 2.0 + 1.0j], [2.0 - 1.0j, -0.2]], dtype=np.complex128)
    rho = project_to_density_matrix(a)
    assert is_density_matrix(rho)
    assert np.isclose(np.trace(rho), 1.0, atol=1e-10)


def _self_test_partial_trace_bell() -> None:
    ket = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0)
    rho = np.outer(ket, np.conjugate(ket))
    red0 = partial_trace(rho, dims=[2, 2], keep=[0])
    red1 = partial_trace(rho, dims=[2, 2], keep=[1])
    target = np.eye(2, dtype=np.complex128) / 2.0
    assert np.allclose(red0, target, atol=1e-10)
    assert np.allclose(red1, target, atol=1e-10)


def _self_test_permutation() -> None:
    dims = [2, 3]
    rho = np.arange(36).reshape(6, 6).astype(np.complex128)
    rho_perm = permute_subsystems(rho, dims=dims, perm=[1, 0])
    rho_back = permute_subsystems(rho_perm, dims=[3, 2], perm=[1, 0])
    assert np.allclose(rho, rho_back)


def _self_test_column_stochastic() -> None:
    c = np.array([[1.2, -0.1], [-0.2, 1.1]], dtype=float)
    cp = project_to_column_stochastic(c)
    assert is_column_stochastic(cp)


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a small suite of smoke tests for this module.
    """
    tests = [
        ("simplex projection", _self_test_simplex),
        ("density projection", _self_test_density_projection),
        ("partial trace (Bell state)", _self_test_partial_trace_bell),
        ("subsystem permutation", _self_test_permutation),
        ("column stochastic projection", _self_test_column_stochastic),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All core_ops self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

# objectives.py
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from config import ExperimentConfig, LossConfig, RegionConfig
from core_ops import DEFAULT_PROB_FLOOR, frobenius_norm, partial_trace
from measurements import POVM, measurement_map, measurement_map_adjoint
from noise import apply_confusion_matrix, confusion_frobenius_regularizer
from regions import RegionGraph


# ============================================================
# Small helpers
# ============================================================

def _as_numpy_array(x, dtype=None) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _ensure_positive_float(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _ensure_nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _resolve_loss_name(loss: Union[str, LossConfig]) -> str:
    if isinstance(loss, LossConfig):
        return loss.name
    return str(loss)


def _resolve_prob_floor(loss: Union[str, LossConfig], prob_floor: Optional[float]) -> float:
    if prob_floor is not None:
        return _ensure_positive_float(prob_floor, "prob_floor")
    if isinstance(loss, LossConfig):
        return _ensure_positive_float(loss.prob_floor, "loss.prob_floor")
    return float(DEFAULT_PROB_FLOOR)


def _validate_probability_vector(p: np.ndarray, name: str) -> np.ndarray:
    p = _as_numpy_array(p, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(~np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values.")
    s = float(np.sum(p))
    if s <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return p


def _canonical_overlap_name_key(
    graph: RegionGraph,
    region_i: Union[str, RegionConfig, int],
    region_j: Union[str, RegionConfig, int],
) -> Tuple[str, str]:
    return graph.canonical_overlap_key(region_i, region_j)


def _relative_change(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = max(frobenius_norm(b), eps)
    return frobenius_norm(a - b) / denom


# ============================================================
# Discrepancy functions
# ============================================================

def l2_discrepancy(
    empirical: np.ndarray,
    predicted: np.ndarray,
) -> float:
    r"""
    Squared Euclidean discrepancy
        D(\hat p, p) = 0.5 ||\hat p - p||_2^2.
    """
    empirical = _validate_probability_vector(empirical, "empirical")
    predicted = _validate_probability_vector(predicted, "predicted")

    if empirical.shape != predicted.shape:
        raise ValueError(
            f"empirical and predicted must have the same shape, got {empirical.shape} and {predicted.shape}."
        )

    diff = predicted - empirical
    return 0.5 * float(np.dot(diff, diff))


def nll_discrepancy(
    empirical: np.ndarray,
    predicted: np.ndarray,
    shots: Optional[int] = None,
    prob_floor: float = DEFAULT_PROB_FLOOR,
) -> float:
    r"""
    Multinomial negative log-likelihood (up to an additive constant):
        D(\hat p, p) = -s \sum_m \hat p_m \log p_m,

    where s = shots if provided, otherwise s = 1.

    Notes
    -----
    - If `shots` is None, this reduces to cross-entropy with empirical frequencies.
    - `predicted` is clipped below by `prob_floor` for numerical stability.
    """
    empirical = _validate_probability_vector(empirical, "empirical")
    predicted = _validate_probability_vector(predicted, "predicted")
    prob_floor = _ensure_positive_float(prob_floor, "prob_floor")

    if empirical.shape != predicted.shape:
        raise ValueError(
            f"empirical and predicted must have the same shape, got {empirical.shape} and {predicted.shape}."
        )

    scale = 1.0 if shots is None else float(int(shots))
    p = np.maximum(predicted, prob_floor)
    return float(-scale * np.dot(empirical, np.log(p)))


def discrepancy_value(
    empirical: np.ndarray,
    predicted: np.ndarray,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Unified discrepancy-value dispatcher.
    """
    loss_name = _resolve_loss_name(loss)

    if loss_name == "l2":
        return l2_discrepancy(empirical, predicted)

    if loss_name == "nll":
        return nll_discrepancy(
            empirical=empirical,
            predicted=predicted,
            shots=shots,
            prob_floor=_resolve_prob_floor(loss, prob_floor),
        )

    raise ValueError(f"Unsupported loss '{loss_name}'.")


def gradient_wrt_predicted(
    empirical: np.ndarray,
    predicted: np.ndarray,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> np.ndarray:
    r"""
    Gradient of the discrepancy with respect to the predicted probability vector.

    For L2:
        \nabla_p D = p - \hat p.

    For NLL:
        \nabla_p D = - s * \hat p / clip(p),
    where s = shots if provided, otherwise s = 1.
    """
    empirical = _validate_probability_vector(empirical, "empirical")
    predicted = _validate_probability_vector(predicted, "predicted")

    if empirical.shape != predicted.shape:
        raise ValueError(
            f"empirical and predicted must have the same shape, got {empirical.shape} and {predicted.shape}."
        )

    loss_name = _resolve_loss_name(loss)

    if loss_name == "l2":
        return predicted - empirical

    if loss_name == "nll":
        scale = 1.0 if shots is None else float(int(shots))
        floor = _resolve_prob_floor(loss, prob_floor)
        p = np.maximum(predicted, floor)
        return -scale * empirical / p

    raise ValueError(f"Unsupported loss '{loss_name}'.")


# ============================================================
# Region-level prediction and gradients
# ============================================================

def region_prediction(
    rho: np.ndarray,
    povm: POVM,
    confusion: np.ndarray,
    prob_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute ideal and noisy region-level probability vectors:
        p_ideal = M(rho),
        p_noisy = C p_ideal.
    """
    ideal = measurement_map(rho, povm)
    noisy = apply_confusion_matrix(confusion, ideal, prob_floor=prob_floor)
    return ideal, noisy


def region_gradient_components(
    empirical: np.ndarray,
    rho: np.ndarray,
    povm: POVM,
    confusion: np.ndarray,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    r"""
    Compute region-level gradients for the data-fit term.

    Let
        p = M(rho),     q = C p,
        g_q = \nabla_q D(\hat p, q).

    Then
        g_p   = C^T g_q,
        grad_rho = M^*(g_p),
        grad_C   = g_q p^T.

    Returns
    -------
    dict
        Contains:
        - ideal_probabilities
        - noisy_probabilities
        - grad_predicted
        - grad_ideal
        - grad_rho
        - grad_C
    """
    ideal, noisy = region_prediction(
        rho=rho,
        povm=povm,
        confusion=confusion,
        prob_floor=0.0,
    )

    grad_pred = gradient_wrt_predicted(
        empirical=empirical,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )

    grad_ideal = confusion.T @ grad_pred
    grad_rho = measurement_map_adjoint(grad_ideal, povm)
    grad_c = np.outer(grad_pred, ideal)

    return {
        "ideal_probabilities": ideal,
        "noisy_probabilities": noisy,
        "grad_predicted": grad_pred,
        "grad_ideal": grad_ideal,
        "grad_rho": grad_rho,
        "grad_C": grad_c,
    }


def region_fit_objective(
    empirical: np.ndarray,
    rho: np.ndarray,
    povm: POVM,
    confusion: np.ndarray,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Region-level data-fit objective.
    """
    _, noisy = region_prediction(
        rho=rho,
        povm=povm,
        confusion=confusion,
        prob_floor=0.0,
    )
    return discrepancy_value(
        empirical=empirical,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )


def state_subproblem_region_objective(
    empirical: np.ndarray,
    rho: np.ndarray,
    povm: POVM,
    confusion: np.ndarray,
    rho_prev: np.ndarray,
    gamma_rho: float,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
) -> float:
    r"""
    Region-local contribution to the proximal state subproblem:
        D(\hat p, C M(rho)) + (gamma_rho / 2) ||rho - rho_prev||_F^2.
    """
    gamma_rho = _ensure_positive_float(gamma_rho, "gamma_rho")
    fit = region_fit_objective(
        empirical=empirical,
        rho=rho,
        povm=povm,
        confusion=confusion,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    prox = 0.5 * gamma_rho * frobenius_norm(rho - rho_prev) ** 2
    return fit + prox


def confusion_subproblem_region_objective(
    empirical: np.ndarray,
    ideal_probabilities: np.ndarray,
    confusion: np.ndarray,
    reference_confusion: np.ndarray,
    lambda_confusion: float,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
    gamma_c: float = 0.0,
    confusion_prev: Optional[np.ndarray] = None,
) -> float:
    r"""
    Region-local confusion-update objective:
        D(\hat p, C p_ideal)
        + lambda ||C - C_ref||_F^2
        + (gamma_c / 2) ||C - C_prev||_F^2.
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_nonnegative_float(gamma_c, "gamma_c")

    noisy = apply_confusion_matrix(confusion, ideal_probabilities, prob_floor=0.0)

    val = discrepancy_value(
        empirical=empirical,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )
    val += lambda_confusion * confusion_frobenius_regularizer(confusion, reference_confusion)

    if gamma_c > 0.0:
        if confusion_prev is None:
            raise ValueError("confusion_prev must be provided when gamma_c > 0.")
        val += 0.5 * gamma_c * frobenius_norm(confusion - confusion_prev) ** 2

    return val


# ============================================================
# Aggregation across regions
# ============================================================

def build_region_shot_dict(cfg: ExperimentConfig) -> Dict[str, int]:
    """
    Build a region-name -> shots dictionary from the config.
    """
    return {region.name: int(region.shots) for region in cfg.regions}


def total_data_fit_objective(
    empirical_probabilities: Mapping[str, np.ndarray],
    region_states: Mapping[str, np.ndarray],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, np.ndarray],
    loss: Union[str, LossConfig],
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> float:
    """
    Sum of region-level data-fit objectives over all regions.
    """
    names = set(empirical_probabilities.keys())
    if set(region_states.keys()) != names:
        raise ValueError("region_states keys must match empirical_probabilities keys.")
    if set(region_povms.keys()) != names:
        raise ValueError("region_povms keys must match empirical_probabilities keys.")
    if set(region_confusions.keys()) != names:
        raise ValueError("region_confusions keys must match empirical_probabilities keys.")
    if region_shots is not None and set(region_shots.keys()) != names:
        raise ValueError("region_shots keys must match empirical_probabilities keys.")

    total = 0.0
    for name in names:
        shots = None if region_shots is None else int(region_shots[name])
        total += region_fit_objective(
            empirical=empirical_probabilities[name],
            rho=region_states[name],
            povm=region_povms[name],
            confusion=region_confusions[name],
            loss=loss,
            shots=shots,
            prob_floor=prob_floor,
        )
    return float(total)


def total_regularized_objective(
    empirical_probabilities: Mapping[str, np.ndarray],
    region_states: Mapping[str, np.ndarray],
    region_povms: Mapping[str, POVM],
    region_confusions: Mapping[str, np.ndarray],
    reference_confusions: Mapping[str, np.ndarray],
    lambda_confusion: float,
    loss: Union[str, LossConfig],
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
) -> float:
    r"""
    Total regularized objective:
        sum_r D(\hat p_r, C_r M_r(rho_r))
        + lambda sum_r ||C_r - C_ref,r||_F^2.
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")

    names = set(empirical_probabilities.keys())
    if set(reference_confusions.keys()) != names:
        raise ValueError("reference_confusions keys must match empirical_probabilities keys.")

    total = total_data_fit_objective(
        empirical_probabilities=empirical_probabilities,
        region_states=region_states,
        region_povms=region_povms,
        region_confusions=region_confusions,
        loss=loss,
        region_shots=region_shots,
        prob_floor=prob_floor,
    )

    for name in names:
        total += lambda_confusion * confusion_frobenius_regularizer(
            region_confusions[name],
            reference_confusions[name],
        )

    return float(total)


# ============================================================
# ADMM overlap residual helpers
# ============================================================

def overlap_primal_residual_norm(
    graph: RegionGraph,
    region_states: Mapping[str, np.ndarray],
    eta: Mapping[Tuple[str, str], np.ndarray],
) -> float:
    r"""
    Compute the aggregate primal residual norm for overlap consensus:
        sqrt( sum_{(r,r')} ||Tr_r(rho_r) - eta_rr'||_F^2
            + sum_{(r,r')} ||Tr_r'(rho_r') - eta_rr'||_F^2 ).
    """
    total_sq = 0.0

    for i, j in graph.overlap_pairs:
        name_i = graph.region_name(i)
        name_j = graph.region_name(j)
        key = (name_i, name_j)

        if key not in eta:
            raise ValueError(f"Missing eta entry for overlap key {key}.")

        info = graph.overlap_info(i, j)

        rho_i = region_states[name_i]
        rho_j = region_states[name_j]
        eta_ij = eta[key]

        red_i = partial_trace(rho_i, dims=graph.region_site_dims(i), keep=info.local_keep_i)
        red_j = partial_trace(rho_j, dims=graph.region_site_dims(j), keep=info.local_keep_j)

        total_sq += frobenius_norm(red_i - eta_ij) ** 2
        total_sq += frobenius_norm(red_j - eta_ij) ** 2

    return float(np.sqrt(total_sq))


def overlap_dual_residual_norm(
    beta: float,
    eta_current: Mapping[Tuple[str, str], np.ndarray],
    eta_previous: Mapping[Tuple[str, str], np.ndarray],
) -> float:
    r"""
    Compute the aggregate dual residual norm:
        beta * sqrt( sum_{keys} ||eta_current - eta_previous||_F^2 ).
    """
    beta = _ensure_positive_float(beta, "beta")

    if set(eta_current.keys()) != set(eta_previous.keys()):
        raise ValueError("eta_current and eta_previous must have identical keys.")

    total_sq = 0.0
    for key in eta_current.keys():
        total_sq += frobenius_norm(eta_current[key] - eta_previous[key]) ** 2

    return float(beta * np.sqrt(total_sq))


def max_overlap_residual(
    graph: RegionGraph,
    region_states: Mapping[str, np.ndarray],
    eta: Mapping[Tuple[str, str], np.ndarray],
) -> float:
    """
    Maximum Frobenius overlap-consensus residual across all individual constraints.
    """
    max_val = 0.0

    for i, j in graph.overlap_pairs:
        name_i = graph.region_name(i)
        name_j = graph.region_name(j)
        key = (name_i, name_j)
        info = graph.overlap_info(i, j)

        rho_i = region_states[name_i]
        rho_j = region_states[name_j]
        eta_ij = eta[key]

        red_i = partial_trace(rho_i, dims=graph.region_site_dims(i), keep=info.local_keep_i)
        red_j = partial_trace(rho_j, dims=graph.region_site_dims(j), keep=info.local_keep_j)

        max_val = max(max_val, frobenius_norm(red_i - eta_ij))
        max_val = max(max_val, frobenius_norm(red_j - eta_ij))

    return float(max_val)


# ============================================================
# Relative-change helpers
# ============================================================

def relative_change_dict(
    current: Mapping[str, np.ndarray],
    previous: Mapping[str, np.ndarray],
) -> float:
    """
    Aggregate relative change for a dictionary of matrices:
        ||x_k - x_{k-1}|| / max(||x_{k-1}||, eps)
    using a global Frobenius aggregation.
    """
    if set(current.keys()) != set(previous.keys()):
        raise ValueError("current and previous must have identical keys.")

    num_sq = 0.0
    den_sq = 0.0
    for key in current.keys():
        num_sq += frobenius_norm(current[key] - previous[key]) ** 2
        den_sq += frobenius_norm(previous[key]) ** 2

    denom = max(np.sqrt(den_sq), 1e-12)
    return float(np.sqrt(num_sq) / denom)


# ============================================================
# Lightweight self-tests
# ============================================================

def _finite_difference_check_grad_predicted_l2() -> None:
    empirical = np.array([0.2, 0.5, 0.3], dtype=float)
    predicted = np.array([0.3, 0.4, 0.3], dtype=float)

    grad = gradient_wrt_predicted(empirical, predicted, loss="l2")
    eps = 1e-7

    direction = np.array([0.4, -0.2, 0.1], dtype=float)
    direction = direction / np.linalg.norm(direction)

    f_plus = discrepancy_value(empirical, predicted + eps * direction, loss="l2")
    f_minus = discrepancy_value(empirical, predicted - eps * direction, loss="l2")
    fd = (f_plus - f_minus) / (2.0 * eps)
    ip = float(np.dot(grad, direction))

    assert np.isclose(fd, ip, atol=1e-6)


def _finite_difference_check_grad_predicted_nll() -> None:
    empirical = np.array([0.2, 0.5, 0.3], dtype=float)
    predicted = np.array([0.31, 0.39, 0.30], dtype=float)

    grad = gradient_wrt_predicted(empirical, predicted, loss="nll", prob_floor=1e-12)
    eps = 1e-7

    direction = np.array([0.2, -0.1, 0.05], dtype=float)
    direction = direction / np.linalg.norm(direction)

    f_plus = discrepancy_value(empirical, predicted + eps * direction, loss="nll", prob_floor=1e-12)
    f_minus = discrepancy_value(empirical, predicted - eps * direction, loss="nll", prob_floor=1e-12)
    fd = (f_plus - f_minus) / (2.0 * eps)
    ip = float(np.dot(grad, direction))

    assert np.isclose(fd, ip, atol=1e-5)


def _self_test_region_gradient_shapes() -> None:
    from measurements import make_computational_povm
    from noise import make_noisy_identity_confusion

    rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=np.complex128)
    povm = make_computational_povm(2)
    c = make_noisy_identity_confusion(2, strength=0.1)
    empirical = np.array([0.6, 0.4], dtype=float)

    out = region_gradient_components(
        empirical=empirical,
        rho=rho,
        povm=povm,
        confusion=c,
        loss="l2",
    )

    assert out["ideal_probabilities"].shape == (2,)
    assert out["noisy_probabilities"].shape == (2,)
    assert out["grad_predicted"].shape == (2,)
    assert out["grad_ideal"].shape == (2,)
    assert out["grad_rho"].shape == (2, 2)
    assert out["grad_C"].shape == (2, 2)


def _self_test_total_objective_l2_zero_fit() -> None:
    from config import make_default_experiment_config
    from measurements import build_all_region_povms
    from noise import build_all_reference_confusions, build_all_true_confusions
    from states import generate_consistent_regional_truth_from_global_product

    cfg = make_default_experiment_config()
    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=123,
    )
    povms = build_all_region_povms(cfg, rng=123)
    confusions = build_all_true_confusions(cfg, rng=123)
    refs = build_all_reference_confusions(cfg)

    empirical = {}
    for name in region_states.keys():
        _, noisy = region_prediction(region_states[name], povms[name], confusions[name])
        empirical[name] = noisy

    val_fit = total_data_fit_objective(
        empirical_probabilities=empirical,
        region_states=region_states,
        region_povms=povms,
        region_confusions=confusions,
        loss="l2",
    )
    assert np.isclose(val_fit, 0.0, atol=1e-10)

    val_reg = total_regularized_objective(
        empirical_probabilities=empirical,
        region_states=region_states,
        region_povms=povms,
        region_confusions=confusions,
        reference_confusions=refs,
        lambda_confusion=0.0,
        loss="l2",
    )
    assert np.isclose(val_reg, 0.0, atol=1e-10)


def _self_test_overlap_residual_helpers() -> None:
    from config import make_default_experiment_config
    from states import generate_consistent_regional_truth_from_global_product

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)
    _, _, region_states = generate_consistent_regional_truth_from_global_product(
        cfg=cfg,
        site_model="random_mixed",
        rng=321,
    )

    eta = {}
    for i, j in graph.overlap_pairs:
        info = graph.overlap_info(i, j)
        name_i = graph.region_name(i)
        rho_i = region_states[name_i]
        eta[(name_i, graph.region_name(j))] = partial_trace(
            rho_i,
            dims=graph.region_site_dims(i),
            keep=info.local_keep_i,
        )

    primal = overlap_primal_residual_norm(graph, region_states, eta)
    dual = overlap_dual_residual_norm(beta=1.0, eta_current=eta, eta_previous=eta)

    assert np.isclose(primal, 0.0, atol=1e-10)
    assert np.isclose(dual, 0.0, atol=1e-10)


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the objectives module.
    """
    tests = [
        ("L2 gradient wrt predicted", _finite_difference_check_grad_predicted_l2),
        ("NLL gradient wrt predicted", _finite_difference_check_grad_predicted_nll),
        ("region gradient shapes", _self_test_region_gradient_shapes),
        ("total objective zero-fit check", _self_test_total_objective_l2_zero_fit),
        ("overlap residual helpers", _self_test_overlap_residual_helpers),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All objectives self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

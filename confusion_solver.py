# confusion_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from config import ExperimentConfig, LossConfig, RegionConfig
from measurements import POVM, measurement_map
from noise import (
    apply_confusion_matrix,
    build_all_reference_confusions,
    project_confusion_matrix,
    validate_confusion_matrix,
    validate_region_confusion_collection,
)
from objectives import (
    build_region_shot_dict,
    confusion_subproblem_region_objective,
    gradient_wrt_predicted,
)
from states import validate_region_state_collection


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


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _resolve_loss(loss: Optional[Union[str, LossConfig]], cfg: ExperimentConfig) -> Union[str, LossConfig]:
    return cfg.loss if loss is None else loss


def _resolve_prob_floor(
    loss: Union[str, LossConfig],
    prob_floor: Optional[float],
) -> Optional[float]:
    if prob_floor is not None:
        return float(prob_floor)
    if isinstance(loss, LossConfig):
        return float(loss.prob_floor)
    return None


def _copy_matrix_dict(d: Mapping) -> Dict:
    return {k: np.array(v, copy=True) for k, v in d.items()}


# ============================================================
# Region-local objective and gradient
# ============================================================

def confusion_region_gradient(
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
) -> np.ndarray:
    r"""
    Gradient of the local confusion-update objective:
        D(\hat p, C p_ideal)
        + lambda ||C - C_ref||_F^2
        + (gamma_c / 2) ||C - C_prev||_F^2.

    If
        q = C p_ideal
        g_q = \nabla_q D(\hat p, q),
    then
        grad_C(data) = g_q p_ideal^T.

    Regularization gradients:
        grad_C(reg)  = 2 lambda (C - C_ref)
        grad_C(prox) = gamma_c (C - C_prev)
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_nonnegative_float(gamma_c, "gamma_c")

    empirical = _as_numpy_array(empirical, dtype=float).reshape(-1)
    ideal_probabilities = _as_numpy_array(ideal_probabilities, dtype=float).reshape(-1)
    confusion = _as_numpy_array(confusion, dtype=float)
    reference_confusion = _as_numpy_array(reference_confusion, dtype=float)

    validate_confusion_matrix(confusion)
    validate_confusion_matrix(reference_confusion)

    if confusion.shape != reference_confusion.shape:
        raise ValueError(
            f"confusion and reference_confusion must have the same shape, got "
            f"{confusion.shape} and {reference_confusion.shape}."
        )

    if confusion.shape[1] != ideal_probabilities.size:
        raise ValueError(
            f"ideal_probabilities has length {ideal_probabilities.size}, but confusion "
            f"expects {confusion.shape[1]}."
        )
    if confusion.shape[0] != empirical.size:
        raise ValueError(
            f"empirical has length {empirical.size}, but confusion outputs "
            f"{confusion.shape[0]} outcomes."
        )

    noisy = apply_confusion_matrix(confusion, ideal_probabilities, prob_floor=0.0)
    grad_pred = gradient_wrt_predicted(
        empirical=empirical,
        predicted=noisy,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
    )

    grad = np.outer(grad_pred, ideal_probabilities)
    grad += 2.0 * lambda_confusion * (confusion - reference_confusion)

    if gamma_c > 0.0:
        if confusion_prev is None:
            raise ValueError("confusion_prev must be provided when gamma_c > 0.")
        confusion_prev = _as_numpy_array(confusion_prev, dtype=float)
        if confusion_prev.shape != confusion.shape:
            raise ValueError(
                f"confusion_prev has shape {confusion_prev.shape}, expected {confusion.shape}."
            )
        grad += gamma_c * (confusion - confusion_prev)

    return grad


# ============================================================
# Region-local projected gradient solver
# ============================================================

@dataclass
class RegionConfusionPGInfo:
    converged: bool
    num_iters: int
    initial_objective: float
    final_objective: float
    final_grad_norm: float
    final_relative_change: float


def solve_region_confusion_update_pg(
    empirical: np.ndarray,
    ideal_probabilities: np.ndarray,
    confusion_init: np.ndarray,
    reference_confusion: np.ndarray,
    lambda_confusion: float,
    loss: Union[str, LossConfig],
    shots: Optional[int] = None,
    prob_floor: Optional[float] = None,
    gamma_c: float = 0.0,
    confusion_prev: Optional[np.ndarray] = None,
    step_size: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-8,
    backtracking_factor: float = 0.5,
    armijo_c: float = 1e-4,
    max_backtracking_iters: int = 25,
) -> Tuple[np.ndarray, RegionConfusionPGInfo]:
    """
    Solve one local confusion update by projected gradient descent with backtracking.

    Returns
    -------
    tuple
        (updated_confusion, info)
    """
    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_nonnegative_float(gamma_c, "gamma_c")
    step_size = _ensure_positive_float(step_size, "step_size")
    max_iters = _ensure_positive_int(max_iters, "max_iters")
    tol = _ensure_positive_float(tol, "tol")
    if not (0.0 < backtracking_factor < 1.0):
        raise ValueError(
            f"backtracking_factor must lie in (0,1), got {backtracking_factor}."
        )
    armijo_c = _ensure_nonnegative_float(armijo_c, "armijo_c")
    max_backtracking_iters = _ensure_positive_int(
        max_backtracking_iters,
        "max_backtracking_iters",
    )

    confusion = project_confusion_matrix(confusion_init)
    validate_confusion_matrix(confusion)
    validate_confusion_matrix(reference_confusion)

    current_obj = confusion_subproblem_region_objective(
        empirical=empirical,
        ideal_probabilities=ideal_probabilities,
        confusion=confusion,
        reference_confusion=reference_confusion,
        lambda_confusion=lambda_confusion,
        loss=loss,
        shots=shots,
        prob_floor=prob_floor,
        gamma_c=gamma_c,
        confusion_prev=confusion_prev,
    )
    initial_obj = current_obj

    converged = False
    final_grad_norm = np.nan
    final_rel_change = np.nan
    it_used = 0

    for it in range(1, max_iters + 1):
        grad = confusion_region_gradient(
            empirical=empirical,
            ideal_probabilities=ideal_probabilities,
            confusion=confusion,
            reference_confusion=reference_confusion,
            lambda_confusion=lambda_confusion,
            loss=loss,
            shots=shots,
            prob_floor=prob_floor,
            gamma_c=gamma_c,
            confusion_prev=confusion_prev,
        )
        grad_norm = float(np.linalg.norm(grad, ord="fro"))
        final_grad_norm = grad_norm
        it_used = it

        if grad_norm <= tol:
            converged = True
            final_rel_change = 0.0
            break

        accepted = False
        candidate_best = confusion
        obj_best = current_obj
        step = step_size

        for _ in range(max_backtracking_iters):
            candidate = project_confusion_matrix(confusion - step * grad)
            rel_change = float(
                np.linalg.norm(candidate - confusion, ord="fro")
                / max(np.linalg.norm(confusion, ord="fro"), 1e-12)
            )

            cand_obj = confusion_subproblem_region_objective(
                empirical=empirical,
                ideal_probabilities=ideal_probabilities,
                confusion=candidate,
                reference_confusion=reference_confusion,
                lambda_confusion=lambda_confusion,
                loss=loss,
                shots=shots,
                prob_floor=prob_floor,
                gamma_c=gamma_c,
                confusion_prev=confusion_prev,
            )

            sufficient_decrease_rhs = current_obj - armijo_c * (
                np.linalg.norm(candidate - confusion, ord="fro") ** 2
            ) / max(step, 1e-12)

            if cand_obj <= sufficient_decrease_rhs + 1e-14:
                accepted = True
                candidate_best = candidate
                obj_best = cand_obj
                final_rel_change = rel_change
                break

            if cand_obj < obj_best - 1e-14:
                candidate_best = candidate
                obj_best = cand_obj
                final_rel_change = rel_change

            step *= backtracking_factor

        if not accepted:
            if obj_best < current_obj - 1e-14:
                confusion = candidate_best
                current_obj = obj_best
            else:
                final_rel_change = 0.0
                break
        else:
            confusion = candidate_best
            current_obj = obj_best

        if final_rel_change <= tol:
            converged = True
            break

    confusion = project_confusion_matrix(confusion)
    validate_confusion_matrix(confusion)

    info = RegionConfusionPGInfo(
        converged=converged,
        num_iters=it_used,
        initial_objective=float(initial_obj),
        final_objective=float(current_obj),
        final_grad_norm=float(final_grad_norm),
        final_relative_change=float(final_rel_change),
    )
    return confusion, info


# ============================================================
# Batch solver and result object
# ============================================================

@dataclass
class ConfusionUpdateResult:
    region_confusions: Dict[str, np.ndarray]
    converged: bool
    average_pg_iters: float
    max_pg_iters: int
    history: Dict[str, List[float]]

    def validate(self, cfg: ExperimentConfig) -> None:
        validate_region_confusion_collection(cfg, self.region_confusions)

    def pretty_print(self) -> None:
        print("=" * 72)
        print("ConfusionUpdateResult")
        print("-" * 72)
        print(f"Converged: {self.converged}")
        print(f"Average PG iterations: {self.average_pg_iters:.2f}")
        print(f"Max PG iterations: {self.max_pg_iters}")
        print("=" * 72)


def update_all_confusions(
    cfg: ExperimentConfig,
    empirical_probabilities: Mapping[str, np.ndarray],
    region_states_fixed: Mapping[str, np.ndarray],
    region_povms: Mapping[str, POVM],
    confusion_prev: Mapping[str, np.ndarray],
    reference_confusions: Optional[Mapping[str, np.ndarray]] = None,
    *,
    loss: Optional[Union[str, LossConfig]] = None,
    region_shots: Optional[Mapping[str, int]] = None,
    prob_floor: Optional[float] = None,
    lambda_confusion: Optional[float] = None,
    gamma_c: Optional[float] = None,
    step_size: Optional[float] = None,
    max_iters: Optional[int] = None,
    tol: Optional[float] = None,
    verbose: Optional[bool] = None,
) -> ConfusionUpdateResult:
    """
    Update all regional confusion matrices independently for fixed regional states.

    Parameters
    ----------
    cfg :
        Experiment configuration.
    empirical_probabilities :
        Region-name -> empirical frequency vector.
    region_states_fixed :
        Region-name -> fixed regional state rho^{k+1}.
    region_povms :
        Region-name -> POVM.
    confusion_prev :
        Region-name -> previous confusion matrix C^k.
    reference_confusions :
        Region-name -> reference confusion matrix. If None, built from cfg.
    loss :
        Optional loss override. If None, use cfg.loss.
    region_shots :
        Optional region-name -> shots mapping. If None, use cfg.
    prob_floor :
        Optional probability floor override for NLL.
    lambda_confusion, gamma_c, step_size, max_iters, tol, verbose :
        Optional overrides of cfg.admm settings.

    Returns
    -------
    ConfusionUpdateResult
        Updated confusion matrices and diagnostics.
    """
    loss = _resolve_loss(loss, cfg)
    prob_floor = _resolve_prob_floor(loss, prob_floor)

    lambda_confusion = (
        cfg.admm.lambda_confusion if lambda_confusion is None else lambda_confusion
    )
    gamma_c = cfg.admm.gamma_c if gamma_c is None else gamma_c
    step_size = cfg.admm.confusion_step_size if step_size is None else step_size
    max_iters = cfg.admm.confusion_gd_max_iters if max_iters is None else max_iters
    tol = cfg.admm.confusion_gd_tol if tol is None else tol
    verbose = cfg.admm.verbose if verbose is None else bool(verbose)

    lambda_confusion = _ensure_nonnegative_float(lambda_confusion, "lambda_confusion")
    gamma_c = _ensure_positive_float(gamma_c, "gamma_c")
    step_size = _ensure_positive_float(step_size, "step_size")
    max_iters = _ensure_positive_int(max_iters, "max_iters")
    tol = _ensure_positive_float(tol, "tol")

    validate_region_state_collection(cfg, dict(region_states_fixed), check_overlap_consistency=False)
    validate_region_confusion_collection(cfg, dict(confusion_prev))

    expected_names = {region.name for region in cfg.regions}
    if set(empirical_probabilities.keys()) != expected_names:
        raise ValueError("empirical_probabilities keys must match cfg.regions.")
    if set(region_povms.keys()) != expected_names:
        raise ValueError("region_povms keys must match cfg.regions.")

    if reference_confusions is None:
        reference_confusions = build_all_reference_confusions(cfg)
    else:
        validate_region_confusion_collection(cfg, dict(reference_confusions))

    if region_shots is None:
        region_shots = build_region_shot_dict(cfg)

    updated_confusions: Dict[str, np.ndarray] = {}
    per_region_iters: List[int] = []
    per_region_converged: List[bool] = []
    initial_objectives: List[float] = []
    final_objectives: List[float] = []

    for region in cfg.regions:
        name = region.name
        rho = region_states_fixed[name]
        povm = region_povms[name]
        ideal_probabilities = measurement_map(rho, povm)

        c_new, info = solve_region_confusion_update_pg(
            empirical=empirical_probabilities[name],
            ideal_probabilities=ideal_probabilities,
            confusion_init=confusion_prev[name],
            reference_confusion=reference_confusions[name],
            lambda_confusion=lambda_confusion,
            loss=loss,
            shots=int(region_shots[name]),
            prob_floor=prob_floor,
            gamma_c=gamma_c,
            confusion_prev=confusion_prev[name],
            step_size=step_size,
            max_iters=max_iters,
            tol=tol,
        )

        updated_confusions[name] = c_new
        per_region_iters.append(info.num_iters)
        per_region_converged.append(info.converged)
        initial_objectives.append(info.initial_objective)
        final_objectives.append(info.final_objective)

        if verbose:
            print(
                f"[confusion_solver] region={name} "
                f"iters={info.num_iters:03d} "
                f"conv={info.converged} "
                f"obj0={info.initial_objective:.6e} "
                f"objf={info.final_objective:.6e} "
                f"grad={info.final_grad_norm:.6e}"
            )

    result = ConfusionUpdateResult(
        region_confusions=updated_confusions,
        converged=all(per_region_converged),
        average_pg_iters=float(np.mean(per_region_iters)) if per_region_iters else 0.0,
        max_pg_iters=max(per_region_iters) if per_region_iters else 0,
        history={
            "per_region_iters": [float(v) for v in per_region_iters],
            "initial_objectives": [float(v) for v in initial_objectives],
            "final_objectives": [float(v) for v in final_objectives],
        },
    )
    result.validate(cfg)
    return result


# ============================================================
# Lightweight self-tests
# ============================================================

def _finite_difference_check_confusion_gradient_l2() -> None:
    empirical = np.array([0.55, 0.45], dtype=float)
    ideal = np.array([0.7, 0.3], dtype=float)

    c = np.array([[0.90, 0.10], [0.10, 0.90]], dtype=float)
    c = project_confusion_matrix(c)
    ref = np.eye(2, dtype=float)

    grad = confusion_region_gradient(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion=c,
        reference_confusion=ref,
        lambda_confusion=0.3,
        loss="l2",
        shots=None,
        gamma_c=0.2,
        confusion_prev=c,
    )

    eps = 1e-7
    direction = np.array([[0.3, -0.4], [-0.1, 0.2]], dtype=float)
    direction = direction / np.linalg.norm(direction, ord="fro")

    f_plus = confusion_subproblem_region_objective(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion=c + eps * direction,
        reference_confusion=ref,
        lambda_confusion=0.3,
        loss="l2",
        gamma_c=0.2,
        confusion_prev=c,
    )
    f_minus = confusion_subproblem_region_objective(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion=c - eps * direction,
        reference_confusion=ref,
        lambda_confusion=0.3,
        loss="l2",
        gamma_c=0.2,
        confusion_prev=c,
    )

    fd = (f_plus - f_minus) / (2.0 * eps)
    ip = float(np.sum(grad * direction))

    assert np.isclose(fd, ip, atol=1e-5)


def _self_test_region_confusion_solver_fixed_point() -> None:
    empirical = np.array([0.72, 0.28], dtype=float)
    ideal = np.array([0.72, 0.28], dtype=float)

    c0 = np.eye(2, dtype=float)
    ref = np.eye(2, dtype=float)

    c_new, info = solve_region_confusion_update_pg(
        empirical=empirical,
        ideal_probabilities=ideal,
        confusion_init=c0,
        reference_confusion=ref,
        lambda_confusion=0.1,
        loss="l2",
        gamma_c=1.0,
        confusion_prev=c0,
        step_size=0.2,
        max_iters=50,
        tol=1e-10,
    )

    validate_confusion_matrix(c_new, num_outcomes=2)
    assert np.linalg.norm(c_new - c0, ord="fro") <= 1e-8


def _self_test_batch_update_shapes() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment
    from noise import build_all_initial_confusions

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    sim = simulate_experiment(cfg, truth_mode="global_consistent")
    init_confusions = build_all_initial_confusions(cfg)

    result = update_all_confusions(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_states_fixed=sim.region_states,
        region_povms=sim.region_povms,
        confusion_prev=init_confusions,
        reference_confusions=build_all_reference_confusions(cfg),
        loss="l2",
        region_shots=sim.region_shots,
        lambda_confusion=cfg.admm.lambda_confusion,
        gamma_c=cfg.admm.gamma_c,
        step_size=0.1,
        max_iters=20,
        tol=1e-8,
        verbose=False,
    )

    result.validate(cfg)
    assert set(result.region_confusions.keys()) == {region.name for region in cfg.regions}
    for region in cfg.regions:
        c = result.region_confusions[region.name]
        validate_confusion_matrix(c)


def _self_test_truth_fixed_point_identity_case() -> None:
    from config import make_default_experiment_config
    from simulator import simulate_experiment

    cfg = make_default_experiment_config()
    cfg.loss.name = "l2"
    cfg.simulation.use_shot_noise = False

    # Force true confusion = identity and initial = identity
    for region in cfg.regions:
        region.true_confusion_model = "identity"
        region.init_confusion_method = "identity"

    sim = simulate_experiment(cfg, truth_mode="global_consistent")
    result = update_all_confusions(
        cfg=cfg,
        empirical_probabilities=sim.empirical_probabilities,
        region_states_fixed=sim.region_states,
        region_povms=sim.region_povms,
        confusion_prev=sim.region_confusions,
        reference_confusions=build_all_reference_confusions(cfg),
        loss="l2",
        region_shots=sim.region_shots,
        lambda_confusion=cfg.admm.lambda_confusion,
        gamma_c=cfg.admm.gamma_c,
        step_size=0.1,
        max_iters=50,
        tol=1e-10,
        verbose=False,
    )

    for region in cfg.regions:
        name = region.name
        assert np.linalg.norm(
            result.region_confusions[name] - sim.region_confusions[name],
            ord="fro",
        ) <= 1e-7


def run_self_tests(verbose: bool = True) -> None:
    """
    Run a lightweight smoke-test suite for the confusion_solver module.
    """
    tests = [
        ("finite-difference confusion gradient (L2)", _finite_difference_check_confusion_gradient_l2),
        ("region confusion fixed point", _self_test_region_confusion_solver_fixed_point),
        ("batch confusion update shapes", _self_test_batch_update_shapes),
        ("truth fixed point in identity case", _self_test_truth_fixed_point_identity_case),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All confusion_solver self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)

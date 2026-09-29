from __future__ import annotations

import math
import time
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ag_vs.utils import isolated_trial
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from ag_vs.EarlyStopMechanism import EarlyStopMechanism
from ag_vs.Figure import AGVSTreeFigure
from ag_vs.GaussianProcess import GaussianProcessManager
from ag_vs.helper_functions import merge, split, traverse, update
from ag_vs.utils import (
    AGVSResult,
    Array,
    Node,
    build_node_score_history,
    calculate_score,
    format_plot_value,
    sample_acquisition_candidates,
    sample_initial_points,
    SharedObservationIndices,
    weighted_score,
)
from ag_vs.inner_optimizer.turbo import (
    fit_turbo_node_gp,
    propose_turbo_one,
    propose_turbo_q,
    update_turbo_after_batch,
)
from ag_vs.inner_optimizer.vanillabo import (
    propose_vanilla_independent_batch,
    propose_vanilla_joint_q,
    propose_vanilla_one,
)
from Experiment.benchmark.baseline.reference_infrastructure import (
    BO_REFERENCE_SPEC,
    TURBO_REFERENCE_SPEC,
    bo_reference_dict,
    classical_expected_improvement,
    turbo_reference_dict,
)


def agvs_selection_batch_size(eff_dim: int) -> int:
    """Return the current AG-VS experiment batch_size."""
    del eff_dim
    return 3


def _validated_integer(
    name: str,
    value,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Return an integer-valued configuration parameter or raise ValueError."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite integer; got {value!r}") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite integer; got {value!r}")
    result = int(numeric)
    if minimum is not None and result < int(minimum):
        raise ValueError(f"{name} must be at least {int(minimum)}; got {result}")
    if maximum is not None and result > int(maximum):
        raise ValueError(f"{name} must be at most {int(maximum)}; got {result}")
    return result


class AGVSBaseOptimizer:
    """Adaptive Grouping Decomposition Bayesian Optimization.

    Maximization is assumed. For minimization use objective=lambda x: -f(x).

    Dimension groups are formed from seeded random permutations.  Node-local
    acquisition GPs use isotropic Matérn-5/2 kernels.
    """

    def __init__(
        self,
        objective: Callable[[Array], float],
        bounds: Array,
        eff_dim: int,
        max_iter: int = 200,
        n_init: Optional[int] = None,
        noise_std: Optional[float] = None,
        noise_variance: Optional[float] = None,
        learn_noise: bool = False,
        tau: Optional[float] = None,
        beta: float = 2.0,
        acquisition: str = "ei",
        candidate_count: int = BO_REFERENCE_SPEC.candidate_count,
        acquisition_sampling_strategy: str = "sobol",
        q_batch: Optional[int] = None,
        batch_size: Optional[int] = 3,
        t_seq: int = 2,
        initial_sampling_strategy: str = "lhs",
        initial_points: Optional[Array] = None,
        seed: Optional[int] = None,
        data_assignment: str = "all",
        verbose: bool = True,
        printer_on: bool = True,
        active_indices: Optional[List[int]] = None,
        print_tree_each_iter: bool = True,
        tree_plot_on: bool = False,
        tree_plot_path: Optional[str] = None,
        tree_plot_html_path: Optional[str] = None,
        tree_plot_open_browser: bool = False,
        tree_plot_pause: float = 0.001,
        risk_tolerances: Optional[List[float]] = None,
        kernel_selection: str = "fixed",
        default_kernel: str = "matern52",
        early_stop_enabled: bool = False,
        experiment_mode: bool = False,
        figure_mode: bool = False,
        important_indices: Optional[List[int]] = None,
        ipt_solver: str = "vanillabo",
        sim_mode: bool = False,
        n_split: Optional[int] = None,
        N_bad: Optional[int] = None,
        N_bad_enabled: bool = True,
        merge_enabled: bool = True,
        warm_start_root_gp: bool = True,
        incremental_root_gp: bool = True,
    ) -> None:
        if not callable(objective):
            raise ValueError("objective must be callable")
        self.objective = objective
        try:
            self.bounds = np.asarray(bounds, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("bounds must be a finite numeric array with shape (D, 2)") from exc
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or self.bounds.shape[0] == 0:
            raise ValueError("bounds must have non-empty shape (D, 2)")
        if not np.all(np.isfinite(self.bounds)):
            raise ValueError("bounds must contain only finite lower and upper values")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("each bound must satisfy upper > lower")

        self.D = self.bounds.shape[0]
        self.eff_dim = _validated_integer("eff_dim", eff_dim, minimum=1)
        if self.eff_dim >= self.D:
            raise ValueError(
                "eff_dim must be smaller than the ambient dimension D for AG-VS grouping; "
                f"got eff_dim={self.eff_dim}, D={self.D}"
            )
        self.max_iter = _validated_integer("max_iter", max_iter, minimum=1)
        self.sim_mode = bool(sim_mode)

        # q_batch and batch_size are two different acquisition semantics and
        # must never be enabled together.
        if (q_batch is None) == (batch_size is None):
            raise ValueError(
                "AG-VS requires exactly one of q_batch or batch_size; "
                "q_batch uses joint acquisition and batch_size uses independent pointwise acquisition."
            )
        local_q_batch = (
            None
            if q_batch is None
            else _validated_integer("q_batch", q_batch, minimum=1)
        )
        local_batch_size = (
            None
            if batch_size is None
            else _validated_integer("batch_size", batch_size, minimum=1)
        )
        local_t_seq = _validated_integer("t_seq", t_seq, minimum=1)

        if n_init is not None:
            local_n_init = _validated_integer("n_init", n_init, minimum=1)
        else:
            local_n_init = max(2 * (self.eff_dim + 1), self.D + 1)

        # Calculate N_split and expected iteration
        from ag_vs.utils import find_optimum_B_and_cost
        if self.sim_mode:
            opt_B, _ = find_optimum_B_and_cost(
                self.D,
                self.eff_dim,
                local_n_init,
                local_q_batch,
                local_batch_size,
                local_t_seq,
            )
            n_split_val = math.ceil(opt_B)
            self._expected_iteration = None
        else:
            if n_split is None:
                raise ValueError("AGVS sim_mode=False requires n_split to be provided")
            n_split_val = _validated_integer("n_split", n_split, minimum=2, maximum=self.D)
            if n_split_val <= self.eff_dim:
                raise ValueError(
                    f"AGVS n_split must be greater than eff_dim when sim_mode=False; "
                    f"got n_split={n_split_val}, eff_dim={self.eff_dim}"
                )
            self._expected_iteration = None

        self.split_k = max(2, int(n_split_val))
        self.N_bad = (
            3 * int(self.split_k)
            if N_bad is None
            else _validated_integer("N_bad", N_bad, minimum=1)
        )
        self.N_bad_enabled = bool(N_bad_enabled)
        if not isinstance(merge_enabled, (bool, np.bool_)):
            raise ValueError("merge_enabled must be a boolean.")
        self.merge_enabled = bool(merge_enabled)
        self.consecutive_bad_node_selections = 0
        self.tree_restart_count = 0
        self.tree_restart_history: List[Dict] = []
        self.warm_start_root_gp = bool(warm_start_root_gp)
        self.incremental_root_gp = bool(incremental_root_gp)
        self._last_full_dim_gp_model = None

        self.n_init = local_n_init if n_init is not None else max(2 * self.split_k, self.D + 1)
        self.n_init = _validated_integer("n_init", self.n_init, minimum=1)
        if self.sim_mode:
            _, min_cost = find_optimum_B_and_cost(
                self.D,
                self.eff_dim,
                self.n_init,
                local_q_batch,
                local_batch_size,
                local_t_seq,
            )
            self._expected_iteration = int(min_cost)
        if noise_variance is None:
            if noise_std is None:
                resolved_noise_variance = float(BO_REFERENCE_SPEC.alpha)
            else:
                try:
                    resolved_noise_std = float(noise_std)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("noise_std must be finite and non-negative") from exc
                if not np.isfinite(resolved_noise_std) or resolved_noise_std < 0.0:
                    raise ValueError("noise_std must be finite and non-negative")
                resolved_noise_variance = (
                    float(BO_REFERENCE_SPEC.alpha)
                    if resolved_noise_std
                    == float(BO_REFERENCE_SPEC.observation_noise_std)
                    else resolved_noise_std**2
                )
        else:
            try:
                resolved_noise_variance = float(noise_variance)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("noise_variance must be finite and non-negative") from exc
            if not np.isfinite(resolved_noise_variance) or resolved_noise_variance < 0.0:
                raise ValueError("noise_variance must be finite and non-negative")
            if noise_std is not None:
                try:
                    supplied_noise_std = float(noise_std)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("noise_std must be finite and non-negative") from exc
                if (
                    not np.isfinite(supplied_noise_std)
                    or supplied_noise_std < 0.0
                    or not np.isclose(
                        supplied_noise_std**2,
                        resolved_noise_variance,
                    )
                ):
                    raise ValueError(
                        "noise_std and noise_variance must describe the same variance"
                    )
        self.noise_variance = float(resolved_noise_variance)
        self.noise_std = float(math.sqrt(self.noise_variance))
        self.learn_noise = bool(learn_noise)
        if bool(early_stop_enabled) and (not bool(experiment_mode)) and tau is None:
            raise ValueError(
                "AGVS experiment_mode=False requires tau when early stopping is enabled."
            )
        if tau is None:
            self.tau = None
        else:
            try:
                self.tau = float(tau)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"tau must be finite when supplied; got {tau!r}") from exc
            if not np.isfinite(self.tau):
                raise ValueError(f"tau must be finite when supplied; got {tau!r}")

        try:
            self.beta = float(beta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"beta must be finite; got {beta!r}") from exc
        if not np.isfinite(self.beta):
            raise ValueError(f"beta must be finite; got {beta!r}")
        self.acquisition = str(acquisition).strip().lower()
        if self.acquisition not in {"ei"}:
            raise ValueError(
                "AG-VS-BO requires MCTS-VS classical analytic Expected Improvement ('ei')."
            )

        self.candidate_count = _validated_integer("candidate_count", candidate_count, minimum=1)
        if self.candidate_count != int(BO_REFERENCE_SPEC.candidate_count):
            raise ValueError(
                f"AG-VS uses exactly {BO_REFERENCE_SPEC.candidate_count} BO candidates; "
                f"got {self.candidate_count}."
            )
        self.acquisition_sampling_strategy = str(acquisition_sampling_strategy).strip().lower().replace("-", "_")
        if self.acquisition_sampling_strategy not in {"sobol", "sobol_scrambled"}:
            raise ValueError("AG-VS-BO candidate generation must be scrambled Sobol.")
        self.q_batch = local_q_batch
        self.batch_size = local_batch_size
        self.selection_batch_size = self.batch_size
        self.acquisition_batch_mode = (
            "q_batch" if self.q_batch is not None else "batch_size"
        )
        self.t_seq = local_t_seq

        self.seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.initial_sampling_strategy = str(initial_sampling_strategy).strip().lower()
        if initial_points is None:
            self.initial_points = None
        else:
            initial_points = np.asarray(initial_points, dtype=float)
            if initial_points.ndim != 2 or initial_points.shape[1] != self.D:
                raise ValueError("initial_points must have shape (n_points, D)")
            if not np.all(np.isfinite(initial_points)):
                raise ValueError("initial_points must contain only finite values")
            self.initial_points = np.clip(initial_points, self.bounds[:, 0], self.bounds[:, 1])

        if data_assignment not in {"all", "round_robin", "random"}:
            raise ValueError("data_assignment must be one of: 'all', 'round_robin', 'random'")
        self.data_assignment = data_assignment

        self.verbose = bool(verbose)
        self.printer_on = bool(printer_on)
        self.early_stop_enabled = bool(early_stop_enabled)
        self.experiment_mode = bool(experiment_mode)
        self.figure_mode = bool(figure_mode)
        if not self.experiment_mode:
            active_indices = None
        elif important_indices is not None:
            active_indices = important_indices
        if self.experiment_mode:
            if active_indices is None:
                raise ValueError("experiment_mode=True requires active_indices or important_indices.")
            active_array = np.asarray(active_indices, dtype=int).reshape(-1)
            if active_array.size == 0:
                raise ValueError("experiment_mode=True requires at least one active dimension index.")
            invalid = [int(dim) for dim in active_array if int(dim) < 0 or int(dim) >= int(self.D)]
            if invalid:
                raise ValueError(
                    f"active_indices must be valid extrinsic dimension indices in [0, {self.D - 1}]. "
                    f"Invalid active_indices={invalid}."
                )
            self.active_indices = {int(dim) for dim in active_array}
        else:
            self.active_indices = set()
        self.ipt_solver = str(ipt_solver).strip().lower()

        self.print_tree_each_iter = bool(print_tree_each_iter)
        self.tree_plot_on = bool(tree_plot_on)
        self.tree_plot_path = tree_plot_path
        self.tree_plot_html_path = tree_plot_html_path
        self.tree_plot_open_browser = bool(tree_plot_open_browser)
        try:
            self.tree_plot_pause = float(tree_plot_pause)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"tree_plot_pause must be finite and non-negative; got {tree_plot_pause!r}") from exc
        if not np.isfinite(self.tree_plot_pause) or self.tree_plot_pause < 0.0:
            raise ValueError(f"tree_plot_pause must be finite and non-negative; got {tree_plot_pause!r}")
        self.tree_plot_fig = None
        self.tree_plot_ax = None
        self._tree_plot_browser_opened = False

        self.X: List[Array] = []
        self.y: List[float] = []
        self.true_y: List[float] = []
        self.history: List[Dict] = []
        self.gp_fit_history: List[Dict] = []
        self._gp_fit_counter = 0
        self._posterior_gp_fit_counter = 0
        # Global dataset D: one maximum-mu record per tolerance-equivalent x.
        self.posterior_x_tolerance = 1e-10
        # Reinitialization uses this absolute, component-wise epsilon for both
        # x and the observed y when identifying duplicate observation pairs.
        self.restart_pair_epsilon = 1e-10
        self.global_dataset: Dict[Tuple[int, ...], Dict] = {}
        self.global_best_record: Optional[Dict] = None
        self.historical_mu_best_by_index: Dict[int, float] = {}
        self.historical_best_observed_by_index: Dict[int, float] = {}
        if risk_tolerances is None:
            raise ValueError("AGVS requires risk_tolerances.")
        self.risk_tolerances = [float(value) for value in risk_tolerances]
        if not self.risk_tolerances:
            raise ValueError("risk_tolerances must contain at least one value.")
        invalid_risks = [value for value in self.risk_tolerances if not (0.0 < float(value) < 1.0)]
        if invalid_risks:
            raise ValueError(f"risk_tolerances must be in (0, 1). Invalid values: {invalid_risks}")
        self.threshold_trace: List[Dict] = []
        self.kernel_selection = str(kernel_selection).strip().lower()
        self.default_kernel = str(default_kernel).strip().lower()
        if self.kernel_selection not in {"fixed", "default"} or self.default_kernel != "matern52":
            raise ValueError("AG-VS fair-comparison runs require a fixed Constant x Matern-5/2 kernel.")
        self.bo_infrastructure = bo_reference_dict()
        self.bo_infrastructure.update(
            {
                "kernel": "ConstantKernel(1.0) * Matern(shared length_scale, nu=2.5)",
                "ard": False,
                "isotropic": True,
                "acquisition": "classical analytic Expected Improvement",
                "candidate_selection": (
                    "joint Monte-Carlo qEI in q_batch mode; top-k pointwise "
                    "analytic EI in independent batch_size mode"
                ),
                "effective_acquisition": (
                    "greedy Monte-Carlo joint qExpectedImprovement"
                    if self.q_batch is not None
                    else "top-k independent pointwise analytic Expected Improvement"
                ),
                "qei_mc_samples": 256 if self.q_batch is not None else None,
                "alpha": self.noise_variance,
                "noise_variance": self.noise_variance,
                "observation_noise_std": self.noise_std,
            }
        )
        self.gp_manager = GaussianProcessManager(
            noise_variance=self.noise_variance,
            learn_noise=self.learn_noise,
            kernel_selection=self.kernel_selection,
            default_kernel=self.default_kernel,
        )
        self.tree_figure = AGVSTreeFigure(self)
        self.early_stop = EarlyStopMechanism(
            tau=self.tau,
            risk_tolerances=self.risk_tolerances,
            n_init=self.n_init,
            noise_std=self.noise_std,
            learn_noise=self.learn_noise,
        )
        self.best_stoppable_threshold_by_delta = self.early_stop.best_threshold_by_delta
        self.threshold_trace = self.early_stop.trace

        self.root = Node(dims=list(range(self.D)), h=0, idx=0, parent=None)
        self.root.local_bounds = self.bounds.copy()
        self.root.diameter = self._node_diameter(self.root)
        # The no-merge comparison changes only cohort merging. Children are
        # still scored, the highest-scoring active child is selected, and it
        # continues splitting while its dimensionality permits.
        self.root.sad_no_merge_branch = not self.merge_enabled
        self.root.merge_disabled = int(not self.merge_enabled)

        self.roots: List[Node] = [self.root]
        self._all_nodes: List[Node] = [self.root]
        self._plot_nodes: List[Node] = [self.root]
        self._next_node_idx = 1

    @property
    def expected_iteration(self) -> Optional[int]:
        return self._expected_iteration

    # ------------------------------------------------------------------
    # Formatting / printing
    # ------------------------------------------------------------------
    def _print_event(self, message: str) -> None:
        if self.verbose and self.printer_on:
            print(message, flush=True)

    def _print_run_settings(self) -> None:
        if not self.printer_on:
            return
        runtime_eliminate = getattr(self, "sad_eliminate", None)
        if runtime_eliminate is None:
            runtime_eliminate = max(1, int(self.split_k) - int(self.eff_dim))
        print(
            "[AGVS][settings] "
            f"backend={self.ipt_solver} "
            f"acquisition={self.acquisition} "
            f"q_batch={self.q_batch} "
            f"batch_size={self.batch_size} "
            f"selection_batch_size={self.selection_batch_size} "
            f"t_seq={self.t_seq} "
            f"N_bad={self.N_bad} "
            f"N_bad_enabled={self.N_bad_enabled} "
            f"merge_enabled={self.merge_enabled} "
            f"warm_start_root_gp={self.warm_start_root_gp} "
            f"incremental_root_gp={self.incremental_root_gp} "
            f"sim_mode={self.sim_mode} "
            f"noise_std={self.noise_std:g} "
            f"noise_variance={self.noise_variance:g} "
            f"kernel={self.kernel_selection}:{self.default_kernel} "
            f"device={getattr(self, 'turbo_device', 'cpu')} "
            f"early_stop={self.early_stop_enabled} "
            f"experiment_mode={self.experiment_mode} "
            f"figure_mode={self.figure_mode}",
            flush=True,
        )
        calculated_parts = [
            "[AGVS][calculated-settings]",
            f"N_split={int(self.split_k)}",
            f"Eliminate={int(runtime_eliminate)}",
            "selection_mechanism=canonical_merged_leaf",
        ]
        if self.sim_mode and self.expected_iteration is not None:
            calculated_parts.append(f"expected_iter={int(self.expected_iteration)}")
        calculated_parts.extend([
            f"max_iter={int(self.max_iter)}",
            f"n_init={int(self.n_init)}",
            f"D={int(self.D)}",
            f"eff_dim={int(self.eff_dim)}",
            f"selection_batch_size={self.selection_batch_size}",
        ])
        print(
            " ".join(calculated_parts),
            flush=True,
        )

    def _terminal_summary_on(self) -> bool:
        return bool(self.printer_on)

    def _early_stop_triggered(self, stop_prob: float) -> bool:
        if not self.early_stop_enabled or not np.isfinite(float(stop_prob)):
            return False
        risk_tolerance = float(self.risk_tolerances[0])
        return float(stop_prob) >= 1.0 - float(risk_tolerance)

    def _threshold_label(self) -> Optional[str]:
        if not self.risk_tolerances:
            return None
        return f"{float(self.risk_tolerances[0]):.12g}"

    def _threshold_value(self, threshold_entry: Optional[Dict], field: str) -> Optional[float]:
        if not isinstance(threshold_entry, dict):
            return None
        label = self._threshold_label()
        values = threshold_entry.get(field, {})
        if not isinstance(values, dict) or label is None:
            return None
        value = values.get(label)
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _format_optional_float(value: Optional[float]) -> str:
        if value is None or not np.isfinite(float(value)):
            return "None"
        return f"{float(value):.6g}"

    def _print_iteration_summary(
        self,
        *,
        iteration: int,
        best_value: float,
        current_value: float,
        selected_node: Node,
        threshold_entry: Optional[Dict],
        selected_dims_for_display: Optional[List[int]] = None,
    ) -> None:
        if not self._terminal_summary_on():
            return
        display_dims = (
            list(selected_node.dims)
            if selected_dims_for_display is None
            else [int(dim) for dim in selected_dims_for_display]
        )
        unique_display_dims = {int(dim) for dim in display_dims}
        if self.experiment_mode:
            selected_active_count = len(unique_display_dims.intersection(self.active_indices))
            selected_dim_text = f"Selected Active Dims: {selected_active_count}/{len(self.active_indices)}"
        else:
            selected_dim_text = f"Selected Dim Count: {len(unique_display_dims)}"
        selected_node_text = (
            f" Selected Node: $N_{{{int(selected_node.h)},{int(selected_node.idx)}}}$"
        )
        green = "\033[92m"
        reset = "\033[0m"
        best_text = f"{green}{float(best_value):.6g}{reset}"
        infinite_dim_score_text = ""
        if hasattr(self, "_terminal_infinite_dim_score_count"):
            try:
                infinite_dim_score_text = (
                    " # of Inf Dim Score: "
                    f"{int(getattr(self, '_terminal_infinite_dim_score_count')(selected_node))}"
                )
            except Exception:
                infinite_dim_score_text = ""
        if self.early_stop_enabled:
            print(
                f"Iter: {int(iteration)} "
                f"Best Value: {best_text} "
                f"Current Value: {float(current_value):.6g} "
                f"{selected_dim_text}"
                f"{infinite_dim_score_text}"
                f"{selected_node_text}",
                flush=True,
            )
        else:
            best_threshold = self._threshold_value(threshold_entry, "best_stoppable_threshold_by_delta")
            current_threshold = self._threshold_value(threshold_entry, "current_stoppable_threshold_by_delta")
            best_threshold_text = f"{green}{self._format_optional_float(best_threshold)}{reset}"
            print(
                f"Iter: {int(iteration)} "
                f"Best Value: {best_text} "
                f"Best Stopable Threshold: {best_threshold_text} "
                f"Current Value: {float(current_value):.6g} "
                f"Current Stopable Threshold: {self._format_optional_float(current_threshold)} "
                f"{selected_dim_text}"
                f"{infinite_dim_score_text}"
                f"{selected_node_text}",
                flush=True,
            )

    def _format_dim(self, dim: int) -> str:
        text = str(int(dim))
        if self.experiment_mode and int(dim) in self.active_indices:
            return f"\033[92m{text}\033[0m"
        return text

    def _format_dims(self, dims: List[int]) -> str:
        if len(dims) > 70:
            shown = dims[:35] + ["..."] + dims[-20:]  # type: ignore[list-item]
            return "[" + ", ".join(str(d) if d == "..." else self._format_dim(int(d)) for d in shown) + "]"
        return "[" + ", ".join(self._format_dim(dim) for dim in dims) + "]"

    def _node_label(self, node: Node) -> str:
        return f"{int(node.h)},{int(node.idx)}"

    def _node_active_dims(self, node: Node) -> List[int]:
        if not self.experiment_mode:
            return []
        return [dim for dim in node.dims if dim in self.active_indices]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _visible_plot_nodes(self) -> List[Node]:
        return self.tree_figure.visible_nodes()

    @staticmethod
    def _format_plot_value(value: float, digits: int = 2) -> str:
        return format_plot_value(value, digits=digits)

    def _plot_node_score_lines(self, node: Node) -> List[str]:
        if self._uses_cached_ei_selection():
            return [
                f"EI={self._format_plot_value(getattr(node, 'cached_ei_value', -np.inf))}",
            ]
        return []

    def _plot_node_marker(self, ax, x: float, y: float, node: Node, selected: bool, on_path: bool, has_active: bool) -> None:
        is_frozen = bool(getattr(node, "split_frozen", False))
        is_dropped = bool(getattr(node, "sad_dropped", False))
        is_excluded = bool(getattr(node, "sad_excluded_from_merge", False))
        is_merged_node = bool(getattr(node, "sad_merged_node", False))
        if is_frozen:
            face = "#e5e7eb"
        elif is_merged_node:
            face = "#fef3c7"
        else:
            face = "#22c55e" if has_active else ("#ead7ff" if node.split_ready else "#d8e8ff")
        edge = "#d62728" if selected else ("#ef4444" if on_path else "#1f2937")
        linewidth = 3.8 if selected else (3.0 if on_path else 1.2)
        ax.scatter([x], [y], s=230000 if selected else 200000, c=face, edgecolors=edge, linewidths=linewidth, zorder=3)

    def _plot_group_tree(
        self,
        selected: Optional[Node] = None,
        iteration: Optional[int] = None,
        force: bool = False,
    ) -> None:
        self.tree_figure.plot(selected=selected, iteration=iteration, force=force)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @isolated_trial
    def optimize(self) -> AGVSResult:
        start = time.time()
        self._print_run_settings()
        self._initial_design()
        if len(self.y) >= self.n_init:
            self._fit_local_gp(self.root)
            update(self, self.root)
            if self._uses_cached_ei_selection():
                self._refresh_node_cached_ei(self.root)
            # Posterior stopping/threshold diagnostics are disabled. The
            # initial fit is retained because acquisition and splitting need it.
        stopped = False
        stop_iteration: Optional[int] = None

        while len(self.y) < self.max_iter:
            q_start_eval = len(self.y) + 1
            self._print_event(
                f"[AGVS][progress] starting iteration {q_start_eval}/{self.max_iter} "
                f"q_batch={self.q_batch} batch_size={self.batch_size} t_seq={self.t_seq}"
            )

            if self.root.is_leaf:
                self._fit_local_gp(self.root)

            selection_best_observed_before = self._global_observed_best()
            selected_node, traversal_path = traverse(self)
            selected_node = self._prepare_selected_node_for_evaluation(selected_node)
            # Exactly one point-batch mode is active. q_batch uses joint
            # acquisition; batch_size uses independent pointwise acquisition.
            # t_seq repeats acquisition/evaluate. A stale model is refitted only
            # before the next acquisition, never merely because a batch ended.
            configured_point_batch_size = (
                self.q_batch if self.q_batch is not None else self.batch_size
            )
            remaining_evaluations = self.max_iter - len(self.y)
            budget_safe_stages = int(
                math.ceil(remaining_evaluations / int(configured_point_batch_size))
            )
            seq_limit = min(int(self.t_seq), budget_safe_stages)

            for seq_index in range(seq_limit):
                sequence_index = seq_index + 1
                acquisition_group_index = seq_index + 1
                q_limit = self._evaluation_batch_size(
                    selected_node,
                    self.max_iter - len(self.y),
                )
                if q_limit <= 0:
                    break
                # Every point in this proposal batch is scored against the
                # same pre-batch incumbent. Refresh it for the next batch,
                # including the next t_seq step within this node visit.
                batch_best_observed_before = self._global_observed_best()
                if int(q_limit) > 1:
                    if not self._node_gp_is_current(selected_node):
                        self._fit_local_gp(selected_node)
                    proposals = self._propose_q_from_leaf(selected_node, q_limit)
                    joint_q_acquisition = bool(getattr(self, "_last_joint_q_acquisition", False))
                    q_results = []
                    for q_index, (x_next, acq_value) in enumerate(proposals, start=1):
                        y_true, y_obs = self._evaluate(x_next)
                        new_idx = len(self.y) - 1
                        self.historical_best_observed_by_index[int(new_idx)] = float(
                            batch_best_observed_before
                        )
                        self._assign_new_index_to_leaf_tree(new_idx, selected_node)
                        q_results.append((int(new_idx), float(y_true), float(y_obs), float(acq_value), int(q_index)))
                    if not q_results:
                        break

                    if hasattr(self, "_after_proposal_batch"):
                        self._after_proposal_batch(
                            selected_node,
                            [int(item[0]) for item in q_results],
                        )

                    if seq_index + 1 == seq_limit:
                        update(self, selected_node, traversal_path=traversal_path)

                    threshold_entry = None
                    stop_prob, stop_mu, stop_std, stop_index = 0.0, float("nan"), float("nan"), None

                    for new_idx, y_true, y_obs, acq_value, q_index in q_results:
                        observed_prefix = np.asarray(
                            self.y[: int(new_idx) + 1], dtype=float
                        )
                        best_idx = int(np.argmax(observed_prefix))
                        best_value = float(observed_prefix[best_idx])
                        t_eval = int(new_idx) + 1
                        rec = {
                            "iteration": t_eval,
                            "batch_iteration": int(q_start_eval),
                            "q_iteration": int(q_start_eval),
                            "q_index": int(q_index),
                            "q_batch": None if self.q_batch is None else int(self.q_batch),
                            "batch_index": int(acquisition_group_index),
                            "batch_size": None if self.batch_size is None else int(self.batch_size),
                            "acquisition_batch_size": int(len(q_results)),
                            "selection_batch_size": (
                                None
                                if self.selection_batch_size is None
                                else int(self.selection_batch_size)
                            ),
                            "acquisition_batch_mode": self.acquisition_batch_mode,
                            "optimizer_point_batch_size": int(len(q_results)),
                            "t_seq": int(self.t_seq),
                            "sequence_index": int(sequence_index),
                            "selected_node": selected_node.idx,
                            "selected_node_label": self._node_label(selected_node),
                            "selected_node_split_type": str(getattr(selected_node, "split_type", "")),
                            "selected_dims": list(selected_node.dims),
                            "D_h_i": selected_node.D_hi,
                            "Dim_h_i": selected_node.Dim_hi,
                            "node_depth": selected_node.h,
                            "node_diameter": float(selected_node.diameter),
                            "local_n": int(selected_node.n),
                            "node_ave_diff_score": (
                                None
                                if not hasattr(selected_node, "ave_diff_score")
                                else float(getattr(selected_node, "ave_diff_score"))
                            ),
                            "node_mean_abs_best_diff": (
                                None
                                if not hasattr(selected_node, "mean_abs_best_diff")
                                else float(getattr(selected_node, "mean_abs_best_diff"))
                            ),
                            "node_risk_bonus": (
                                None
                                if not hasattr(selected_node, "risk_bonus")
                                else float(getattr(selected_node, "risk_bonus"))
                            ),
                            "node_score_risk_tolerance": (
                                None
                                if not hasattr(selected_node, "score_risk_tolerance")
                                else float(getattr(selected_node, "score_risk_tolerance"))
                            ),
                            "selected_acq_dims": (
                                None
                                if not hasattr(self, "_imp_eval_dims_by_index")
                                else list(getattr(self, "_imp_eval_dims_by_index", {}).get(int(new_idx), []))
                            ),
                            "acquisition": (
                                self._node_proposal_strategy(selected_node)
                                if hasattr(self, "_node_proposal_strategy")
                                else self.acquisition
                            ),
                            "joint_q_acquisition": joint_q_acquisition,
                            "proposal_strategy": (
                                self._node_proposal_strategy(selected_node)
                                if hasattr(self, "_node_proposal_strategy")
                                else self.acquisition
                            ),
                            "node_backend": (
                                self._node_backend_for_history(selected_node)
                                if hasattr(self, "_node_backend_for_history")
                                else self.acquisition
                            ),
                            "acq_value": float(acq_value),
                            "y_true": float(y_true) if self.experiment_mode else None,
                            "y_obs": float(y_obs),
                            "best_y": best_value,
                            "best_y_source": "running_best_observed_y",
                            "best_true_y": float(self.true_y[best_idx]) if self.experiment_mode else None,
                            "stop_prob": float(stop_prob),
                            "stop_mu": float(stop_mu),
                            "stop_std": float(stop_std),
                            "stop_index": None if stop_index is None else int(stop_index),
                            "batch_endpoint": bool(t_eval == len(self.y)),
                            "threshold_entry": threshold_entry,
                            "elapsed_sec": time.time() - start,
                            "num_nodes": len(self._all_nodes),
                            "num_leaves": len(self._leaves()),
                            "split_after_eval": False,
                            "update_after_all_q": True,
                            "retrained_after_eval": False,
                            "retrained_after_q": False,
                            "gp_refit_after_t_seq_step": False,
                        }
                        self.history.append(rec)
                        self._print_iteration_summary(
                            iteration=t_eval,
                            best_value=best_value,
                            current_value=float(y_obs),
                            selected_node=selected_node,
                            threshold_entry=threshold_entry,
                            selected_dims_for_display=rec.get("selected_acq_dims") or rec.get("selected_dims"),
                        )
                    self._plot_group_tree(selected=selected_node, iteration=len(self.y))
                    continue

                if (
                    self._uses_cached_ei_selection()
                    and seq_index == 0
                    and getattr(selected_node, "cached_ei_x", None) is not None
                    and np.isfinite(float(getattr(selected_node, "cached_ei_value", -np.inf)))
                ):
                    x_next = np.asarray(getattr(selected_node, "cached_ei_x"), dtype=float).copy()
                    acq_value = float(getattr(selected_node, "cached_ei_value"))
                else:
                    self._fit_local_gp(selected_node)
                    x_next, acq_value = self._propose_from_leaf(selected_node)

                y_true, y_obs = self._evaluate(x_next)
                new_idx = len(self.y) - 1
                self.historical_best_observed_by_index[int(new_idx)] = float(
                    batch_best_observed_before
                )
                self._assign_new_index_to_leaf_tree(new_idx, selected_node)
                if hasattr(self, "_after_proposal_batch"):
                    self._after_proposal_batch(selected_node, [int(new_idx)])

                if seq_index + 1 == seq_limit:
                    update(self, selected_node, traversal_path=traversal_path)
                threshold_entry = None
                stop_prob, stop_mu, stop_std, stop_index = 0.0, float("nan"), float("nan"), None

                observed_prefix = np.asarray(
                    self.y[: int(new_idx) + 1], dtype=float
                )
                best_idx = int(np.argmax(observed_prefix))
                best_value = float(observed_prefix[best_idx])
                t_eval = int(new_idx) + 1
                rec = {
                    "iteration": t_eval,
                    "batch_iteration": int(q_start_eval),
                    "q_iteration": int(q_start_eval),
                    "q_index": 1,
                    "q_batch": None if self.q_batch is None else int(self.q_batch),
                    "batch_index": int(acquisition_group_index),
                    "batch_size": None if self.batch_size is None else int(self.batch_size),
                    "acquisition_batch_size": 1,
                    "selection_batch_size": (
                        None
                        if self.selection_batch_size is None
                        else int(self.selection_batch_size)
                    ),
                    "acquisition_batch_mode": self.acquisition_batch_mode,
                    "optimizer_point_batch_size": 1,
                    "t_seq": int(self.t_seq),
                    "sequence_index": int(sequence_index),
                    "selected_node": selected_node.idx,
                    "selected_node_label": self._node_label(selected_node),
                    "selected_node_split_type": str(getattr(selected_node, "split_type", "")),
                    "selected_dims": list(selected_node.dims),
                    "D_h_i": selected_node.D_hi,
                    "Dim_h_i": selected_node.Dim_hi,
                    "node_depth": selected_node.h,
                    "node_diameter": float(selected_node.diameter),
                    "selected_gp_kernel": str(selected_node.gp_kernel_name),
                    "selected_gp_kernel_score": float(selected_node.gp_kernel_score),
                    "node_h": int(selected_node.h),
                    "node_i": int(selected_node.idx),
                    "local_n": int(selected_node.n),
                    "node_ave_diff_score": (
                        None
                        if not hasattr(selected_node, "ave_diff_score")
                        else float(getattr(selected_node, "ave_diff_score"))
                    ),
                    "node_mean_abs_best_diff": (
                        None
                        if not hasattr(selected_node, "mean_abs_best_diff")
                        else float(getattr(selected_node, "mean_abs_best_diff"))
                    ),
                    "node_risk_bonus": (
                        None
                        if not hasattr(selected_node, "risk_bonus")
                        else float(getattr(selected_node, "risk_bonus"))
                    ),
                    "node_score_risk_tolerance": (
                        None
                        if not hasattr(selected_node, "score_risk_tolerance")
                        else float(getattr(selected_node, "score_risk_tolerance"))
                    ),
                    "node_improvement_score": (
                        None
                        if not hasattr(selected_node, "improvement_score")
                        else float(getattr(selected_node, "improvement_score"))
                    ),
                    "node_imp_uncertainty": (
                        None
                        if not hasattr(selected_node, "improvement_uncertainty")
                        else float(getattr(selected_node, "improvement_uncertainty"))
                    ),
                    "node_imp_bound_score": (
                        None
                        if not hasattr(selected_node, "improvement_bound_score")
                        else float(getattr(selected_node, "improvement_bound_score"))
                    ),
                    "selected_acq_dims": (
                        None
                        if not hasattr(self, "_imp_eval_dims_by_index")
                        else list(getattr(self, "_imp_eval_dims_by_index", {}).get(int(new_idx), []))
                    ),
                    "dimension_improvement_value": (
                        None
                        if not hasattr(self, "_imp_eval_value_by_index")
                        else getattr(self, "_imp_eval_value_by_index", {}).get(int(new_idx))
                    ),
                    "acquisition": (
                        self._node_proposal_strategy(selected_node)
                        if hasattr(self, "_node_proposal_strategy")
                        else self.acquisition
                    ),
                    "joint_q_acquisition": False,
                    "proposal_strategy": (
                        self._node_proposal_strategy(selected_node)
                        if hasattr(self, "_node_proposal_strategy")
                        else self.acquisition
                    ),
                    "node_backend": (
                        self._node_backend_for_history(selected_node)
                        if hasattr(self, "_node_backend_for_history")
                        else "ei"
                    ),
                    "acq_value": float(acq_value),
                    "y_true": float(y_true) if self.experiment_mode else None,
                    "y_obs": float(y_obs),
                    "best_y": best_value,
                    "best_y_source": "running_best_observed_y",
                    "best_true_y": float(self.true_y[best_idx]) if self.experiment_mode else None,
                    "stop_prob": float(stop_prob),
                    "stop_mu": float(stop_mu),
                    "stop_std": float(stop_std),
                    "stop_index": None if stop_index is None else int(stop_index),
                    "batch_endpoint": True,
                    "threshold_entry": threshold_entry,
                    "elapsed_sec": time.time() - start,
                    "num_nodes": len(self._all_nodes),
                    "num_leaves": len(self._leaves()),
                    "split_after_eval": False,
                    "update_after_all_q": False,
                    "retrained_after_eval": False,
                    "gp_refit_after_t_seq_step": False,
                }
                self.history.append(rec)

                self._print_iteration_summary(
                    iteration=t_eval,
                    best_value=best_value,
                    current_value=float(y_obs),
                    selected_node=selected_node,
                    threshold_entry=threshold_entry,
                    selected_dims_for_display=rec.get("selected_acq_dims") or rec.get("selected_dims"),
                )
                self._plot_group_tree(selected=selected_node, iteration=t_eval)

            if stopped:
                break
            completed_merge_own_trial = bool(
                getattr(selected_node, "sad_requires_own_trial", False)
            )
            completed_restart_own_trial = bool(
                getattr(selected_node, "restart_requires_own_trial", False)
            )
            if completed_merge_own_trial:
                selected_node.sad_requires_own_trial = False
            if completed_restart_own_trial:
                selected_node.restart_requires_own_trial = False
            if self._record_node_selection_outcome(
                best_observed_before=selection_best_observed_before,
                selected_node=selected_node,
            ):
                continue
            if (
                completed_merge_own_trial
                and bool(getattr(selected_node, "sad_final_terminal_split", False))
            ):
                selected_node.sad_no_merge_branch = True
                selected_node.sad_children_terminal_after_trial = True
            selected_node.split_ready = self._should_split(selected_node)
            if selected_node.split_ready:
                did_split = split(self, selected_node)
                if did_split:
                    if completed_restart_own_trial and self.tree_restart_history:
                        restart_record = self.tree_restart_history[-1]
                        restart_record["split_after_own_trial_iteration"] = int(len(self.y))
                        restart_record["replacement_child_nodes"] = [
                            int(getattr(child, "idx", -1))
                            for child in selected_node.children
                        ]
                        restart_record["next_node_idx"] = int(self._next_node_idx)
                    if completed_merge_own_trial and self.sad_merge_history:
                        for merge_record in reversed(self.sad_merge_history):
                            if str(merge_record.get("merged_node")) == self._node_label(selected_node):
                                merge_record["split_after_own_trial_iteration"] = int(len(self.y))
                                merge_record["merged_child_nodes"] = [
                                    self._node_label(child)
                                    for child in selected_node.children
                                ]
                                break
                    if self._uses_cached_ei_selection():
                        self._refresh_leaf_cached_ei()
                    if self.history:
                        self.history[-1]["split_after_eval"] = True
                    self._print_event(
                        "[AGVS][seq-split] "
                        f"after_iteration={len(self.y)} node={self._node_label(selected_node)} "
                        f"q_batch={self.q_batch} batch_size={self.batch_size} t_seq={self.t_seq}"
                    )

        X_arr = np.asarray(self.X)
        y_arr = np.asarray(self.y)
        true_arr = np.asarray(self.true_y)
        best_observed_idx = int(np.argmax(y_arr))

        counts = np.zeros(self.D, dtype=int)
        for entry in self.history:
            for dim in entry.get("selected_dims", []):
                if 0 <= int(dim) < self.D:
                    counts[int(dim)] += 1

        top_indices = np.argsort(-counts)
        important_dim = top_indices[: int(self.eff_dim)].tolist()
        posterior_by_n = {
            int(entry["n"]): float(entry["posterior_mean_at_best"])
            for entry in self.threshold_trace
            if isinstance(entry, dict)
            and entry.get("n") is not None
            and entry.get("posterior_mean_at_best") is not None
        }
        posterior_history = [posterior_by_n.get(i) for i in range(1, len(y_arr) + 1)]
        if stopped:
            stop_records = [
                entry for entry in self.history
                if isinstance(entry, dict)
                and np.isfinite(float(entry.get("stop_mu", float("nan"))))
                and self._early_stop_triggered(float(entry.get("stop_prob", float("nan"))))
            ]
            if not stop_records:
                raise RuntimeError("Early stopping triggered without a finite paired stop record.")
            stop_record = stop_records[-1]
            current_index = stop_record.get("stop_index")
            if current_index is None or not 0 <= int(current_index) < len(X_arr):
                raise RuntimeError(
                    "Early stopping produced an invalid triggering candidate index; "
                    f"stop_index={current_index!r}, n_evaluations={len(X_arr)}."
                )
            threshold_entry = stop_record.get("threshold_entry")
            if isinstance(threshold_entry, dict):
                candidate_indices = threshold_entry.get("posterior_candidate_indices")
                if candidate_indices is not None and int(current_index) not in {
                    int(index) for index in candidate_indices
                }:
                    raise RuntimeError(
                        "Early stopping candidate is not in the evaluated posterior batch; "
                        f"stop_index={int(current_index)}."
                    )
            returned_idx = int(current_index)
            returned_mean = float(stop_record["stop_mu"])
        else:
            best_posterior = self._best_posterior_record()
            if best_posterior is not None:
                returned_idx = int(best_posterior["evaluation_index"])
                returned_mean = float(best_posterior["mu"])
            else:
                returned_idx, returned_mean = best_observed_idx, float("nan")

        if stopped and stop_iteration is not None and 1 <= int(stop_iteration) <= len(posterior_history):
            posterior_history[int(stop_iteration) - 1] = float(returned_mean)

        best_pure_x_coordinate = X_arr[returned_idx].copy()[important_dim]

        result = AGVSResult(
            best_x=X_arr[returned_idx].copy(),
            best_y=float(returned_mean),
            best_observed_y=float(y_arr[best_observed_idx]),
            X=X_arr,
            y=y_arr,
            true_y=true_arr,
            stopped=stopped,
            stop_iteration=stop_iteration,
            history=self.history,
            root=self.root,
            threshold_trace=self.threshold_trace,
            important_dim=important_dim,
            best_pure_x_coordinate=best_pure_x_coordinate,
            returned_posterior_mean=float(returned_mean),
            posterior_mean_history=posterior_history,
            best_true_y=float(true_arr[best_observed_idx]),
            posterior_dataset=self._posterior_dataset_records(),
        )
        result.current_x = X_arr[returned_idx].copy() if len(X_arr) > 0 else None
        result.current_y = float(returned_mean)
        return result

    # ------------------------------------------------------------------
    # Evaluation / initialization
    # ------------------------------------------------------------------
    def _evaluate(self, x: Array) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        true_history_before = None
        if hasattr(self.objective, "true_history"):
            try:
                true_history_before = len(getattr(self.objective, "true_history"))
            except Exception:
                true_history_before = None
        y_value = float(self.objective(x))

        if (
            true_history_before is not None
            and hasattr(self.objective, "true_history")
            and hasattr(self.objective, "observed_history")
            and len(getattr(self.objective, "true_history")) > true_history_before
        ):
            y_true = float(getattr(self.objective, "true_history")[-1])
            y_obs = float(getattr(self.objective, "observed_history")[-1])
        elif self.experiment_mode:
            raise RuntimeError(
                "AGVS experiment_mode=True requires an objective/recorder that records both "
                "true_history and observed_history. AGVS never generates observation noise internally."
            )
        else:
            y_obs = y_value
            y_true = y_obs
        self.X.append(x.copy())
        self.y.append(y_obs)
        self.true_y.append(y_true)
        return y_true, y_obs

    def _initial_design(self) -> None:
        n_design = min(self.n_init, self.max_iter)
        if len(self.y) >= n_design:
            return

        n_needed = n_design - len(self.y)
        if self.initial_points is not None and len(self.initial_points) > len(self.y):
            X_init = self.initial_points[len(self.y): min(len(self.initial_points), n_design)]
            if len(X_init) < n_needed:
                extra = sample_initial_points(
                    self.bounds,
                    n_points=n_needed - len(X_init),
                    rng=self.rng,
                    strategy=self.initial_sampling_strategy,
                )
                X_init = np.vstack([X_init, extra])
        else:
            X_init = sample_initial_points(
                self.bounds,
                n_points=n_needed,
                rng=self.rng,
                strategy=self.initial_sampling_strategy,
            )
        for x in X_init:
            y_true, y_obs = self._evaluate(x)
            init_idx = len(self.y) - 1
            self.root.local_indices.append(init_idx)
            self.root.score_indices.append(init_idx)
            self.root.y_best = max(float(self.root.y_best), float(y_obs))
            self._print_event(
                "[AGVS][init] "
                f"t={len(self.y)}/{self.max_iter} node={self._node_label(self.root)} idx={self.root.idx} "
                f"D_h_i={self.root.D_hi} Dim_h_i={self.root.Dim_hi} "
                f"y_true={float(y_true):.6g} y_obs={float(y_obs):.6g}"
            )

    # ------------------------------------------------------------------
    # GP helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _node_gp_fit_signature(node: Node) -> Tuple:
        """Identify the exact node-local dataset used by a GP fit.

        Global evaluation count is intentionally absent: sibling evaluations
        require a refit only when added to this node's training indices, as
        happens before a target-dimension retained node's next visit.
        """
        local_indices = getattr(node, "local_indices", []) or []
        if isinstance(local_indices, SharedObservationIndices):
            if local_indices.shared_indices is not None:
                index_signature = (
                    "shared_subset",
                    local_indices.shared_indices,
                    tuple(int(index) for index in local_indices._extra),
                )
            else:
                index_signature = (
                    "shared",
                    int(local_indices.prefix_start),
                    int(local_indices.prefix_length),
                    tuple(int(index) for index in local_indices._extra),
                )
        else:
            index_signature = tuple(int(idx) for idx in local_indices)
        return (tuple(int(dim) for dim in getattr(node, "dims", []) or []), index_signature)

    @staticmethod
    def _invalidate_node_gp(node: Node) -> None:
        node.model = None
        node._gp_fit_signature = None
        node.gp_kernel_name = "none"
        node.gp_kernel_score = -np.inf
        node.lengthscales = None
        node.lengthscale_gp_kernel_name = "none"
        node.lengthscale_gp_kernel_score = -np.inf
        for attribute in (
            "_posterior_mean_cache",
            "_posterior_mean_cache_fit_signature",
            "_posterior_mean_cache_model_id",
        ):
            if hasattr(node, attribute):
                delattr(node, attribute)

    def _fit_local_gp(self, node: Node) -> Optional[GaussianProcessRegressor]:
        fit_signature = self._node_gp_fit_signature(node)
        cached_model = getattr(node, "model", None)
        if cached_model is not None and getattr(node, "_gp_fit_signature", None) == fit_signature:
            return cached_model

        if (
            cached_model is not None
            and bool(getattr(node, "gp_hold_hyperparameters", False))
            and list(getattr(node, "dims", []) or []) == list(range(self.D))
        ):
            node.gp_incremental_source_model = cached_model

        self._invalidate_node_gp(node)
        try:
            model = self.gp_manager.fit_node_gp(
                node,
                self.X,
                self.y,
            )
        except Exception as exc:
            self._last_gp_fit_error = exc
            node._last_gp_fit_error = exc
            self._invalidate_node_gp(node)
            return None
        if model is None:
            self._last_gp_fit_error = RuntimeError(
                "GP manager returned no model without raising an exception."
            )
            node._last_gp_fit_error = self._last_gp_fit_error
            self._invalidate_node_gp(node)
            return None

        self._last_gp_fit_error = None
        node._last_gp_fit_error = None

        node.model = model
        node._gp_fit_signature = fit_signature
        if list(getattr(node, "dims", []) or []) == list(range(self.D)):
            self._last_full_dim_gp_model = model
        try:
            self._record_gp_fit(node)
            self._update_posterior_dataset_from_node(node, model)
        except Exception:
            # Diagnostics must never turn a successfully fitted current model
            # into an apparent refit failure.
            pass
        return model

    def _refit_after_t_seq_step(
        self,
        node: Node,
        *,
        sequence_index: int,
    ) -> GaussianProcessRegressor:
        """Require a current GP after every completed t_seq point-batch step."""
        model = self._fit_local_gp(node)
        if model is None or not self._node_gp_is_current(node):
            raise RuntimeError(
                "AG-VS could not refit a current node GP after "
                f"t_seq step {int(sequence_index)}."
            )
        node.last_t_seq_gp_refit_step = int(sequence_index)
        node.last_t_seq_gp_refit_after_iteration = int(len(self.y))
        return model

    def _node_gp_is_current(self, node: Node) -> bool:
        """Return whether the model was fitted to this exact node-local dataset."""
        if getattr(node, "model", None) is None:
            return False
        expected_signature = self._node_gp_fit_signature(node)
        if getattr(node, "_gp_fit_signature", None) == expected_signature:
            return True
        return False

    def _record_gp_fit(self, node: Node) -> None:
        dims = [int(dim) for dim in getattr(node, "dims", [])]
        lengthscales = getattr(node, "lengthscales", None)
        if lengthscales is None or len(dims) == 0:
            return
        lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
        if lengthscales.size != len(dims):
            return

        finite_pairs = [
            (int(dim), float(ls))
            for dim, ls in zip(dims, lengthscales)
            if np.isfinite(float(ls))
        ]
        if not finite_pairs:
            return

        isotropic_lengthscale = float(finite_pairs[0][1])

        self._gp_fit_counter += 1
        node._last_gp_fit_id = int(self._gp_fit_counter)
        self.gp_fit_history.append(
            {
                "fit_id": int(self._gp_fit_counter),
                "after_iteration": int(len(self.y)),
                "node": self._node_label(node),
                "node_idx": int(getattr(node, "idx", -1)),
                "node_depth": int(getattr(node, "h", 0)),
                "local_n": int(getattr(node, "n", len(getattr(node, "local_indices", [])))),
                "gp_fit_mode": str(getattr(node, "gp_fit_mode", "full_hyperparameter_fit")),
                "gp_cholesky_jitter": float(
                    getattr(node, "gp_cholesky_jitter", 0.0)
                ),
                "Dim_h_i": int(len(dims)),
                "kernel": str(getattr(node, "gp_kernel_name", "none")),
                "kernel_score": float(getattr(node, "gp_kernel_score", float("nan"))),
                "lengthscale_kernel": str(
                    getattr(node, "lengthscale_gp_kernel_name", "none")
                ),
                "lengthscale_kernel_score": float(
                    getattr(node, "lengthscale_gp_kernel_score", float("nan"))
                ),
                "dims": list(dims),
                "lengthscales": [float(value) for value in lengthscales.tolist()],
                "isotropic_lengthscale": isotropic_lengthscale,
                "active_lengthscale_ranks": [],
            }
        )

    def _posterior_x_key(self, x: Array) -> Tuple[int, ...]:
        x = np.asarray(x, dtype=float).reshape(-1)
        widths = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1.0)
        normalized = (x - self.bounds[:, 0]) / widths
        return tuple(np.rint(normalized / float(self.posterior_x_tolerance)).astype(np.int64).tolist())

    def _update_posterior_dataset_from_node(
        self,
        node: Node,
        model: GaussianProcessRegressor,
    ) -> None:
        """Keep only the largest local-GP mean for each x in global dataset D."""
        valid_indices = [
            int(idx) for idx in getattr(node, "local_indices", [])
            if 0 <= int(idx) < len(self.X)
        ]
        dims = [int(dim) for dim in getattr(node, "dims", [])]
        if not valid_indices or not dims:
            return
        self._posterior_gp_fit_counter += 1
        posterior_fit_id = int(self._posterior_gp_fit_counter)
        X_full = np.asarray(self.X, dtype=float)[valid_indices]
        try:
            mu_values = np.asarray(model.predict(X_full[:, dims]), dtype=float).reshape(-1)
        except Exception:
            return
        # The same fitted model's score update consumes a subset of these
        # means. Cache them by global evaluation index so it does not issue an
        # identical second posterior prediction.
        node._posterior_mean_cache = {
            int(evaluation_index): float(mu)
            for evaluation_index, mu in zip(valid_indices, mu_values)
        }
        node._posterior_mean_cache_fit_signature = getattr(
            node,
            "_gp_fit_signature",
            None,
        )
        node._posterior_mean_cache_model_id = int(id(model))
        for evaluation_index, x, mu in zip(valid_indices, X_full, mu_values):
            if not np.isfinite(float(mu)):
                continue
            key = self._posterior_x_key(x)
            previous = self.global_dataset.get(key)
            if previous is not None and float(previous["mu"]) >= float(mu):
                continue
            record = {
                "x": np.asarray(x, dtype=float).copy(),
                "mu": float(mu),
                "evaluation_index": int(evaluation_index),
                "node_idx": int(getattr(node, "idx", -1)),
                "node_depth": int(getattr(node, "h", 0)),
                "node_dims": list(dims),
                "gp_fit_id": posterior_fit_id,
                "after_iteration": int(len(self.y)),
            }
            self.global_dataset[key] = record
            if self.global_best_record is None or float(mu) > float(self.global_best_record["mu"]):
                self.global_best_record = record

    def _posterior_dataset_records(self) -> List[Dict]:
        return list(self.global_dataset.values())

    def _global_mu_best(self) -> float:
        """Return pseudo-code mu_best immediately before a new evaluation."""
        record = self.global_best_record
        return float(record["mu"]) if record is not None else float("-inf")

    def _global_observed_best(self) -> float:
        """Return the best finite observation available before a new evaluation."""
        if not self.y:
            return float("-inf")
        values = np.asarray(self.y, dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else float("-inf")

    def _record_node_selection_outcome(
        self,
        *,
        best_observed_before: float,
        selected_node: Node,
    ) -> bool:
        """Update N_bad patience after one complete node-selection block.

        One selection comprises traversal plus every acquisition/evaluation in
        its t_seq block. Point batches therefore count as one node selection,
        rather than one failure per evaluated point.
        """
        if not self.N_bad_enabled:
            self.consecutive_bad_node_selections = 0
            return False

        best_observed_after = self._global_observed_best()
        improved = bool(float(best_observed_after) > float(best_observed_before))
        if improved:
            self.consecutive_bad_node_selections = 0
            return False

        self.consecutive_bad_node_selections += 1
        if self.consecutive_bad_node_selections < self.N_bad:
            return False

        # A root (including a freshly reinitialized root) is scored before its
        # children are created. With N_bad=1, the patience threshold can
        # therefore be reached while there is not yet any child-state tree to
        # rebuild. Keep the failed node visit counted and defer the restart
        # until an active child leaf exists; the normal split immediately after
        # this method then creates that state.
        has_active_child_leaf = any(
            getattr(node, "parent", None) is not None
            and bool(getattr(node, "is_leaf", False))
            and not bool(getattr(node, "split_frozen", False))
            and not bool(getattr(node, "plot_archived", False))
            for node in self._all_nodes
        )
        if not has_active_child_leaf:
            return False

        self._restart_tree_with_shared_data(
            selected_node=selected_node,
            best_observed_before=float(best_observed_before),
            best_observed_after=float(best_observed_after),
        )
        return True

    def _restart_child_snapshot(
        self,
    ) -> Tuple[List[Node], List[int], Dict[str, int]]:
        """Collect deduplicated active-child training data for a replacement GP."""
        source_children = [
            node
            for node in self._all_nodes
            if getattr(node, "parent", None) is not None
            and bool(getattr(node, "is_leaf", False))
            and not bool(getattr(node, "split_frozen", False))
            and not bool(getattr(node, "plot_archived", False))
        ]
        if not source_children:
            raise RuntimeError(
                "AG-VS cannot reinitialize from child state because the old tree "
                "has no active child leaves."
            )

        # Reinitialization inherits the actual training views of the active
        # child leaves. Repeated references are harmless: they are collapsed
        # below before exact (x, observed-y) pairs are deduplicated.
        child_index_references = [
            int(index)
            for child in source_children
            for index in list(getattr(child, "local_indices", []) or [])
            if 0 <= int(index) < min(len(self.X), len(self.y))
        ]
        unique_child_indices = sorted(set(child_index_references))
        if not unique_child_indices:
            raise RuntimeError(
                "AG-VS cannot reinitialize from child state because active child "
                "leaves have no training observations."
            )

        # Keep the earliest representative when every x coordinate and the
        # observed y differ by at most restart_pair_epsilon. If either x or y
        # exceeds that tolerance, the observation remains distinct. Global
        # X/y histories stay immutable; only the replacement-root training view
        # is deduplicated.
        epsilon = float(self.restart_pair_epsilon)
        retained_x: List[np.ndarray] = []
        retained_y: List[float] = []
        child_owned_indices: List[int] = []
        for index in unique_child_indices:
            current_x = np.asarray(self.X[index], dtype=float).reshape(-1)
            current_y = float(self.y[index])
            is_duplicate = any(
                abs(current_y - previous_y) <= epsilon
                and bool(np.all(np.abs(current_x - previous_x) <= epsilon))
                for previous_x, previous_y in zip(retained_x, retained_y)
            )
            if is_duplicate:
                continue
            child_owned_indices.append(int(index))
            retained_x.append(current_x.copy())
            retained_y.append(current_y)
        deduplication_stats = {
            "child_index_reference_count": int(len(child_index_references)),
            "unique_child_evaluation_count_before_x_y_dedup": int(
                len(unique_child_indices)
            ),
            "duplicate_child_index_references_removed": int(
                len(child_index_references) - len(unique_child_indices)
            ),
            "duplicate_x_y_observations_removed": int(
                len(unique_child_indices) - len(child_owned_indices)
            ),
        }

        return (
            source_children,
            child_owned_indices,
            deduplication_stats,
        )

    def _restart_tree_with_shared_data(
        self,
        *,
        selected_node: Node,
        best_observed_before: float,
        best_observed_after: float,
    ) -> None:
        """Rebuild a root that earns its own trial before random splitting."""
        source_children, child_owned_indices, deduplication_stats = (
            self._restart_child_snapshot()
        )
        old_root = self.root
        old_node_count = len(self._all_nodes)
        new_root = Node(
            dims=list(range(self.D)),
            h=0,
            idx=0,
            parent=None,
        )
        self._next_node_idx = 1
        new_root.local_bounds = self.bounds.copy()
        new_root.local_indices = SharedObservationIndices(
            self.y,
            indices=child_owned_indices,
        )
        # A replacement root inherits training data, not score history. It
        # starts unscored and must complete its own visit before being split.
        new_root.score_indices = []
        new_root.node_score_history = []
        new_root.ave_diff_score = float("inf")
        new_root.mean_abs_best_diff = float("inf")
        new_root.selection_score = float("-inf")
        new_root.score_value_source = "awaiting_reinitialized_root_own_visit"
        new_root.y_best = float(
            np.max(np.asarray([self.y[index] for index in child_owned_indices], dtype=float))
        )
        new_root.diameter = self._node_diameter(new_root)
        new_root.restart_requires_own_trial = True
        new_root.sad_no_merge_branch = not self.merge_enabled
        new_root.merge_disabled = int(not self.merge_enabled)
        root_warm_start_requested = bool(
            self.warm_start_root_gp and self._last_full_dim_gp_model is not None
        )
        if root_warm_start_requested:
            new_root.gp_warm_start_model = self._last_full_dim_gp_model

        self.root = new_root
        self.roots = [new_root]
        self._all_nodes = [new_root]
        self._plot_nodes = [new_root]
        # Future observations repopulate ownership for the replacement tree.
        self._source_leaf_by_index = {}
        self.consecutive_bad_node_selections = 0
        self.tree_restart_count += 1
        update(self, new_root)

        restart_record = {
            "restart_number": int(self.tree_restart_count),
            "after_iteration": int(len(self.y)),
            "N_bad": int(self.N_bad),
            "selected_node": int(getattr(selected_node, "idx", -1)),
            "old_root": int(getattr(old_root, "idx", -1)),
            "new_root": int(new_root.idx),
            "node_index_reset": True,
            "old_node_count": int(old_node_count),
            "transferred_evaluation_count": int(len(new_root.local_indices)),
            "excluded_parent_evaluation_count": int(
                len(self.y)
                - deduplication_stats["unique_child_evaluation_count_before_x_y_dedup"]
            ),
            "duplicate_child_index_references_removed": int(
                deduplication_stats["duplicate_child_index_references_removed"]
            ),
            "duplicate_x_y_observations_removed": int(
                deduplication_stats["duplicate_x_y_observations_removed"]
            ),
            "restart_deduplication_policy": (
                "unique evaluation indices, then earliest representative when "
                "every abs(delta-x) <= 1e-10 and abs(delta-observed-y) <= 1e-10"
            ),
            "restart_pair_epsilon": float(self.restart_pair_epsilon),
            "observation_storage": "shared_active_child_training_union",
            "restart_data_source": "active_leaf_local_indices_union",
            "restart_split_source": "seeded_optimizer_rng_random_permutation",
            "restart_split_policy": (
                "fit when the replacement root is selected; evaluate its t_seq point "
                "batches without a trailing refit, then randomly permute and group dimensions"
            ),
            "required_own_trial_evaluations": int(
                self.t_seq * self.selection_batch_size
            ),
            "source_child_count": int(len(source_children)),
            "source_child_nodes": [
                int(getattr(child, "idx", -1)) for child in source_children
            ],
            "root_gp_warm_start_requested": bool(root_warm_start_requested),
            "root_gp_incremental_requested": False,
            "root_gp_fit_mode": str(
                getattr(new_root, "gp_fit_mode", "deferred_until_selected")
            ),
            "root_direct_split": False,
            "replacement_child_nodes": [],
            "next_node_idx": int(self._next_node_idx),
            "best_observed_before": float(best_observed_before),
            "best_observed_after": float(best_observed_after),
        }
        self.tree_restart_history.append(restart_record)
        if self.history:
            self.history[-1]["tree_restarted_after_selection"] = True
            self.history[-1]["tree_restart_number"] = int(self.tree_restart_count)

        self._print_event(
            "[AGVS][tree-restart] "
            f"restart={self.tree_restart_count} after_iteration={len(self.y)} "
            f"N_bad={self.N_bad} old_nodes={old_node_count} "
            f"new_root={self._node_label(new_root)} transferred={len(new_root.local_indices)} "
            f"source=active-leaf-only own_trial_evaluations="
            f"{self.t_seq * self.selection_batch_size} "
            f"best_observed={best_observed_after:.6g}"
        )
        restart_record["root_gp_incremental_added"] = 0
        self._plot_group_tree(selected=new_root, iteration=len(self.y))

    def _best_posterior_record(self, allowed_indices: Optional[List[int]] = None) -> Optional[Dict]:
        if allowed_indices is None:
            return self.global_best_record
        records = self._posterior_dataset_records()
        allowed = {int(idx) for idx in allowed_indices}
        records = [record for record in records if int(record["evaluation_index"]) in allowed]
        if not records:
            return None
        return max(records, key=lambda record: float(record["mu"]))

    # ------------------------------------------------------------------
    # Geometry and tree
    # ------------------------------------------------------------------
    def _node_bounds(self, node: Node) -> Array:
        if len(node.dims) == 0:
            return np.empty((0, 2), dtype=float)
        if node.local_bounds is None:
            return self.bounds[node.dims].copy()
        return np.asarray(node.local_bounds, dtype=float).copy()

    def _node_diameter(self, node: Node) -> float:
        bnds = self._node_bounds(node)
        if bnds.size == 0:
            return 0.0
        widths = bnds[:, 1] - bnds[:, 0]
        return float(np.linalg.norm(widths, ord=2))

    def _update_node_geometry(self, node: Node) -> None:
        node.diameter = self._node_diameter(node)

    def _leaves(self) -> List[Node]:
        return [n for n in self._all_nodes if n.is_leaf]

    def _assign_new_index_to_leaf_tree(self, new_idx: int, leaf: Node) -> None:
        new_idx = int(new_idx)

        # Once this node is selected/evaluated, its next update fits its own
        # local isotropic GP.
        leaf.selected_count += 1
        leaf.initial_selection_priority = False

        current: Optional[Node] = leaf
        while current is not None:
            if new_idx not in current.local_indices:
                current.local_indices.append(new_idx)
            if new_idx not in current.score_indices:
                current.score_indices.append(new_idx)
            current = current.parent

    def _inherited_local_indices(self, node: Node):
        local_indices = getattr(node, "local_indices", []) or []
        if isinstance(local_indices, SharedObservationIndices):
            return local_indices.frozen()
        return list(local_indices)

    # ------------------------------------------------------------------
    # Selection scores
    # ------------------------------------------------------------------
    def _update_node_split_stats(self, node: Node) -> None:
        self._update_node_geometry(node)

    def _update_leaf_stats(self, node: Node) -> None:
        self._update_node_split_stats(node)

    def _update_path_scores_bottom_up(self, leaf: Node) -> None:
        path: List[Node] = []
        node: Optional[Node] = leaf
        while node is not None:
            path.append(node)
            node = node.parent

        for node in path:
            self._update_node_split_stats(node)

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------
    def _uses_cached_ei_selection(self) -> bool:
        return self.__class__ is AGVSBaseOptimizer

    def _refresh_node_cached_ei(self, node: Node) -> Tuple[Array, float]:
        if not self._uses_cached_ei_selection():
            return np.asarray([], dtype=float), -np.inf
        try:
            x_next, acq_value = self._propose_from_leaf(node)
            node.cached_ei_x = np.asarray(x_next, dtype=float).copy()
            node.cached_ei_value = float(acq_value)
            node.selection_score = -float(acq_value)
            node.gp_best_mean = float(acq_value)
            return node.cached_ei_x.copy(), float(node.cached_ei_value)
        except Exception:
            node.cached_ei_x = None
            node.cached_ei_value = -np.inf
            node.selection_score = np.inf
            node.gp_best_mean = -np.inf
            return np.asarray([], dtype=float), -np.inf

    def _refresh_leaf_cached_ei(self) -> None:
        if not self._uses_cached_ei_selection():
            return
        for leaf in self._leaves():
            if getattr(leaf, "cached_ei_x", None) is None or not np.isfinite(
                float(getattr(leaf, "cached_ei_value", -np.inf))
            ):
                self._fit_local_gp(leaf)
                self._refresh_node_cached_ei(leaf)

    def _select_node(self) -> Node:
        if not self.roots:
            raise RuntimeError("No root nodes available for selection.")

        if self._uses_cached_ei_selection():
            self._refresh_leaf_cached_ei()
            leaves = list(self._leaves())
            if not leaves:
                raise RuntimeError("No leaf nodes available for EI selection.")
            selected = min(
                leaves,
                key=lambda node: (
                    -float(getattr(node, "cached_ei_value", -np.inf)),
                    int(node.h),
                    int(node.idx),
                ),
            )
            selected.selection_score = -float(getattr(selected, "cached_ei_value", -np.inf))
            selected.gp_best_mean = float(getattr(selected, "cached_ei_value", -np.inf))
            return selected

        raise NotImplementedError("AG-VS subclasses must implement node selection.")

    def _prepare_selected_node_for_evaluation(self, selected_node: Node) -> Node:
        # The retained node uses all observations at d_target, including those
        # collected on excluded branches since its previous visit. Training
        # inheritance must not create direct score evidence for this node.
        if (
            bool(getattr(selected_node, "sad_merged_node", False))
            and len(selected_node.dims) == self.eff_dim
            and not bool(getattr(selected_node, "sad_no_merge_branch", False))
            and not bool(getattr(selected_node, "exclusion_state", 0))
        ):
            selected_node.node_dataset = range(len(self.y))
        return selected_node

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def _sample_acquisition_candidates(self, lows: Array, highs: Array, n: int) -> Array:
        return sample_acquisition_candidates(
            lows,
            highs,
            n,
            rng=self.rng,
            strategy=self.acquisition_sampling_strategy,
        )

    def _evaluation_batch_size(self, leaf: Node, remaining_budget: int) -> int:
        """Number of objective points evaluated for the selected leaf.

        q_batch is the joint-acquisition size; batch_size is the independent
        pointwise-acquisition size. Exactly one is configured.
        """

        del leaf
        point_batch_size = (
            self.q_batch if self.q_batch is not None else self.batch_size
        )
        return min(int(point_batch_size), max(0, int(remaining_budget)))

    def _base_fill_for_leaf(self, leaf: Node) -> Array:
        """Complete inactive coordinates from the global best finite observation."""
        del leaf
        values = np.asarray(self.y, dtype=float)
        finite_indices = np.flatnonzero(np.isfinite(values))
        if finite_indices.size:
            best_index = int(finite_indices[np.argmax(values[finite_indices])])
            return np.asarray(self.X[best_index], dtype=float).copy()
        return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _propose_q_from_leaf(self, leaf: Node, q: int) -> List[Tuple[Array, float]]:
        if self.q_batch is not None:
            return propose_vanilla_joint_q(self, leaf, q)
        return propose_vanilla_independent_batch(self, leaf, q)

    def _propose_from_leaf(self, leaf: Node) -> Tuple[Array, float]:
        return propose_vanilla_one(self, leaf)

    # ------------------------------------------------------------------
    # Posterior stopping
    # ------------------------------------------------------------------
    def _should_record_posterior_for_node(self, node: Node) -> bool:
        del node
        return False

    def _posterior_stop_probability(
        self,
        node: Node,
        candidate_indices: Optional[List[int]] = None,
    ) -> Tuple[float, float, float, Optional[int]]:
        if not self._node_gp_is_current(node):
            return 0.0, float("nan"), float("nan"), None
        candidate_indices = self._node_posterior_history_indices(node, candidate_indices)
        return self.early_stop.stop_probability(
            node=node,
            X=self.X,
            y=self.y,
            candidate_indices=candidate_indices,
        )

    def _record_stoppable_thresholds(
        self,
        node: Node,
        candidate_indices: Optional[List[int]] = None,
    ) -> Optional[Dict]:
        if not self._node_gp_is_current(node):
            return None
        candidate_indices = self._node_posterior_history_indices(node, candidate_indices)
        return self.early_stop.record_thresholds(
            node=node,
            X=self.X,
            y=self.y,
            candidate_indices=candidate_indices,
        )

    def _node_posterior_history_indices(
        self,
        node: Node,
        candidate_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """Query the current node GP only at points tried in the current iteration."""
        del node
        return sorted({
            int(idx)
            for idx in list(candidate_indices or [])
            if 0 <= int(idx) < len(self.X)
        })


class AGVSRegroupingOptimizer(AGVSBaseOptimizer):
    """AGVS regrouping variant.

    Every splittable selected node is split into up to N_split random
    coordinate groups without waiting for the usual split condition. Each child is tried
    for q_batch evaluations. Exactly N_split - d_eff children are eliminated
    from every completed cohort; the remaining highest-scoring children are
    merged and split again while it has more dimensions than Eliminate. A node
    with fewer than two dimensions is never split.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("selection_mode", None)
        super().__init__(*args, **kwargs)
        self.sad_n_split = self.split_k
        self.sad_eliminate = int(self.sad_n_split - self.eff_dim)
        if self.sad_n_split < 2:
            raise ValueError("AGVS N_split must be at least 2")
        if self.sad_eliminate < 1:
            raise ValueError("AGVS Eliminate must be at least 1")
        if self.sad_eliminate >= self.sad_n_split:
            raise ValueError("AGVS Eliminate must be smaller than N_split")
        self.split_k = int(self.sad_n_split)
        self.sad_merge_history: List[Dict] = []


    def _assign_new_index_to_leaf_tree(self, new_idx: int, leaf: Node) -> None:
        super()._assign_new_index_to_leaf_tree(new_idx, leaf)
        if not hasattr(self, "_source_leaf_by_index"):
            self._source_leaf_by_index = {}
        self._source_leaf_by_index[int(new_idx)] = leaf

    def _is_descendant_node(self, node: Optional[Node], ancestor: Node) -> bool:
        ancestor_idx = int(getattr(ancestor, "idx", -1))
        current = node
        while current is not None:
            if current is ancestor or int(getattr(current, "idx", -2)) == ancestor_idx:
                return True
            current = getattr(current, "parent", None)
        return False

    def _owned_indices_for_nodes(self, nodes: List[Node], fallback_union: Optional[List[int]] = None) -> List[int]:
        owned: set[int] = set()
        source_by_index = getattr(self, "_source_leaf_by_index", {}) or {}
        for idx, source_leaf in source_by_index.items():
            idx_int = int(idx)
            if not (0 <= idx_int < len(self.y)):
                continue
            if any(self._is_descendant_node(source_leaf, node) for node in nodes):
                owned.add(idx_int)
        if owned:
            return sorted(owned)
        if fallback_union is None:
            return []
        return sorted({int(idx) for idx in fallback_union if 0 <= int(idx) < len(self.y)})

    def _is_completed_pending_cohort_child(self, node: Node) -> bool:
        parent = getattr(node, "parent", None)
        return (
            getattr(node, "sad_cohort_id", None) is not None
            and bool(getattr(node, "sad_trial_done", False))
            and parent is not None
            and not bool(getattr(parent, "sad_deferred_cohort_merged", False))
        )

    def _eligible_children(self, node: Node) -> List[Node]:
        return [
            child
            for child in list(getattr(node, "children", []) or [])
            if not bool(getattr(child, "split_frozen", False))
            and not bool(getattr(child, "plot_archived", False))
            and not self._is_completed_pending_cohort_child(child)
        ]

    def _leaves(self) -> List[Node]:
        return [
            node
            for node in self._all_nodes
            if node.is_leaf and not bool(getattr(node, "split_frozen", False))
            and not bool(getattr(node, "plot_archived", False))
            and not self._is_completed_pending_cohort_child(node)
        ]

    def _plot_node_marker(self, ax, x: float, y: float, node: Node, selected: bool, on_path: bool, has_active: bool) -> None:
        try:
            from matplotlib.patches import Circle, Wedge
        except Exception:
            return super()._plot_node_marker(ax, x, y, node, selected, on_path, has_active)

        is_done = bool(getattr(node, "sad_trial_done", False))
        edge = "#d62728" if selected else ("#ef4444" if on_path else "#1f2937")
        linewidth = 3.8 if selected else (3.0 if on_path else 1.2)
        radius = 0.98 if selected else 0.88
        green = "#22c55e"
        orange = "#f97316"
        default_face = "#fef3c7" if bool(getattr(node, "sad_merged_node", False)) else "#d8e8ff"

        if has_active and is_done:
            ax.add_patch(Wedge((x, y), radius, 90, 270, facecolor=green, edgecolor="none", zorder=3))
            ax.add_patch(Wedge((x, y), radius, -90, 90, facecolor=orange, edgecolor="none", zorder=3))
            ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor=edge, linewidth=linewidth, zorder=3.2))
        else:
            face = green if has_active else (orange if is_done else default_face)
            ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=3))

    def _plot_node_score_lines(self, node: Node) -> List[str]:
        return []

    def _visible_plot_nodes(self) -> List[Node]:
        nodes_by_idx: Dict[int, Node] = {}
        for node in list(self._all_nodes) + list(self._plot_nodes):
            if bool(getattr(node, "plot_archived", False)):
                continue
            nodes_by_idx[int(node.idx)] = node
        return sorted(nodes_by_idx.values(), key=lambda node: (int(node.h), int(node.idx)))

    def _refresh_leaf_scores(self) -> None:
        for node in getattr(self, "_all_nodes", []) or []:
            if bool(getattr(node, "split_frozen", False)) or bool(getattr(node, "plot_archived", False)):
                node.selection_score = np.inf
                continue
            if self._is_completed_pending_cohort_child(node):
                node.selection_score = np.inf
                continue
            if not bool(getattr(node, "is_leaf", False)):
                continue

            self._fit_local_gp(node)
            self._update_leaf_stats(node)

    def _select_node(self) -> Node:
        raise NotImplementedError("AG-VS node selection is implemented by AGVSOptimizer.")

    def _should_split(self, node: Node) -> bool:
        if len(getattr(node, "dims", []) or []) < 2:
            return False
        if bool(getattr(node, "sad_never_split", False)):
            return False
        if bool(getattr(node, "sad_terminal_branch", False)):
            return False
        if bool(getattr(node, "split_frozen", False)):
            return False
        if bool(getattr(node, "expanded", False)):
            return False
        if bool(getattr(node, "sad_final_terminal_split", False)):
            return True
        if (
            bool(getattr(node, "sad_merged_node", False))
            and int(self.eff_dim) < len(getattr(node, "dims", []) or []) <= int(self.sad_n_split)
        ):
            return True
        if (
            getattr(node, "sad_cohort_id", None) is not None
            and not bool(getattr(getattr(node, "parent", None), "sad_deferred_cohort_merged", False))
        ):
            return False
        groups, _eliminate_count, _has_remainder = self._planned_split_groups(list(getattr(node, "dims", [])))
        return bool(len(groups) >= 2)

    def _split_node(self, node: Node) -> bool:
        if not self._should_split(node):
            return False
        return self._split_node_by_dimensions(node)

    def _node_best_trial(self, node: Node) -> Tuple[Optional[int], float, Optional[np.ndarray]]:
        valid = [int(idx) for idx in getattr(node, "local_indices", []) if 0 <= int(idx) < len(self.X)]
        record = self._best_posterior_record(valid)
        if record is None:
            return None, -np.inf, None
        return (
            int(record["evaluation_index"]),
            float(record["mu"]),
            np.asarray(record["x"], dtype=float).copy(),
        )

    def _overlay_better_child_coordinates(self, parent: Node, base: np.ndarray, parent_best_y: float) -> None:
        for child in list(getattr(parent, "children", []) or []):
            _child_idx, child_best_y, child_best_x = self._node_best_trial(child)
            if child_best_x is not None and np.isfinite(child_best_y) and child_best_y > float(parent_best_y):
                dims = [int(dim) for dim in getattr(child, "dims", []) if 0 <= int(dim) < int(self.D)]
                if dims:
                    base[dims] = np.asarray(child_best_x, dtype=float)[dims]
            next_parent_best = child_best_y if np.isfinite(child_best_y) else parent_best_y
            self._overlay_better_child_coordinates(child, base, next_parent_best)

    def _parent_best_base(self, node: Node) -> np.ndarray:
        return super()._base_fill_for_leaf(node)

    def _base_fill_for_leaf(self, leaf: Node) -> np.ndarray:
        return self._parent_best_base(leaf)

    def _propose_from_leaf(self, leaf: Node) -> Tuple[np.ndarray, float]:
        base = self._parent_best_base(leaf)
        dims = list(getattr(leaf, "dims", []))
        if len(dims) == 0:
            return np.clip(base, self.bounds[:, 0], self.bounds[:, 1]), 0.0

        node_bounds = self._node_bounds(leaf)
        lows = node_bounds[:, 0]
        highs = node_bounds[:, 1]

        if leaf.n == 0:
            base[dims] = self.rng.uniform(lows, highs)
            return np.clip(base, self.bounds[:, 0], self.bounds[:, 1]), float("inf")

        local_model = self._fit_local_gp(leaf)
        if local_model is None:
            base[dims] = self.rng.uniform(lows, highs)
            return np.clip(base, self.bounds[:, 0], self.bounds[:, 1]), 0.0

        Z = self._sample_acquisition_candidates(lows, highs, self.candidate_count)
        acq = classical_expected_improvement(
            local_model,
            np.asarray(local_model.X_train_, dtype=float),
            Z,
        )

        j = int(np.argmax(acq))
        base[dims] = Z[j]
        return np.clip(base, self.bounds[:, 0], self.bounds[:, 1]), float(acq[j])

    def _split_node_by_dimensions(self, node: Node) -> bool:
        dims = list(getattr(node, "dims", []))
        no_merge_branch = bool(getattr(node, "sad_no_merge_branch", False))
        dims_random = [
            int(dim)
            for dim in self.rng.permutation(np.asarray(dims, dtype=int)).tolist()
        ]
        node.sad_split_permutation = list(dims_random)
        split_target_groups = getattr(node, "sad_split_target_groups", None)
        groups, eliminate_count, has_remainder = self._planned_split_groups(
            dims_random,
            target_groups=split_target_groups,
        )
        finalize_top_eff_after_cohort = bool(
            getattr(node, "sad_merged_node", False)
            and int(self.eff_dim) < len(dims_random) <= int(self.sad_n_split)
            and not bool(getattr(node, "sad_final_terminal_split", False))
        )
        if len(groups) < 2:
            return False

        terminal_children = bool(getattr(node, "sad_children_terminal_after_trial", False))
        parent_bounds = self._node_bounds(node)
        dim_to_local = {int(dim): idx for idx, dim in enumerate(node.dims)}
        # Preserve the parent's own score history at the exact moment before
        # its children are created. Future parent scores combine this frozen
        # baseline with the children's direct post-split score data.
        node.sad_pre_split_ave_diff_score = float(getattr(node, "ave_diff_score", np.inf))
        node.sad_pre_split_data_count = int(len(self._node_score_indices(node)))
        node.children = []

        cohort_id = f"{int(getattr(node, 'idx', -1))}:{len(self.sad_merge_history)}:{len(self.y)}"
        children: List[Node] = []
        for trial_order, group in enumerate(groups):
            child = Node(dims=list(group), h=int(node.h) + 1, idx=self._next_node_idx, parent=node)
            self._next_node_idx += 1
            child.split_type = "sad_random_group"
            child.local_bounds = np.asarray(
                [parent_bounds[dim_to_local[int(dim)]] for dim in group],
                dtype=float,
            )
            child.local_indices = self._inherited_local_indices(node)
            child.score_indices = []
            child.ave_diff_score = np.inf
            child.selection_score = -np.inf
            child.initial_selection_priority = False
            child.selected_count = 0
            child.sad_cohort_id = cohort_id
            child.sad_parent_idx = int(node.idx)
            child.sad_trial_order = int(trial_order)
            child.sad_trial_done = False
            child.sad_excluded_from_merge = False
            child.sad_merged = False
            child.sad_no_merge_branch = bool(no_merge_branch)
            child.sad_terminal_branch = bool(terminal_children)
            child.exclusion_state = int(getattr(node, "exclusion_state", 0))
            child.merge_disabled = int(bool(no_merge_branch) or bool(getattr(node, "merge_disabled", 0)))
            child.evaluation_completed = 0
            self._update_node_geometry(child)
            node.children.append(child)
            self._all_nodes.append(child)
            self._plot_nodes.append(child)
            children.append(child)

        node.expanded = True
        node.sad_no_merge_branch = bool(no_merge_branch)
        node.sad_children_terminal_after_trial = False
        node.sad_finalize_top_eff_after_cohort = bool(finalize_top_eff_after_cohort)
        node.sad_deferred_cohort_merged = False
        node.sad_eliminate_for_cohort = int(eliminate_count)
        node.sad_split_had_remainder = bool(has_remainder)
        for child in children:
            self._update_leaf_stats(child)

        self._print_event(
            "[AGVS][split-trial-cohort] "
            f"parent={self._node_label(node)} cohort={cohort_id} "
            f"n_children={len(children)} target_n_split={self.sad_n_split} "
            f"effective_eliminate={eliminate_count} "
            f"remainder_group={bool(has_remainder)} "
            f"merge_enabled={not bool(no_merge_branch)} "
            f"finalize_top_eff_after_cohort={bool(finalize_top_eff_after_cohort)} "
            f"mode=deferred_ave_diff_score_children"
        )
        self._plot_group_tree(selected=None, iteration=len(self.y))
        return True

    def _planned_split_groups(
        self,
        dims_sorted: List[int],
        target_groups: Optional[int] = None,
    ) -> Tuple[List[List[int]], int, bool]:
        dims_sorted = [int(dim) for dim in list(dims_sorted)]
        dim_count = len(dims_sorted)
        if dim_count <= 1:
            return [], int(getattr(self, "sad_eliminate", 1)), False

        target_groups = max(
            2,
            int(getattr(self, "sad_n_split", 2) if target_groups is None else target_groups),
        )
        if dim_count <= target_groups:
            groups = [[int(dim)] for dim in dims_sorted]
            has_remainder = False
        else:
            base_size = max(1, dim_count // target_groups)
            remainder = dim_count % target_groups
            groups = [
                list(dims_sorted[i * base_size : (i + 1) * base_size])
                for i in range(target_groups)
            ]
            remainder_elements = dims_sorted[base_size * target_groups:]
            for idx, elem in enumerate(remainder_elements):
                # The order is already a random permutation. Distribute extras
                # from the last group back toward the first for balanced sizes.
                reverse_group_idx = target_groups - 1 - (idx % target_groups)
                groups[reverse_group_idx].append(int(elem))
            groups = [group for group in groups if len(group) > 0]
            has_remainder = bool(remainder > 0)

        eliminate_count = int(getattr(self, "sad_eliminate", 1))
        return groups, int(eliminate_count), bool(has_remainder)

    def _update_path_scores_bottom_up(self, leaf: Node) -> None:
        super()._update_path_scores_bottom_up(leaf)
        if bool(getattr(self, "_sad_running_immediate_cohort", False)):
            return
        self._mark_deferred_child_trial_done(leaf)

    def _mark_deferred_child_trial_done(self, leaf: Node) -> None:
        if getattr(leaf, "parent", None) is None:
            return
        cohort_id = getattr(leaf, "sad_cohort_id", None)
        if cohort_id is None:
            return
        if int(getattr(leaf, "selected_count", 0)) <= 0:
            return
        if bool(getattr(leaf, "sad_trial_done", False)):
            self._maybe_complete_deferred_cohort(leaf.parent, str(cohort_id))
            return
        leaf.sad_trial_done = True
        self._print_event(
            "[AGVS][deferred-child-trial] "
            f"node={self._node_label(leaf)} cohort={cohort_id} "
            f"ave_diff_score={float(getattr(leaf, 'ave_diff_score', np.inf)):.6g}"
        )
        self._maybe_complete_deferred_cohort(leaf.parent, str(cohort_id))

    def _maybe_complete_deferred_cohort(self, parent: Node, cohort_id: str) -> None:
        if bool(getattr(parent, "sad_deferred_cohort_merged", False)):
            return
        siblings = [
            child
            for child in list(getattr(parent, "children", []) or [])
            if str(getattr(child, "sad_cohort_id", "")) == str(cohort_id)
            and not bool(getattr(child, "split_frozen", False))
        ]
        if not siblings:
            return
        if not all(bool(getattr(child, "sad_trial_done", False)) for child in siblings):
            return
        for child in siblings:
            child.evaluation_completed = 1
        parent.sad_deferred_cohort_merged = True
        if bool(getattr(parent, "sad_no_merge_branch", False)):
            for child in siblings:
                child.sad_no_merge_branch = True
                child.sad_excluded_from_merge = False
                child.sad_merged = False
                if hasattr(child, "sad_cohort_id"):
                    delattr(child, "sad_cohort_id")
            self._print_event(
                "[AGVS][no-merge-cohort-complete] "
                f"parent={self._node_label(parent)} cohort={cohort_id} "
                f"children={[self._node_label(child) for child in siblings]}"
            )
            return
        self._print_event(
            "[AGVS][deferred-cohort-complete] "
            f"parent={self._node_label(parent)} cohort={cohort_id} "
            f"children={[self._node_label(child) for child in siblings]}"
        )
        merge(self, parent, siblings, d_eff=self.eff_dim, cohort_id=cohort_id)

    def _merge_sad_cohort(self, parent: Node, cohort_id: str, siblings: List[Node]) -> Optional[Node]:
        if bool(getattr(parent, "sad_no_merge_branch", False)):
            for child in siblings:
                child.sad_no_merge_branch = True
                child.sad_excluded_from_merge = False
                child.sad_merged = False
                if hasattr(child, "sad_cohort_id"):
                    delattr(child, "sad_cohort_id")
            self._print_event(
                "[AGVS][no-merge-cohort-complete] "
                f"parent={self._node_label(parent)} cohort={cohort_id} "
                f"children={[self._node_label(child) for child in siblings]}"
            )
            return None

        finite_siblings = [
            node for node in siblings
            if np.isfinite(float(getattr(node, "ave_diff_score", np.inf)))
        ]
        eliminate_count = int(getattr(parent, "sad_eliminate_for_cohort", self.sad_eliminate))
        if len(finite_siblings) < len(siblings):
            self._print_event(
                "[AGVS][merge-skip] "
                f"parent={self._node_label(parent)} cohort={cohort_id} "
                f"finite_ave_diff_score={len(finite_siblings)}/{len(siblings)}"
            )
            parent.sad_deferred_cohort_merged = False
            return None
        ranked_highest = sorted(
            finite_siblings,
            key=lambda node: (
                -float(getattr(node, "ave_diff_score", -np.inf)),
                int(node.h),
                int(node.idx),
            ),
        )
        final_terminal_split = bool(
            getattr(parent, "sad_finalize_top_eff_after_cohort", False)
            and len(ranked_highest) >= int(self.eff_dim)
        )
        keep_count = int(self.eff_dim) if final_terminal_split else len(ranked_highest) - eliminate_count
        keep = ranked_highest[:keep_count]
        keep_ids = {int(node.idx) for node in keep}
        excluded = [node for node in siblings if int(node.idx) not in keep_ids]
        if not keep:
            for node in excluded:
                node.split_frozen = False
                node.sad_excluded_from_merge = True
                node.sad_no_merge_branch = True
            return None

        merged_dims = sorted({int(dim) for node in keep for dim in getattr(node, "dims", [])})
        merged_local_indices = sorted({
            int(idx)
            for node in keep
            for idx in getattr(node, "local_indices", [])
            if 0 <= int(idx) < len(self.y)
        })

        merged = Node(dims=merged_dims, h=int(parent.h) + 1, idx=self._next_node_idx, parent=parent)
        self._next_node_idx += 1
        merged.split_type = "sad_merge"
        merged.local_indices = (
            list(range(len(self.y))) if len(merged_dims) == self.eff_dim
            else list(merged_local_indices)
        )
        merged.score_indices = []
        merged.node_score_history = []
        merged.initial_selection_priority = True
        merged.selected_count = 0
        merged.ave_diff_score = float("inf")
        merged.mean_abs_best_diff = float("inf")
        merged.selection_score = float("-inf")
        merged.score_value_source = "awaiting_merged_node_own_visit"
        merged.sad_merged_node = True
        merged.sad_source_cohort_id = cohort_id
        merged.sad_source_nodes = [int(node.idx) for node in keep]
        merged.sad_excluded_nodes = [int(node.idx) for node in excluded]
        merged.sad_final_terminal_split = bool(final_terminal_split)
        merged.sad_split_target_groups = int(self.eff_dim) if final_terminal_split else None
        merged.sad_never_split = False
        merged.exclusion_state = 0
        merged.merge_disabled = 0
        merged.evaluation_completed = 0

        parent_bounds = self._node_bounds(parent)
        dim_to_local = {int(dim): idx for idx, dim in enumerate(parent.dims)}
        merged.local_bounds = np.asarray(
            [parent_bounds[dim_to_local[int(dim)]] for dim in merged_dims],
            dtype=float,
        )
        self._update_node_geometry(merged)

        for child in keep:
            child.split_frozen = True
            child.plot_archived = True
            child.sad_merged = True

        for child in excluded:
            owned_indices = self._owned_indices_for_nodes(
                [child],
                fallback_union=list(getattr(child, "score_indices", []) or []),
            )
            child.local_indices = list(owned_indices)
            child.score_indices = list(owned_indices)
            child.split_frozen = False
            child.plot_archived = False
            child.sad_excluded_from_merge = True
            child.sad_no_merge_branch = True
            child.sad_merged = False
            child.exclusion_state = 1
            child.merge_disabled = 1
            child.evaluation_completed = 1
            if hasattr(child, "sad_cohort_id"):
                delattr(child, "sad_cohort_id")

        parent.children = list(excluded) + [merged]
        self._plot_nodes = [
            node for node in self._plot_nodes
            if int(getattr(node, "idx", -1)) not in keep_ids
        ]
        self._all_nodes.append(merged)
        self._plot_nodes.append(merged)

        self._update_parent_score_from_children(parent)
        merged.sad_requires_own_trial = True
        self._refresh_leaf_scores()
        merged_was_split = False

        merged_best_idx, merged_best_y, _ = self._node_best_trial(merged)
        record = {
            "after_iteration": int(len(self.y)),
            "parent": self._node_label(parent),
            "cohort_id": str(cohort_id),
            "excluded_nodes": [self._node_label(node) for node in excluded],
            "excluded_ave_diff_scores": [float(getattr(node, "ave_diff_score", np.inf)) for node in excluded],
            "merged_node": self._node_label(merged),
            "merged_source_nodes": [self._node_label(node) for node in keep],
            "merge_selection_policy": (
                f"top_{int(self.eff_dim)}_ave_diff_score_final"
                if final_terminal_split
                else f"eliminate_bottom_{eliminate_count}_ave_diff_score"
            ),
            "eliminate_count": int(eliminate_count),
            "final_terminal_split": bool(final_terminal_split),
            "merged_dims": list(merged_dims),
            "merged_split_source": "seeded_optimizer_rng_random_permutation",
            "merged_n": int(merged.n),
            "merged_immediate_split": bool(merged_was_split),
            "merged_requires_own_trial": True,
            "required_own_trial_evaluations": int(
                self.t_seq * self.selection_batch_size
            ),
            "merged_split_policy": (
                "fit when selected, evaluate t_seq point batches without a trailing "
                "refit, then randomly permute and group dimensions"
            ),
            "merged_terminal_after_child_trials": bool(getattr(merged, "sad_no_merge_branch", False)),
            "merged_child_nodes": [self._node_label(child) for child in list(getattr(merged, "children", []) or [])],
            "merged_best_base_index": None if merged_best_idx is None else int(merged_best_idx),
            "merged_best_base_y": float(merged_best_y),
        }
        self.sad_merge_history.append(record)
        self._print_event(
            "[AGVS][merge] "
            f"after_iteration={record['after_iteration']} parent={record['parent']} "
            f"excluded={record['excluded_nodes']} merged={record['merged_node']} "
            f"merged_sources={record['merged_source_nodes']}"
        )
        self._plot_group_tree(selected=merged, iteration=len(self.y))
        return merged

class AGVSOptimizer(AGVSRegroupingOptimizer):
    """AG-VS with runner-selectable absolute or signed score mode.

    The tree mechanics, split/merge schedule, GP fitting, acquisition, q_batch,
    and best-fill behavior are inherited from AGVS. Node choice and merge
    elimination use ave_diff_score:

        abs_mode=True:
        Score(N) = mean_i(|y_obs_i - historical global best_y_obs_{i-1}|)

        abs_mode=False:
        Score(N) = mean_i(y_obs_i - historical global best_y_obs_{i-1})

    The historical global best observation is captured immediately before
    evaluation i and never rewritten. Score computation does not call the GP.

    Larger ave_diff_score is selected by following the max-ave_diff_score child at each
    tree level. Pending children on that path are tried in split order. During
    a normal merge, exactly N_split - d_eff low-scoring children are excluded.
    If a merged node has more than d_eff and at most N_split dimensions, its
    singleton cohort keeps and merges the top-d_eff children. Every merged node
    first fits its own GP and receives one full t_seq point-batch trial before
    any split. If it already has exactly d_eff dimensions, that redundant
    re-merge is skipped. The resulting node is then split into exactly d_eff
    terminal children from a new random dimension permutation. Their scores remain +inf,
    they are selected round-robin within the chosen path, and they never split
    or merge again. Nodes with fewer than two dimensions never split.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("selection_mode", None)
        configured_abs_mode = kwargs.pop("abs_mode", True)
        if not isinstance(configured_abs_mode, (bool, np.bool_)):
            raise ValueError("abs_mode must be a boolean.")
        self.abs_mode = bool(configured_abs_mode)
        ipt_solver = str(kwargs.get("ipt_solver", kwargs.get("node_backend", "vanillabo"))).strip().lower()
        if ipt_solver == "bo":
            ipt_solver = "vanillabo"
        self.node_backend = str(kwargs.pop("node_backend", ipt_solver)).strip().lower()
        if self.node_backend not in {"vanillabo", "turbo"}:
            raise ValueError("AG-VS node_backend must be 'vanillabo' or 'turbo'.")
        kwargs["ipt_solver"] = self.node_backend
        self.turbo_max_evals = int(kwargs.pop("turbo_max_evals", TURBO_REFERENCE_SPEC.turbo_max_evals))
        self.turbo_device = str(kwargs.pop("turbo_device", TURBO_REFERENCE_SPEC.device)).strip().lower()
        self.turbo_min_cuda = int(kwargs.pop("turbo_min_cuda", TURBO_REFERENCE_SPEC.min_cuda))
        if self.turbo_min_cuda < 0:
            raise ValueError("turbo_min_cuda must be non-negative.")
        if self.turbo_device not in {"cpu", "cuda"}:
            raise ValueError("turbo_device must be 'cpu' or 'cuda'.")
        if self.turbo_device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise ImportError("turbo_device='cuda' requires PyTorch with CUDA support.") from exc
            if not torch.cuda.is_available():
                raise RuntimeError("turbo_device='cuda' requested, but CUDA is unavailable.")
        super().__init__(*args, **kwargs)
        configured_point_batch_size = (
            self.q_batch if self.q_batch is not None else self.batch_size
        )
        if self.turbo_max_evals <= int(configured_point_batch_size):
            raise ValueError(
                "turbo_max_evals must exceed the AG-VS scheduler proposal count B."
            )
        self.selection_mode = "ave_diff_score"
        self.turbo_infrastructure = turbo_reference_dict(device=self.turbo_device)
        self.turbo_infrastructure.update(
            {
                "min_cuda": int(self.turbo_min_cuda),
                "small_data_runtime": (
                    "CUDA float32 from the first local GP fit (min_cuda=0)"
                    if self.turbo_min_cuda == 0 and self.turbo_device == "cuda"
                    else self.turbo_infrastructure.get("small_data_runtime")
                ),
                "gp_kernel": "ScaleKernel(MaternKernel(nu=2.5, ARD=False))",
                "use_ard": False,
                "anisotropic": False,
                "isotropic": True,
                "internal_batch_size": int(configured_point_batch_size),
                "turbo_backend_batch_policy": (
                    "no independent TuRBO batch setting; proposal count supplied "
                    "by the AG-VS scheduler B"
                ),
                "final_batch_budget_rule": (
                    f"AG-VS requests B={int(configured_point_batch_size)} Thompson candidates and evaluates "
                    "only the remaining budget-safe prefix in the terminal batch"
                ),
                "candidate_selection": (
                    f"TuRBO Thompson proposals requested by AG-VS with B={int(configured_point_batch_size)}"
                ),
            }
        )
        self.turbo_infrastructure["turbo_max_evals"] = int(self.turbo_max_evals)
        self.bo_infrastructure.update(
            {
                "compute_device": self.turbo_device,
                "cuda_execution": (
                    "canonical sklearn hyperparameter fit on CPU; fitted Matern-5/2 "
                    "posterior mean/std/covariance and analytic-EI candidate scoring on CUDA"
                    if self.node_backend == "vanillabo" and self.turbo_device == "cuda"
                    else None
                ),
            }
        )

    def _fit_local_gp(self, node: Node):
        """Fit/cache the one node GP used by every GP-dependent operation."""
        if self.node_backend != "turbo":
            model = super()._fit_local_gp(node)
            if (
                model is not None
                and self.turbo_device == "cuda"
                and model.__class__.__name__ != "CudaSklearnGPPredictor"
            ):
                from ag_vs.inner_optimizer.vanillabo_cuda import cuda_wrap_sklearn_gp

                model = cuda_wrap_sklearn_gp(model)
                node.model = model
            return model

        fit_signature = self._node_gp_fit_signature(node)
        cached_model = getattr(node, "model", None)
        if cached_model is not None and getattr(node, "_gp_fit_signature", None) == fit_signature:
            return cached_model
        self._invalidate_node_gp(node)
        try:
            model = fit_turbo_node_gp(self, node)
        except Exception as exc:
            self._last_gp_fit_error = exc
            node._last_gp_fit_error = exc
            self._invalidate_node_gp(node)
            return None
        if model is None:
            self._last_gp_fit_error = RuntimeError(
                "TuRBO GP fitter returned no model without raising an exception."
            )
            node._last_gp_fit_error = self._last_gp_fit_error
            self._invalidate_node_gp(node)
            return None
        self._last_gp_fit_error = None
        node._last_gp_fit_error = None
        node.model = model
        node._gp_fit_signature = fit_signature
        if list(getattr(node, "dims", []) or []) == list(range(self.D)):
            self._last_full_dim_gp_model = model
        try:
            self._record_gp_fit(node)
            self._update_posterior_dataset_from_node(node, model)
        except Exception:
            pass
        return model

    def _is_completed_pending_cohort_child(self, node: Node) -> bool:
        return False

    def _evaluate(self, x: np.ndarray):
        y_true, y_obs = super()._evaluate(x)
        idx = int(len(self.y) - 1)
        if not hasattr(self, "_diff_value_by_index"):
            self._diff_value_by_index = {}
        if not hasattr(self, "_best_y_at_index"):
            self._best_y_at_index = {}
        if idx < self.n_init:
            best_y_so_far = -np.inf
            self._diff_value_by_index[idx] = float("inf")
        else:
            best_y_so_far = float(np.max(np.asarray(self.y[:idx], dtype=float)))
            self._diff_value_by_index[idx] = float(abs(float(y_obs) - best_y_so_far))
        self._best_y_at_index[idx] = float(best_y_so_far)
        return y_true, y_obs

    def _assign_new_index_to_leaf_tree(self, new_idx: int, leaf: Node) -> None:
        super()._assign_new_index_to_leaf_tree(new_idx, leaf)
        leaf.diff_source_count = int(len(getattr(leaf, "score_indices", []) or []))
        if not hasattr(self, "_source_leaf_by_index"):
            self._source_leaf_by_index = {}
        self._source_leaf_by_index[int(new_idx)] = leaf

    def _propose_from_leaf_turbo(self, leaf: Node):
        return propose_turbo_one(self, leaf)

    def _propose_q_from_leaf_turbo(self, leaf: Node, q: int):
        return propose_turbo_q(self, leaf, q)

    def _uses_turbo_for_leaf(self, leaf: Node) -> bool:
        del leaf
        return self.node_backend == "turbo"

    def _evaluation_batch_size(self, leaf: Node, remaining_budget: int) -> int:
        if self._uses_turbo_for_leaf(leaf):
            # TuRBO proposes exactly the configured q-batch and updates its
            # trust region from the same observations.
            return super()._evaluation_batch_size(leaf, remaining_budget)
        return super()._evaluation_batch_size(leaf, remaining_budget)

    def _node_backend_for_history(self, leaf: Node) -> str:
        if self._uses_turbo_for_leaf(leaf):
            return "turbo"
        return "ei_cuda" if self.turbo_device == "cuda" else "ei"

    def _node_proposal_strategy(self, leaf: Node) -> str:
        if self._uses_turbo_for_leaf(leaf):
            return "turbo_thompson_sampling"
        if self.q_batch is not None:
            return "greedy_monte_carlo_joint_q_expected_improvement"
        return "independent_top_k_pointwise_expected_improvement"

    def _should_record_posterior_for_node(self, node: Node) -> bool:
        del node
        return False

    def _after_proposal_batch(self, leaf: Node, evaluated_indices: Sequence[int]) -> None:
        if self._uses_turbo_for_leaf(leaf):
            update_turbo_after_batch(self, leaf, evaluated_indices)

    def _propose_from_leaf(self, leaf: Node):
        if self._uses_turbo_for_leaf(leaf):
            return self._propose_from_leaf_turbo(leaf)
        return super()._propose_from_leaf(leaf)

    def _propose_q_from_leaf(self, leaf: Node, q: int):
        if self._uses_turbo_for_leaf(leaf):
            return self._propose_q_from_leaf_turbo(leaf, int(q))
        return super()._propose_q_from_leaf(leaf, int(q))


    def _node_score_indices(self, node: Node) -> List[int]:
        score_indices = getattr(node, "score_indices", None)
        if score_indices is None:
            score_indices = getattr(node, "local_indices", []) or []
        return [int(idx) for idx in score_indices if self.n_init <= int(idx) < len(self.y)]


    def _node_percent_best_diff(self, node: Node) -> float:
        valid = self._node_score_indices(node)
        if not valid or len(self.y) == 0:
            node.node_score_history = []
            node.mean_abs_best_diff = float("inf")
            node.risk_bonus = float("inf")
            return float("inf")

        node.node_score_history = build_node_score_history(
            score_indices=valid,
            X=self.X,
            n_init=self.n_init,
            observed_values=self.y,
            historical_best_observed_by_index=self.historical_best_observed_by_index,
        )
        score = calculate_score(
            node.node_score_history,
            abs_mode=self.abs_mode,
        )
        node.mean_abs_best_diff = score
        node.risk_bonus = 0.0
        node.diff_best_bonus = 0.0
        node.score_value_source = "observed_y_vs_historical_best_observed_y"
        node.score_posterior_source = "not_used"
        node.diff_score_mode = (
            "absolute_unscaled_mean_observation_gap"
            if self.abs_mode
            else "signed_unscaled_mean_observation_gap"
        )
        return score

    def _update_node_ave_diff_score(self, node: Node) -> float:
        ave_diff_score = self._node_percent_best_diff(node)
        node.ave_diff_score = float(ave_diff_score)
        node.selection_score = float(ave_diff_score)
        return float(ave_diff_score)

    def _weighted_score_from_nodes(self, nodes: List[Node]) -> float:
        return weighted_score(
            (
                float(getattr(child, "ave_diff_score", np.inf)),
                int(len(self._node_score_indices(child))),
            )
            for child in nodes
        )

    def _update_parent_score_from_children(self, parent: Node) -> float:
        """Set a parent score without fitting a parent GP."""
        baseline_count = int(getattr(parent, "sad_pre_split_data_count", 0))
        baseline_score = float(getattr(parent, "sad_pre_split_ave_diff_score", np.inf))
        score_and_count = [(baseline_score, baseline_count)]
        score_and_count.extend(
            (
                float(getattr(child, "ave_diff_score", np.inf)),
                int(len(self._node_score_indices(child))),
            )
            for child in list(getattr(parent, "children", []) or [])
        )
        parent_score = weighted_score(score_and_count)
        parent.ave_diff_score = float(parent_score)
        parent.mean_abs_best_diff = float(parent_score)
        parent.score_value_source = "weighted_child_scores"
        parent.diff_path_score = float(parent_score)
        parent.selection_score = -float(parent_score)
        return float(parent_score)

    def _update_leaf_stats(self, node: Node) -> None:
        self._update_node_geometry(node)
        if bool(getattr(node, "sad_terminal_branch", False)):
            node.ave_diff_score = np.inf
            node.diff_path_score = np.inf
            node.selection_score = -np.inf
            return
        self._update_node_ave_diff_score(node)

    def _refresh_leaf_scores(self) -> None:
        """Refresh selection metadata without recomputing unchanged node scores.

        Scores are updated only for the selected data-receiving path after its
        t_seq block.  Unselected branches keep their last local-GP score.
        """
        for node in sorted(getattr(self, "_all_nodes", []) or [], key=lambda item: int(item.h), reverse=True):
            if bool(getattr(node, "split_frozen", False)) or bool(getattr(node, "plot_archived", False)):
                node.ave_diff_score = -np.inf
                node.diff_path_score = -np.inf
                node.selection_score = np.inf
                continue
            if self._is_completed_pending_cohort_child(node):
                node.selection_score = np.inf
                continue
            if bool(getattr(node, "sad_terminal_branch", False)):
                self._update_node_geometry(node)
                node.ave_diff_score = np.inf
                node.diff_path_score = np.inf
                node.selection_score = -np.inf
                continue
            node.diff_path_score = float(getattr(node, "ave_diff_score", -np.inf))
            node.selection_score = -float(node.diff_path_score)

    def _select_node(self) -> Node:
        self._refresh_leaf_scores()
        roots = [
            node
            for node in list(getattr(self, "roots", []) or [])
            if not bool(getattr(node, "split_frozen", False))
            and not bool(getattr(node, "plot_archived", False))
        ]
        if not roots:
            raise RuntimeError("No active root nodes available for AGVS selection.")

        def choose(nodes: List[Node]) -> Node:
            candidates = [
                node for node in nodes
                if not bool(getattr(node, "split_frozen", False))
                and not bool(getattr(node, "plot_archived", False))
                and not self._is_completed_pending_cohort_child(node)
            ]
            if not candidates:
                raise RuntimeError("No active nodes available while following AGVS path.")
            untried_siblings = [
                node for node in candidates
                if getattr(node, "sad_cohort_id", None) is not None
                and not bool(getattr(node, "sad_trial_done", False))
            ]
            if untried_siblings:
                return min(
                    untried_siblings,
                    key=lambda node: (
                        int(getattr(node, "sad_trial_order", int(node.idx))),
                        int(node.h),
                        int(node.idx),
                    ),
                )
            own_trial_nodes = [
                node for node in candidates
                if bool(getattr(node, "sad_requires_own_trial", False))
                or bool(getattr(node, "restart_requires_own_trial", False))
            ]
            if own_trial_nodes:
                return min(
                    own_trial_nodes,
                    key=lambda node: (int(node.h), int(node.idx)),
                )
            terminal_siblings = [
                node for node in candidates
                if bool(getattr(node, "sad_terminal_branch", False))
            ]
            if terminal_siblings:
                return min(
                    terminal_siblings,
                    key=lambda node: (
                        int(getattr(node, "selected_count", 0)),
                        int(getattr(node, "sad_trial_order", int(node.idx))),
                        int(node.h),
                        int(node.idx),
                    ),
                )
            return min(
                candidates,
                key=lambda node: (
                    -float(getattr(node, "ave_diff_score", -np.inf)),
                    int(node.h),
                    int(node.idx),
                ),
            )

        current = choose(roots)
        while True:
            children = self._eligible_children(current)
            if not children:
                return current
            current = choose(children)

    def _prepare_selected_node_for_evaluation(self, selected_node: Node) -> Node:
        selected_node = super()._prepare_selected_node_for_evaluation(selected_node)
        if bool(getattr(selected_node, "sad_requires_own_trial", False)):
            self._print_event(
                "[AGVS][merged-own-trial] "
                f"node={self._node_label(selected_node)} will receive "
                f"t_seq={self.t_seq} x batch_size={self.selection_batch_size} "
                "evaluations before random dimension splitting"
            )
        if bool(getattr(selected_node, "restart_requires_own_trial", False)):
            self._print_event(
                "[AGVS][restart-root-own-trial] "
                f"node={self._node_label(selected_node)} will receive "
                f"t_seq={self.t_seq} x batch_size={self.selection_batch_size} "
                "evaluations before random dimension splitting"
            )
        return selected_node

    def _plot_node_score_lines(self, node: Node) -> List[str]:
        backend = self._node_backend_for_history(node)
        if bool(getattr(node, "sad_terminal_branch", False)):
            return [
                f"backend={backend}",
                f"terminal_order={int(getattr(node, 'sad_trial_order', int(node.idx)))}",
                f"selected={int(getattr(node, 'selected_count', 0))}",
            ]
        return [
            f"backend={backend}",
            f"ave_diff_score={self._format_plot_value(getattr(node, 'ave_diff_score', np.inf), digits=3)}",
            f"bonus={self._format_plot_value(getattr(node, 'risk_bonus', np.inf), digits=3)}",
        ]

    def _mark_deferred_child_trial_done(self, leaf: Node) -> None:
        if getattr(leaf, "parent", None) is None:
            return
        cohort_id = getattr(leaf, "sad_cohort_id", None)
        if cohort_id is None:
            return
        if int(getattr(leaf, "selected_count", 0)) <= 0:
            return
        if bool(getattr(leaf, "sad_trial_done", False)):
            self._maybe_complete_deferred_cohort(leaf.parent, str(cohort_id))
            return
        leaf.sad_trial_done = True
        if bool(getattr(leaf, "sad_terminal_branch", False)):
            leaf.ave_diff_score = np.inf
            self._print_event(
                "[AGVS][deferred-child-trial] "
                f"node={self._node_label(leaf)} cohort={cohort_id} ave_diff_score=inf_terminal"
            )
        else:
            self._print_event(
                "[AGVS][deferred-child-trial] "
                f"node={self._node_label(leaf)} cohort={cohort_id} "
                f"ave_diff_score={float(getattr(leaf, 'ave_diff_score', np.inf)):.6g}"
            )
        self._maybe_complete_deferred_cohort(leaf.parent, str(cohort_id))

    def _update_path_scores_bottom_up(self, leaf: Node) -> None:
        # Only the evaluated leaf owns/fits a GP score. Ancestors receive the
        # data and aggregate their direct children's stored scores; they never
        # fit a separate GP for parent scoring.
        if bool(getattr(leaf, "sad_terminal_branch", False)):
            self._update_node_geometry(leaf)
            leaf.ave_diff_score = np.inf
        else:
            self._update_leaf_stats(leaf)
        leaf.diff_path_score = float(getattr(leaf, "ave_diff_score", -np.inf))
        leaf.selection_score = -float(leaf.diff_path_score)
        parent = getattr(leaf, "parent", None)
        while parent is not None:
            self._update_parent_score_from_children(parent)
            parent = getattr(parent, "parent", None)
        if bool(getattr(self, "_sad_running_immediate_cohort", False)):
            return
        self._mark_deferred_child_trial_done(leaf)

    def _merge_sad_cohort(self, parent: Node, cohort_id: str, siblings: List[Node]) -> Optional[Node]:
        if bool(getattr(parent, "sad_no_merge_branch", False)):
            for child in siblings:
                child.sad_no_merge_branch = True
                child.sad_excluded_from_merge = False
                child.sad_merged = False
                if hasattr(child, "sad_cohort_id"):
                    delattr(child, "sad_cohort_id")
            self._print_event(
                "[AGVS][no-merge-cohort-complete] "
                f"parent={self._node_label(parent)} cohort={cohort_id} "
                f"children={[self._node_label(child) for child in siblings]}"
            )
            return None

        finite_siblings = [
            node for node in siblings
            if np.isfinite(float(getattr(node, "ave_diff_score", np.inf)))
        ]
        eliminate_count = int(getattr(parent, "sad_eliminate_for_cohort", self.sad_eliminate))
        if len(finite_siblings) < len(siblings):
            self._print_event(
                "[AGVS][merge-skip] "
                f"parent={self._node_label(parent)} cohort={cohort_id} "
                f"finite_ave_diff_score={len(finite_siblings)}/{len(siblings)}"
            )
            parent.sad_deferred_cohort_merged = False
            return None
        ranked_highest_diff = sorted(
            finite_siblings,
            key=lambda node: (
                -float(getattr(node, "ave_diff_score", -np.inf)),
                int(node.h),
                int(node.idx),
            ),
        )
        final_terminal_split = bool(
            getattr(parent, "sad_finalize_top_eff_after_cohort", False)
            and len(ranked_highest_diff) >= int(self.eff_dim)
        )
        keep_count = int(self.eff_dim) if final_terminal_split else len(ranked_highest_diff) - eliminate_count
        keep = ranked_highest_diff[:keep_count]
        keep_ids = {int(node.idx) for node in keep}
        excluded = [node for node in siblings if int(node.idx) not in keep_ids]
        if not keep:
            for node in excluded:
                node.split_frozen = False
                node.sad_excluded_from_merge = True
                node.sad_no_merge_branch = True
            return None

        merged_dims = sorted({int(dim) for node in keep for dim in getattr(node, "dims", [])})
        merged_local_indices = sorted({
            int(idx)
            for node in keep
            for idx in getattr(node, "local_indices", [])
            if 0 <= int(idx) < len(self.y)
        })

        merged = Node(dims=merged_dims, h=int(parent.h) + 1, idx=self._next_node_idx, parent=parent)
        self._next_node_idx += 1
        merged.split_type = "diff_merge"
        merged.local_indices = (
            list(range(len(self.y))) if len(merged_dims) == self.eff_dim
            else list(merged_local_indices)
        )
        merged.score_indices = []
        merged.node_score_history = []
        merged.initial_selection_priority = True
        merged.selected_count = 0
        merged.ave_diff_score = float("inf")
        merged.mean_abs_best_diff = float("inf")
        merged.selection_score = float("-inf")
        merged.score_value_source = "awaiting_merged_node_own_visit"
        merged.sad_merged_node = True
        merged.sad_source_cohort_id = cohort_id
        merged.sad_source_nodes = [int(node.idx) for node in keep]
        merged.sad_excluded_nodes = [int(node.idx) for node in excluded]
        merged.sad_final_terminal_split = bool(final_terminal_split)
        merged.sad_split_target_groups = int(self.eff_dim) if final_terminal_split else None
        merged.sad_never_split = False
        merged.exclusion_state = 0
        merged.merge_disabled = 0
        merged.evaluation_completed = 0

        parent_bounds = self._node_bounds(parent)
        dim_to_local = {int(dim): idx for idx, dim in enumerate(parent.dims)}
        merged.local_bounds = np.asarray(
            [parent_bounds[dim_to_local[int(dim)]] for dim in merged_dims],
            dtype=float,
        )
        self._update_node_geometry(merged)

        for child in keep:
            child.split_frozen = True
            child.plot_archived = True
            child.sad_merged = True

        for child in excluded:
            owned_indices = self._owned_indices_for_nodes(
                [child],
                fallback_union=list(getattr(child, "score_indices", []) or []),
            )
            child.local_indices = list(owned_indices)
            child.score_indices = list(owned_indices)
            child.split_frozen = False
            child.plot_archived = False
            child.sad_excluded_from_merge = True
            child.sad_no_merge_branch = True
            child.sad_merged = False
            child.exclusion_state = 1
            child.merge_disabled = 1
            child.evaluation_completed = 1
            if hasattr(child, "sad_cohort_id"):
                delattr(child, "sad_cohort_id")

        parent.children = list(excluded) + [merged]
        self._plot_nodes = [
            node for node in self._plot_nodes
            if int(getattr(node, "idx", -1)) not in keep_ids
        ]
        self._all_nodes.append(merged)
        self._plot_nodes.append(merged)

        self._update_parent_score_from_children(parent)
        merged.sad_requires_own_trial = True
        self._refresh_leaf_scores()
        merged_was_split = False

        merged_best_idx, merged_best_y, _ = self._node_best_trial(merged)
        record = {
            "after_iteration": int(len(self.y)),
            "parent": self._node_label(parent),
            "cohort_id": str(cohort_id),
            "excluded_nodes": [self._node_label(node) for node in excluded],
            "excluded_ave_diff_scores": [float(getattr(node, "ave_diff_score", np.inf)) for node in excluded],
            "merged_node": self._node_label(merged),
            "merged_source_nodes": [self._node_label(node) for node in keep],
            "merge_selection_policy": (
                f"top_{int(self.eff_dim)}_ave_diff_score_final"
                if final_terminal_split
                else f"eliminate_bottom_{eliminate_count}_ave_diff_score"
            ),
            "eliminate_count": int(eliminate_count),
            "final_terminal_split": bool(final_terminal_split),
            "merged_dims": list(merged_dims),
            "merged_split_source": "seeded_optimizer_rng_random_permutation",
            "merged_n": int(merged.n),
            "merged_ave_diff_score": float(getattr(merged, "ave_diff_score", np.inf)),
            "merged_immediate_split": bool(merged_was_split),
            "merged_requires_own_trial": True,
            "required_own_trial_evaluations": int(
                self.t_seq * self.selection_batch_size
            ),
            "merged_split_policy": (
                "fit when selected, evaluate t_seq point batches without a trailing "
                "refit, then randomly permute and group dimensions"
            ),
            "merged_terminal_after_child_trials": bool(getattr(merged, "sad_no_merge_branch", False)),
            "merged_child_nodes": [self._node_label(child) for child in list(getattr(merged, "children", []) or [])],
            "merged_best_base_index": None if merged_best_idx is None else int(merged_best_idx),
            "merged_best_base_y": float(merged_best_y),
        }
        self.sad_merge_history.append(record)
        self._print_event(
            "[AGVS][merge] "
            f"after_iteration={record['after_iteration']} parent={record['parent']} "
            f"excluded={record['excluded_nodes']} "
            f"excluded_ave_diff_scores={record['excluded_ave_diff_scores']} "
            f"merged={record['merged_node']} merged_sources={record['merged_source_nodes']}"
        )
        self._plot_group_tree(selected=merged, iteration=len(self.y))
        return merged

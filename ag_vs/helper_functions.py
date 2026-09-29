from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ag_vs.utils import Node


def traverse(optimizer: Any) -> Tuple[Node, List[Node]]:
    """Follow maximum Score values from an active root to an eligible leaf."""
    optimizer._refresh_leaf_scores()
    roots = [n for n in optimizer.roots if not n.split_frozen and not n.plot_archived]
    if not roots:
        raise RuntimeError("No active root is available for Traverse.")

    def choose(nodes: List[Node]) -> Node:
        eligible = [
            n for n in nodes
            if not n.split_frozen and not n.plot_archived
            and not optimizer._is_completed_pending_cohort_child(n)
        ]
        if not eligible:
            raise RuntimeError("No eligible node is available on the traversal path.")
        pending = [n for n in eligible if getattr(n, "sad_cohort_id", None) is not None and not getattr(n, "sad_trial_done", False)]
        if pending:
            return min(pending, key=lambda n: (n.sad_trial_order, n.h, n.idx))
        terminal = [n for n in eligible if getattr(n, "sad_terminal_branch", False)]
        if terminal:
            return min(terminal, key=lambda n: (n.selected_count, getattr(n, "sad_trial_order", n.idx), n.h, n.idx))
        return min(eligible, key=lambda n: (-float(getattr(n, "ave_diff_score", -np.inf)), n.h, n.idx))

    current = choose(roots)
    traversal_path = [current]
    while True:
        children = optimizer._eligible_children(current)
        if not children:
            return current, traversal_path
        current = choose(children)
        traversal_path.append(current)


def update(
    optimizer: Any,
    selected_node: Node,
    node_score_history: Optional[List[Dict]] = None,
    traversal_path: Optional[List[Node]] = None,
) -> None:
    """Update the selected leaf and aggregate its traversal path bottom-up."""
    del node_score_history
    if getattr(selected_node, "sad_terminal_branch", False):
        optimizer._update_node_geometry(selected_node)
        selected_node.ave_diff_score = np.inf
    else:
        optimizer._update_leaf_stats(selected_node)
    selected_node.diff_path_score = float(getattr(selected_node, "ave_diff_score", -np.inf))
    selected_node.selection_score = -selected_node.diff_path_score
    parents = list((traversal_path or [])[:-1])
    if not parents:
        parent = selected_node.parent
        while parent is not None:
            parents.append(parent)
            parent = parent.parent
    for parent in reversed(parents):
        optimizer._update_parent_score_from_children(parent)
    if not getattr(optimizer, "_sad_running_immediate_cohort", False):
        optimizer._mark_deferred_child_trial_done(selected_node)


def split(optimizer: Any, node: Node, K: Optional[int] = None) -> bool:
    """Create randomly permuted child coordinate groups directly."""
    dimensions = list(node.active_dimensions)
    if len(dimensions) < 2 or node.expanded or node.split_frozen:
        return False
    if getattr(node, "sad_never_split", False) or getattr(node, "sad_terminal_branch", False):
        return False
    if getattr(node, "sad_cohort_id", None) is not None and not getattr(node.parent, "sad_deferred_cohort_merged", False):
        return False
    ordered = [
        int(dim)
        for dim in optimizer.rng.permutation(np.asarray(dimensions, dtype=int)).tolist()
    ]
    groups, eliminate_count, has_remainder = optimizer._planned_split_groups(
        ordered, target_groups=K if K is not None else getattr(node, "sad_split_target_groups", None)
    )
    if len(groups) < 2:
        return False
    node.sad_split_permutation = list(ordered)
    bounds = optimizer._node_bounds(node)
    position = {dim: i for i, dim in enumerate(dimensions)}
    cohort_id = f"{node.idx}:{len(optimizer.sad_merge_history)}:{len(optimizer.y)}"
    node.sad_pre_split_ave_diff_score = float(getattr(node, "ave_diff_score", np.inf))
    node.sad_pre_split_data_count = len(optimizer._node_score_indices(node))
    node.children = []
    for order, group in enumerate(groups):
        child = Node(list(group), node.h + 1, optimizer._next_node_idx, parent=node)
        optimizer._next_node_idx += 1
        child.split_type = "sad_random_group"
        child.local_bounds = np.asarray([bounds[position[d]] for d in group])
        child.local_indices = optimizer._inherited_local_indices(node)
        child.score_indices = []
        child.ave_diff_score = np.inf
        child.selection_score = -np.inf
        child.sad_cohort_id = cohort_id
        child.sad_trial_order = order
        child.sad_trial_done = False
        child.sad_no_merge_branch = bool(getattr(node, "sad_no_merge_branch", False))
        child.sad_terminal_branch = bool(getattr(node, "sad_children_terminal_after_trial", False))
        child.exclusion_state = node.exclusion_state
        child.merge_disabled = node.merge_disabled
        optimizer._update_node_geometry(child)
        node.children.append(child)
        optimizer._all_nodes.append(child)
        optimizer._plot_nodes.append(child)
    node.expanded = True
    node.sad_children_terminal_after_trial = False
    node.sad_finalize_top_eff_after_cohort = bool(
        getattr(node, "sad_merged_node", False)
        and optimizer.eff_dim < len(ordered) <= optimizer.sad_n_split
        and not getattr(node, "sad_final_terminal_split", False)
    )
    node.sad_deferred_cohort_merged = False
    node.sad_eliminate_for_cohort = eliminate_count
    node.sad_split_had_remainder = has_remainder
    optimizer._plot_group_tree(selected=None, iteration=len(optimizer.y))
    return True


def merge(
    optimizer: Any,
    parent: Node,
    siblings: List[Node],
    d_eff: Optional[int] = None,
    cohort_id: Optional[str] = None,
) -> Optional[Node]:
    """Merge top siblings; target-size nodes train on the full observation history."""
    if not siblings or getattr(parent, "sad_no_merge_branch", False):
        return None
    if any(not np.isfinite(float(getattr(n, "ave_diff_score", np.inf))) for n in siblings):
        parent.sad_deferred_cohort_merged = False
        return None
    d_eff = int(d_eff or optimizer.eff_dim)
    cohort_id = cohort_id or str(getattr(siblings[0], "sad_cohort_id", ""))
    ranked = sorted(siblings, key=lambda n: (-float(n.ave_diff_score), n.h, n.idx))
    keep = ranked[: min(d_eff, len(ranked))]
    excluded = ranked[len(keep):]
    dimensions = sorted({d for child in keep for d in child.active_dimensions})
    merged = Node(dimensions, parent.h + 1, optimizer._next_node_idx, parent=parent)
    optimizer._next_node_idx += 1
    merged.split_type = "diff_merge"
    if len(dimensions) == d_eff:
        merged.node_dataset = range(len(optimizer.y))
    else:
        merged.node_dataset = sorted({i for child in keep for i in child.node_dataset if 0 <= i < len(optimizer.y)})
    # Training data initializes the merged GP, while the new node earns its
    # own score. An infinite fresh score guarantees its first traversal.
    merged.score_indices = []
    merged.node_score_history = []
    merged.initial_selection_priority = True
    merged.selected_count = 0
    merged.ave_diff_score = float("inf")
    merged.mean_abs_best_diff = float("inf")
    merged.selection_score = float("-inf")
    merged.score_value_source = "awaiting_merged_node_own_visit"
    merged.sad_merged_node = True
    merged.sad_requires_own_trial = True
    merged.sad_source_cohort_id = cohort_id
    if len(dimensions) <= d_eff:
        merged.sad_never_split = True
        merged.sad_final_terminal_split = False
        merged.sad_split_target_groups = None
        merged.merge_disabled = 1
    else:
        merged.sad_never_split = False
        merged.sad_final_terminal_split = False
        merged.sad_split_target_groups = None
        merged.merge_disabled = 0
    parent_bounds = optimizer._node_bounds(parent)
    positions = {d: i for i, d in enumerate(parent.active_dimensions)}
    merged.local_bounds = np.asarray([parent_bounds[positions[d]] for d in dimensions])
    optimizer._update_node_geometry(merged)
    for child in keep:
        child.split_frozen = True
        child.plot_archived = True
        child.sad_merged = True
    for child in excluded:
        child.sad_excluded_from_merge = True
        child.sad_no_merge_branch = True
        child.exclusion_state = 1
        child.merge_disabled = 1
        child.evaluation_completed = 1
    parent.children = excluded + [merged]
    optimizer._plot_nodes = [n for n in optimizer._plot_nodes if n not in keep]
    optimizer._all_nodes.append(merged)
    optimizer._plot_nodes.append(merged)
    optimizer._update_parent_score_from_children(parent)
    optimizer._refresh_leaf_scores()
    optimizer.sad_merge_history.append({
        "after_iteration": len(optimizer.y), "parent": optimizer._node_label(parent),
        "cohort_id": cohort_id, "excluded_nodes": [optimizer._node_label(n) for n in excluded],
        "merged_node": optimizer._node_label(merged), "merged_dims": dimensions,
        "merged_immediate_split": False,
        "merged_requires_own_trial": True,
        "required_own_trial_evaluations": int(
            optimizer.t_seq * optimizer.selection_batch_size
        ),
        "merged_split_policy": (
            "use full observation history at the target dimension, otherwise inherit "
            "child training data; fit only when selected, evaluate t_seq point batches "
            "without a trailing refit, then split a seeded random permutation only "
            "above the target dimension"
        ),
    })
    optimizer._plot_group_tree(selected=merged, iteration=len(optimizer.y))
    return merged

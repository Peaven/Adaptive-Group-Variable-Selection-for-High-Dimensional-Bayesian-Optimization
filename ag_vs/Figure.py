from __future__ import annotations

import os
import time
import webbrowser
from typing import Dict, List, Optional, Tuple

import numpy as np

from ag_vs.utils import Node


class AGVSTreeFigure:
    def __init__(self, owner) -> None:
        self.owner = owner

    def visible_nodes(self) -> List[Node]:
        owner = self.owner
        nodes_by_idx: Dict[int, Node] = {}
        for node in list(owner._all_nodes) + list(owner._plot_nodes):
            nodes_by_idx[int(node.idx)] = node
        return sorted(nodes_by_idx.values(), key=lambda node: (int(node.h), int(node.idx)))

    def positions(self) -> Dict[int, Tuple[float, float]]:
        owner = self.owner
        positions: Dict[int, Tuple[float, float]] = {}
        leaf_counter = 0
        # Keep large node labels readable in dense final trees.
        x_spacing = 64.0
        y_spacing = 40.0
        plot_nodes = self.visible_nodes()
        child_map: Dict[int, List[Node]] = {node.idx: list(node.children) for node in plot_nodes}

        def visit(node: Node) -> float:
            nonlocal leaf_counter
            children = child_map.get(node.idx, [])
            if not children:
                x = float(leaf_counter) * x_spacing
                leaf_counter += 1
            else:
                child_x = [visit(child) for child in children]
                x = float(np.mean(child_x)) if child_x else float(leaf_counter)
            positions[node.idx] = (x, -float(node.h) * y_spacing)
            return x

        for root_index, root in enumerate(owner.roots):
            if root_index > 0:
                leaf_counter += 2
            visit(root)
        return positions

    def plot(self, selected: Optional[Node] = None, iteration: Optional[int] = None, force: bool = False) -> None:
        owner = self.owner
        if not owner.tree_plot_on and not bool(force):
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        plot_nodes = self.visible_nodes()
        if owner.tree_plot_fig is None or owner.tree_plot_ax is None:
            owner.tree_plot_fig, owner.tree_plot_ax = plt.subplots(figsize=(210, 108))
            try:
                plt.ion()
                owner.tree_plot_fig.show()
            except Exception:
                pass

        ax = owner.tree_plot_ax
        ax.clear()
        positions = self.positions()
        selected_idx = None if selected is None else int(selected.idx)
        selected_path = set()
        node_on_path = selected
        while node_on_path is not None:
            selected_path.add(int(node_on_path.idx))
            node_on_path = node_on_path.parent

        for node in plot_nodes:
            if node.parent is None:
                continue
            if node.parent.idx not in positions or node.idx not in positions:
                continue
            x0, y0 = positions[node.parent.idx]
            x1, y1 = positions[node.idx]
            is_path_edge = int(node.idx) in selected_path and int(node.parent.idx) in selected_path
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#d62728" if is_path_edge else "#8a8f98",
                linewidth=3.0 if is_path_edge else 1.2,
                zorder=1,
            )

        nodes_by_idx = {int(node.idx): node for node in plot_nodes}
        for node in plot_nodes:
            source_nodes = getattr(node, "sad_source_nodes", None)
            if not source_nodes or int(node.idx) not in positions:
                continue
            x1, y1 = positions[int(node.idx)]
            for source_idx in source_nodes:
                source = nodes_by_idx.get(int(source_idx))
                if source is None or int(source.idx) not in positions:
                    continue
                x0, y0 = positions[int(source.idx)]
                ax.annotate(
                    "",
                    xy=(x1, y1 + 0.8),
                    xytext=(x0, y0 - 0.8),
                    arrowprops={"arrowstyle": "->", "color": "#f97316", "linewidth": 2.2, "alpha": 0.9},
                    zorder=2,
                )

        for node in plot_nodes:
            if node.idx not in positions:
                continue
            x, y = positions[node.idx]
            has_active = bool(owner._node_active_dims(node))
            is_selected = selected_idx == int(node.idx)
            is_path_node = int(node.idx) in selected_path

            is_frozen = bool(getattr(node, "split_frozen", False))
            is_excluded = bool(getattr(node, "sad_excluded_from_merge", False)) or bool(getattr(node, "sel_excluded_from_merge", False))
            is_merged_node = bool(getattr(node, "sad_merged_node", False))
            owner._plot_node_marker(ax, x, y, node, is_selected, is_path_node, has_active)

            active_dims = owner._node_active_dims(node)
            label_lines = [
                f"{'SELECTED ' if is_selected else ''}{owner._node_label(node)}",
                f"idx={node.idx}",
                f"selected_count={int(getattr(node, 'selected_count', 0))}",
                *owner._plot_node_score_lines(node),
                f"D={node.D_hi}, Dim={node.Dim_hi}",
            ]
            if is_excluded:
                label_lines.append("EXCLUDED")
            elif is_merged_node:
                label_lines.append("MERGED")
            elif is_frozen:
                label_lines.append("FROZEN")
            if active_dims:
                active_text = ",".join(str(dim) for dim in active_dims[:10])
                if len(active_dims) > 10:
                    active_text += ",..."
                label_lines.append(f"active: {active_text}")
            is_active_leaf = bool(active_dims) and not bool(list(getattr(node, "children", []) or []))
            label_color = "#dc2626" if is_active_leaf else "#111827"
            ax.text(
                x,
                y,
                "\n".join(label_lines),
                ha="center",
                va="center",
                fontsize=24,
                color=label_color,
                zorder=4,
            )

        title_iter = "init" if iteration is None else f"iteration {int(iteration)}"
        posterior_values = [
            record.get("mu")
            for record in getattr(owner, "_posterior_dataset_by_x", {}).values()
            if isinstance(record, dict) and record.get("mu") is not None
        ]
        best_value_text = f"{float(max(posterior_values)):.6g}" if posterior_values else "GP not fit"
        ax.set_title(
            f"AGVS tree - {title_iter} | Best Posterior Mean: {best_value_text} | "
            "split: seeded random dimension groups when Dim allows",
            fontsize=16,
        )
        ax.axis("off")
        # Dense cohorts need the full canvas; large proportional margins make
        # sibling circles and labels collapse back onto each other.
        ax.margins(x=0.03, y=0.08)
        owner.tree_plot_fig.tight_layout()

        if owner.tree_plot_path:
            directory = os.path.dirname(os.path.abspath(owner.tree_plot_path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            owner.tree_plot_fig.savefig(owner.tree_plot_path, dpi=100)
            self.write_html(iteration=iteration)

        if owner.tree_plot_on and not bool(force):
            try:
                owner.tree_plot_fig.canvas.draw()
                owner.tree_plot_fig.canvas.flush_events()
                plt.pause(owner.tree_plot_pause)
            except Exception:
                pass

    def write_html(self, iteration: Optional[int] = None) -> None:
        owner = self.owner
        if not owner.tree_plot_path or not owner.tree_plot_html_path:
            return

        html_path = os.path.abspath(owner.tree_plot_html_path)
        image_path = os.path.abspath(owner.tree_plot_path)
        directory = os.path.dirname(html_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        try:
            image_src = os.path.relpath(image_path, start=directory or os.getcwd())
        except ValueError:
            image_src = image_path
        image_src = image_src.replace("\\", "/")

        title_iter = "init" if iteration is None else f"iteration {int(iteration)}"
        posterior_values = [
            record.get("mu")
            for record in getattr(owner, "_posterior_dataset_by_x", {}).values()
            if isinstance(record, dict) and record.get("mu") is not None
        ]
        best_value_text = f"{float(max(posterior_values)):.6g}" if posterior_values else "GP not fit"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="2">
  <title>AGVS Live Tree</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #111827; }}
    header {{ padding: 10px 14px; background: #ffffff; border-bottom: 1px solid #d8dde6; }}
    h1 {{ margin: 0; font-size: 16px; }}
    p {{ margin: 4px 0 0; font-size: 12px; color: #4b5563; }}
    img {{ display: block; width: min(100vw, 1800px); height: auto; margin: 0 auto; }}
  </style>
  <script>
    function refreshImage() {{
      const img = document.getElementById("tree");
      img.src = "{image_src}?t=" + Date.now();
    }}
    window.setInterval(refreshImage, 400);
  </script>
</head>
<body>
  <header>
    <h1>AGVS Live Tree - {title_iter} | Best Posterior Mean: {best_value_text}</h1>
    <p>Last update: {stamp}. The tree image refreshes after each saved iteration.</p>
  </header>
  <img id="tree" src="{image_src}?t={time.time():.6f}" alt="AGVS live grouping tree">
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(html)

        if owner.tree_plot_open_browser and not owner._tree_plot_browser_opened:
            try:
                webbrowser.open(f"file://{html_path}", new=1, autoraise=True)
                owner._tree_plot_browser_opened = True
            except Exception:
                pass

"""Tests for ``pair_by_target_columns`` and ``BottleGridColumnFillStrategy``.

The helper produces pairings that fill grid targets column-by-column starting
from the highest-x column.  ``BottleGridColumnFillStrategy`` is a thin
``BottlePickStrategy`` subclass that drives ``initialize_pairings`` through
that helper while preserving bottle drop semantics.
"""

import os
import sys

import numpy as np
import pytest

current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
mock_path = os.path.join(repo_root, "extsMock")
sys.path.insert(0, mock_path)
sys.path.insert(0, repo_root)

from multi_pick_strategy import (
    HORIZONTAL_DROP_QUAT,
    BottleGridColumnFillStrategy,
    pair_by_target_columns,
)
from task_context_base import LightweightObj


def _make_grid_targets(*, rows: int, cols: int, spacing_x: float, spacing_y: float,
                       start_x: float = 0.5, start_y: float = 0.0, z: float = 0.05):
    """Build a flat list of grid targets in row-major order.

    Indices are interleaved across columns (target_0 is row=0,col=0; target_1
    is row=0,col=1; ...) so a "sequential" pairing visits columns interleaved —
    matching the realistic scenario where ``GridPositionGenerator`` shuffles
    its slot order.
    """
    targets = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            pos = np.array([start_x + c * spacing_x,
                            start_y + r * spacing_y,
                            z])
            targets.append(LightweightObj(name=f"target_{k}", position=pos))
            k += 1
    return targets


def _make_picks(n: int):
    return [LightweightObj(name=f"pick_{i}",
                           position=np.array([0.7, 0.1 * i, 0.05]))
            for i in range(n)]


class TestPairByTargetColumns:
    def test_columns_filled_highest_x_first(self):
        # 3 cols x 4 rows, spacing_x negative (matches TableTaskBottles1 grid).
        targets = _make_grid_targets(rows=4, cols=3,
                                     spacing_x=-0.15, spacing_y=0.15,
                                     start_x=0.6, start_y=-0.2)
        picks = _make_picks(12)

        pairings = list(pair_by_target_columns(picks, targets))

        # Pick order should drive each column to completion before the next.
        # The highest-x column is col=0 (spacing_x is negative), targets at
        # indices 0, 3, 6, 9 (one per row in row-major layout).
        first_four_target_idxs = [pairings[i][1] for i in range(4)]
        next_four_target_idxs = [pairings[i][1] for i in range(4, 8)]
        last_four_target_idxs = [pairings[i][1] for i in range(8, 12)]

        # x of every target in the first slice > x of any target in the next.
        def _xs(idxs):
            return [targets[j].get_local_pose()[0][0] for j in idxs]

        assert max(_xs(next_four_target_idxs)) < min(_xs(first_four_target_idxs))
        assert max(_xs(last_four_target_idxs)) < min(_xs(next_four_target_idxs))

    def test_within_column_secondary_sort_y_ascending(self):
        targets = _make_grid_targets(rows=4, cols=2,
                                     spacing_x=-0.15, spacing_y=0.15,
                                     start_x=0.6, start_y=-0.2)
        picks = _make_picks(8)
        pairings = list(pair_by_target_columns(picks, targets))

        # First 4 pairings should be the highest-x column sorted by y ascending.
        first_col_targets = [targets[pairings[i][1]] for i in range(4)]
        ys = [t.get_local_pose()[0][1] for t in first_col_targets]
        assert ys == sorted(ys)

    def test_custom_secondary_key_y_descending(self):
        targets = _make_grid_targets(rows=4, cols=2,
                                     spacing_x=-0.15, spacing_y=0.15,
                                     start_x=0.6, start_y=-0.2)
        picks = _make_picks(8)
        pairings = list(pair_by_target_columns(
            picks, targets, secondary_key=lambda pos: -pos[1]))

        first_col_targets = [targets[pairings[i][1]] for i in range(4)]
        ys = [t.get_local_pose()[0][1] for t in first_col_targets]
        assert ys == sorted(ys, reverse=True)

    def test_surplus_picks_get_none(self):
        targets = _make_grid_targets(rows=2, cols=2,
                                     spacing_x=-0.15, spacing_y=0.15)
        picks = _make_picks(6)
        pairings = list(pair_by_target_columns(picks, targets))

        assert len(pairings) == 6
        # First four picks paired with the four targets (in column-fill order).
        first_targets = [p[1] for p in pairings[:4]]
        assert set(first_targets) == {0, 1, 2, 3}
        # Remaining picks get None.
        assert all(p[1] is None for p in pairings[4:])

    def test_x_tolerance_clusters_drifted_columns(self):
        # Two "columns" whose x values differ by 0.5mm — should be one column.
        targets = [
            LightweightObj(name="t0", position=np.array([0.5, 0.0, 0.05])),
            LightweightObj(name="t1", position=np.array([0.5005, 0.1, 0.05])),
            LightweightObj(name="t2", position=np.array([0.3, 0.0, 0.05])),
            LightweightObj(name="t3", position=np.array([0.3, 0.1, 0.05])),
        ]
        picks = _make_picks(4)
        pairings = list(pair_by_target_columns(picks, targets, x_tolerance=1e-3))

        first_two = {pairings[0][1], pairings[1][1]}
        last_two = {pairings[2][1], pairings[3][1]}
        # First column is the high-x cluster (t0, t1).
        assert first_two == {0, 1}
        assert last_two == {2, 3}


class TestBottleGridColumnFillStrategy:
    def test_initialize_pairings_uses_column_fill_order(self):
        targets = _make_grid_targets(rows=4, cols=3,
                                     spacing_x=-0.15, spacing_y=0.15,
                                     start_x=0.6, start_y=-0.2)
        picks = _make_picks(12)
        strat = BottleGridColumnFillStrategy(pick_objs=picks, target_objs=targets)
        strat.initialize_pairings()

        # The strategy stores pairings as name → name.  Walk the picking order
        # and confirm assigned targets march from highest x to lowest.
        order = strat.picking_order_item_names
        assigned_xs = []
        for name in order:
            tgt_name = strat.pairings_by_pick_name[name]
            tgt = next(t for t in targets if t.name == tgt_name)
            assigned_xs.append(tgt.get_local_pose()[0][0])
        # x is non-increasing across the entire picking order.
        for a, b in zip(assigned_xs, assigned_xs[1:]):
            assert a >= b - 1e-9

    def test_drop_orientation_unchanged(self):
        targets = _make_grid_targets(rows=2, cols=2,
                                     spacing_x=-0.15, spacing_y=0.15)
        picks = _make_picks(4)
        strat = BottleGridColumnFillStrategy(pick_objs=picks, target_objs=targets)
        strat.initialize_pairings()

        orient = strat.get_end_effector_orientation_for_drop("pick_0")
        np.testing.assert_array_equal(orient, HORIZONTAL_DROP_QUAT)

    def test_secondary_key_override_propagates(self):
        targets = _make_grid_targets(rows=4, cols=2,
                                     spacing_x=-0.15, spacing_y=0.15,
                                     start_x=0.6, start_y=-0.2)
        picks = _make_picks(8)
        strat = BottleGridColumnFillStrategy(
            pick_objs=picks, target_objs=targets,
            secondary_key=lambda pos: -pos[1],
        )
        strat.initialize_pairings()

        # First 4 picks should be paired with the high-x column, y descending.
        order = strat.picking_order_item_names
        first_four_ys = []
        for name in order[:4]:
            tgt_name = strat.pairings_by_pick_name[name]
            tgt = next(t for t in targets if t.name == tgt_name)
            first_four_ys.append(tgt.get_local_pose()[0][1])
        assert first_four_ys == sorted(first_four_ys, reverse=True)

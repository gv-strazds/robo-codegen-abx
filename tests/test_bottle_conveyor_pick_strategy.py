"""Tests for BottleConveyorPickStrategy.

Covers:
- Drop orientation is HORIZONTAL_DROP_QUAT.
- stacking_map drives top-down pick ordering (top-layer picks first).
- Regression for the TableTaskBottlesToConveyor bug: with stacking_map
  set, the first pick targets the lowest-Y reachable pad — NOT the +Y-end
  pad that the prior BottlePickStrategy + sequential-pairing combo handed
  out via _reassign_targets_by_picking_order.
- Completed-pick pairing remains stable in the presence of stacking_map.

Pure Python / mock-only — no Isaac Sim required.
"""
import numpy as np

from bottle_conveyor_pick_strategy import BottleConveyorPickStrategy
from multi_pick_strategy import HORIZONTAL_DROP_QUAT


class FakeObj:
    def __init__(self, name, pos=(0.0, 0.5, 0.01)):
        self.name = name
        self.prim_path = f"/World/{name}"
        self._pos = np.asarray(pos, dtype=float)

    def get_world_pose(self):
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def get_local_pose(self):
        return self.get_world_pose()


def _make_targets_descending_to_ascending_y(n=9, y_min=-0.4, y_step=0.1):
    """Mimic ConveyorPositionGenerator: target_0 at lowest Y, target_{n-1} at highest."""
    return [FakeObj(f"target_{i}", pos=(0.0, y_min + i * y_step, 0.01))
            for i in range(n)]


def _make_layered_bottle_picks(n_per_layer=8, lower_z=0.1, upper_z=0.2):
    """Two layers of bottles stacked above each other; same XY per pair."""
    picks = []
    for i in range(n_per_layer):
        picks.append(FakeObj(f"pick_{i}",
                             pos=(0.05 * i, 0.8, lower_z)))
    for i in range(n_per_layer):
        picks.append(FakeObj(f"pick_{n_per_layer + i}",
                             pos=(0.05 * i, 0.8, upper_z)))
    return picks


def _stacking_map_for(picks, n_per_layer):
    """lower → [upper] for each XY-aligned pair."""
    return {picks[i].name: [picks[n_per_layer + i].name]
            for i in range(n_per_layer)}


# ---------------------------------------------------------------------------
# 1. Drop orientation
# ---------------------------------------------------------------------------


def test_drop_orientation_is_horizontal_quat():
    picks = [FakeObj("pick_0", pos=(0.0, 0.8, 0.1))]
    targets = [FakeObj("target_0", pos=(0.0, -0.3, 0.01))]
    s = BottleConveyorPickStrategy(picks, targets)
    quat = s.get_end_effector_orientation_for_drop("pick_0")
    np.testing.assert_allclose(quat, HORIZONTAL_DROP_QUAT)
    # Defensive copy — caller mutating must not affect the constant.
    quat[0] += 1.0
    np.testing.assert_allclose(s.get_end_effector_orientation_for_drop("pick_0"),
                               HORIZONTAL_DROP_QUAT)


# ---------------------------------------------------------------------------
# 2. Stacking drives top-down pick order
# ---------------------------------------------------------------------------


def test_stacking_map_drives_top_down_picking_order():
    picks = _make_layered_bottle_picks(n_per_layer=2)
    targets = _make_targets_descending_to_ascending_y(n=2)
    stacking_map = _stacking_map_for(picks, n_per_layer=2)
    s = BottleConveyorPickStrategy(picks, targets, stacking_map=stacking_map)
    s.initialize_pairings()
    # picks[2] and picks[3] are the upper layer; one of them must come first.
    assert s.get_current_pick_name() in ("pick_2", "pick_3")


# ---------------------------------------------------------------------------
# 3. Regression: first pick targets the lowest-Y pad, NOT the +Y-end pad
# ---------------------------------------------------------------------------


def test_first_pick_targets_lowest_y_pad_with_stacking():
    """Reproduces the TableTaskBottlesToConveyor bug.

    Layout matches the real task: 16 bottles (2 layers x 8) and 9 pads
    on a -Y-flowing conveyor.  target_0 is at the lowest Y (lead pad);
    target_8 is at the highest Y (+Y spawn end).

    With the old BottlePickStrategy + stacking_map, the first picked
    bottle (pick_8, top-layer) ended up paired with target_8 (the +Y
    end).  With BottleConveyorPickStrategy, JIT proximity selection
    returns target_0 (lowest Y, closest to the -Y fall-off edge and to
    the robot).
    """
    n_per_layer = 8
    n_targets = 9
    picks = _make_layered_bottle_picks(n_per_layer=n_per_layer)
    targets = _make_targets_descending_to_ascending_y(n=n_targets)
    stacking_map = _stacking_map_for(picks, n_per_layer=n_per_layer)

    s = BottleConveyorPickStrategy(
        picks, targets,
        conveyor_axis="y", conveyor_sign=-1,
        stacking_map=stacking_map,
    )
    s.initialize_pairings()

    first_pick = s.get_current_pick_name()
    # Top layer picks first.
    assert first_pick in {f"pick_{i}" for i in range(n_per_layer, 2 * n_per_layer)}, (
        f"Expected a top-layer pick to be first, got {first_pick!r}"
    )

    chosen_target = s.get_placing_target_name(first_pick)
    assert chosen_target == "target_0", (
        f"Expected lowest-Y pad 'target_0', got {chosen_target!r} "
        "(regression: bug pre-fix selected 'target_8' — the +Y-end pad)"
    )


# ---------------------------------------------------------------------------
# 4. Simulated cycle loop reaches every target
# ---------------------------------------------------------------------------


def test_simulated_cycles_consume_all_targets_with_stacking():
    """Reproduces the ``--seed 98453`` early-stop bug.

    Walks the strategy through SelectNextPick/MarkPickComplete cycles
    until ``advance_pick_index`` returns ``None``.  Asserts that every
    target gets used.  Pre-fix: only 8 of 9 targets were placed on,
    because the JIT path nulled out other picks' static pairings via
    ``_clear_stale_uncompleted_pairings_to`` and the base
    ``_has_target`` check then falsely reported "no target" for picks
    that JIT would have happily served.  Fix: ``_has_target`` is now
    overridden in ``ConveyorProximityStrategy`` to ask JIT directly.
    """
    n_per_layer = 8
    n_targets = 9
    picks = _make_layered_bottle_picks(n_per_layer=n_per_layer)
    targets = _make_targets_descending_to_ascending_y(n=n_targets)
    stacking_map = _stacking_map_for(picks, n_per_layer=n_per_layer)

    s = BottleConveyorPickStrategy(
        picks, targets,
        conveyor_axis="y", conveyor_sign=-1,
        stacking_map=stacking_map,
    )
    s.initialize_pairings()

    completed_pairs = []
    pick_name = s.get_current_pick_name()
    while pick_name is not None:
        target_name = s.get_placing_target_name(pick_name)
        assert target_name is not None, (
            f"JIT returned no target for selected pick {pick_name!r}"
        )
        completed_pairs.append((pick_name, target_name))
        s.mark_pick_complete(pick_name)
        pick_name = s.advance_pick_index()

    placed_targets = {tgt for _, tgt in completed_pairs}
    assert placed_targets == {f"target_{i}" for i in range(n_targets)}, (
        f"Expected every target to be used; got {placed_targets}. "
        "Regression: pre-fix only 8 of 9 targets were placed on."
    )
    # The first 9 picks (8 top-layer + 1st bottom) should be the ones placed.
    placed_picks = {p for p, _ in completed_pairs}
    expected_placed = (
        {f"pick_{i}" for i in range(n_per_layer, 2 * n_per_layer)}  # top layer
        | {"pick_0"}                                                # first bottom
    )
    assert placed_picks == expected_placed, (
        f"Expected picks {expected_placed} to complete; got {placed_picks}"
    )


# ---------------------------------------------------------------------------
# 5. Completed-pick pairing is stable
# ---------------------------------------------------------------------------


def test_completed_pick_pairing_stable_with_stacking():
    picks = _make_layered_bottle_picks(n_per_layer=2)
    targets = _make_targets_descending_to_ascending_y(n=2)
    stacking_map = _stacking_map_for(picks, n_per_layer=2)
    s = BottleConveyorPickStrategy(
        picks, targets,
        conveyor_axis="y", conveyor_sign=-1,
        stacking_map=stacking_map,
    )
    s.initialize_pairings()

    first_pick = s.get_current_pick_name()
    chosen = s.get_placing_target_name(first_pick)
    s.latch_current_target(first_pick)
    s.mark_pick_complete(first_pick)

    # The completed pick keeps its pairing forever; JIT must not rewrite it
    # even if a more-urgent pad later appears.
    assert s.get_placing_target_name(first_pick) == chosen
    # And the latch was cleared on mark_pick_complete.
    assert first_pick not in s.latched_target_by_pick

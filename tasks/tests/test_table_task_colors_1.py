import types

import pytest


def test_color_matching_pairs_and_order(monkeypatch):
    # Import module and class under test
    from multi_pick_strategy import ColorMatchStrategy

    # Stub has_color to consult simple dicts we define per-prim
    def stub_has_color(prim, color_name: str) -> bool:
        colors = getattr(prim, "_colors", set())
        return color_name.lower() in {c.lower() for c in colors}

    # Minimal prim-like object
    class FakePrim:
        def __init__(self, name, colors):
            self.name = name
            self._colors = set(colors)

        def get_local_pose(self):
            import numpy as np
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def get_world_pose(self):
            import numpy as np
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    # Build picks and targets with colors
    picks = [
        FakePrim("pick_red", {"red"}),
        FakePrim("pick_blue1", {"blue"}),
        FakePrim("pick_green", {"green"}),
        FakePrim("pick_blue2", {"blue"}),
    ]
    targets = [
        FakePrim("tgt_red", {"red"}),
        FakePrim("tgt_green", {"green"}),
    ]

    # Create strategy with color matching
    strategy = ColorMatchStrategy(
        pick_objs=picks,
        target_objs=targets,
        color_palette=["red", "green", "blue", "yellow"],
        has_color_fn=stub_has_color,
    )

    # Perform color-based pairing
    pairings = list(strategy.pair_picks_with_targets())
    # Expected: red->red, blue1->None, green->green, blue2->None
    assert pairings[0][1] is not None
    assert pairings[1][1] is None
    assert pairings[2][1] is not None
    assert pairings[3][1] is None

    # Initialize pairings (which also filters picking order to matched only)
    strategy.initialize_pairings()

    # Verify unmatched picks were excluded from the picking order
    resulting_order = strategy.picking_order_item_names
    assert "pick_red" in resulting_order
    assert "pick_green" in resulting_order
    assert "pick_blue1" not in resulting_order
    assert "pick_blue2" not in resulting_order

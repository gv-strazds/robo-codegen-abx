"""BottleConveyorPickStrategy: ConveyorProximityStrategy with bottle drop orientation.

Combines:
- JIT urgency-aware target selection from ConveyorProximityStrategy
  (lowest-Y reachable unoccupied pad each tick).
- Bottle-on-side drop orientation (HORIZONTAL_DROP_QUAT), matching
  BottlePickStrategy / DynamicTopPickStrategy.
- Optional stacking_map for top-down pick ordering from a layered bin
  (honoured by the base MultiPickStrategy via _apply_stacking_order).

Use this instead of BottlePickStrategy whenever bottle targets are pads
on a moving conveyor.  The sequential default pairing in BottlePickStrategy
combined with stacking_map reassignment puts the first picked bottle on
the wrong (far +Y) pad — see the BottleConveyorPickStrategy regression
tests for details.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from conveyor_proximity_strategy import ConveyorProximityStrategy
from multi_pick_strategy import HORIZONTAL_DROP_QUAT

logger = logging.getLogger(__name__)


class BottleConveyorPickStrategy(ConveyorProximityStrategy):
    """Bottle-flavoured ConveyorProximityStrategy."""

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        *,
        conveyor_axis: str = "y",
        conveyor_sign: int = -1,
        conveyor_end: Optional[float] = None,
        stacking_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(
            pick_objs,
            target_objs,
            conveyor_axis=conveyor_axis,
            conveyor_sign=conveyor_sign,
            conveyor_end=conveyor_end,
        )
        # ConveyorProximityStrategy.__init__ does not forward stacking_map;
        # overwrite the empty dict that MultiPickStrategy.__init__ created.
        if stacking_map:
            self._stacking_map = dict(stacking_map)

    def get_end_effector_orientation_for_drop(
        self, pick_name: str, target_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        return HORIZONTAL_DROP_QUAT.copy()

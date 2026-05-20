"""ConveyorProximityStrategy: just-in-time, urgency-aware target selection.

For conveyor tasks where unoccupied targets can fall off the belt before
a pick is placed on them, the default sequential pairing policy of
``MultiPickStrategy`` is sub-optimal — it can pair a pick with a
newly-arrived (safe) target while an older (about to fall off) target
is ignored.

``ConveyorProximityStrategy`` replaces the persistent pairing model
with just-in-time selection: whenever the current pick needs a target
(``get_placing_target_name``), the strategy scans all reachable,
unoccupied targets and returns the one closest to the conveyor edge.
This naturally re-pairs every tick and maximises the chance that an
imminent drop-off is placed on before it is lost.

To avoid RMPFlow chasing a moving goal mid-descent, the *selected*
target is *latched* for the duration of the place phase (latch is set
by the ``LatchPlacementTarget`` BT behaviour just after grasp and
cleared on ``mark_pick_complete``).  If the latched target becomes
permanently unreachable mid-place, the latch is cleared, JIT
re-selection runs, and the latch is renewed to the new winner — so the
carried item is smoothly redirected rather than dropped.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from multi_pick_strategy import MultiPickStrategy

logger = logging.getLogger(__name__)


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class ConveyorProximityStrategy(MultiPickStrategy):
    """JIT pairing: select the lowest-distance-to-edge reachable target.

    Args:
        pick_objs: List of pick (source) objects.
        target_objs: List of target (destination) objects on the conveyor.
        conveyor_axis: World-frame axis the conveyor moves along, one of
            ``"x"``, ``"y"``, ``"z"``.  Defaults to ``"y"``.
        conveyor_sign: ``-1`` if targets flow toward decreasing coordinate
            (edge in the negative direction — the common case for the UR10
            conveyor setup), ``+1`` if they flow toward increasing
            coordinate.  Used to define "urgency" — a target closer to the
            edge has a smaller distance-to-edge score.
        conveyor_end: Optional coordinate of the belt edge along
            ``conveyor_axis``.  Currently used only for diagnostics (log
            "distance to edge" in re-pair messages); selection only needs
            the coordinate's rank among candidates.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        *,
        conveyor_axis: str = "y",
        conveyor_sign: int = -1,
        conveyor_end: Optional[float] = None,
    ) -> None:
        super().__init__(pick_objs, target_objs)
        if conveyor_axis not in _AXIS_INDEX:
            raise ValueError(
                f"conveyor_axis must be one of 'x'/'y'/'z', got {conveyor_axis!r}"
            )
        if conveyor_sign not in (-1, 1):
            raise ValueError(f"conveyor_sign must be -1 or +1, got {conveyor_sign}")
        self._axis_idx = _AXIS_INDEX[conveyor_axis]
        self._conveyor_sign = conveyor_sign
        self._conveyor_end = conveyor_end
        self._latched_target_by_pick: Dict[str, str] = {}

    # -------------------------------------------------------------------
    # JIT selection core
    # -------------------------------------------------------------------

    def _urgency_key(self, coordinate: float) -> float:
        """Return a sort key where smaller = more urgent (closer to edge).

        With ``conveyor_sign == -1`` (edge in negative direction), smaller
        coordinate is more urgent, so key == coordinate.
        With ``conveyor_sign == +1`` (edge in positive direction), larger
        coordinate is more urgent, so key == -coordinate.
        """
        return -self._conveyor_sign * coordinate

    def _jit_select(
        self, pick_name: str, exclude_latched_of: bool = True
    ) -> Optional[str]:
        """Return the target name closest to the conveyor edge, or None.

        Filters out permanently-unreachable targets, targets occupied by
        completed picks or frozen passing snapshots, and targets currently
        latched to other in-flight picks (so two picks in flight don't
        race for the same target).  Queries ``is_target_reachable`` so
        the ``target_reachable_fn`` predicate is honoured.
        """
        occupied = self._currently_occupied_target_names()
        latched_by_others: set = set()
        if exclude_latched_of:
            latched_by_others = {
                tgt_name for other, tgt_name in self._latched_target_by_pick.items()
                if other != pick_name and tgt_name is not None
            }

        best_key: Optional[float] = None
        best_name: Optional[str] = None
        for t in self._target_objs:
            tgt_name = t.name
            if tgt_name in self._permanently_unreachable_targets:
                continue
            if tgt_name in occupied:
                continue
            if tgt_name in latched_by_others:
                continue
            if not self.is_target_reachable(tgt_name):
                continue
            try:
                pos, _ = t.get_world_pose()
            except Exception:
                continue
            coord = float(pos[self._axis_idx])
            key = self._urgency_key(coord)
            if best_key is None or key < best_key:
                best_key = key
                best_name = tgt_name
        return best_name

    def _clear_stale_uncompleted_pairings_to(
        self, target_name: str, except_pick: str,
    ) -> None:
        """Null out uncompleted picks currently paired to ``target_name``.

        JIT selection calls ``get_placing_target_name`` once per candidate
        pick it considers, and each call writes ``_pairings_by_pick_name``.
        Without this cleanup, every candidate the strategy cycled through
        leaves a stale entry pointing at the same target — a violation of
        the invariant that each target is claimed by at most one
        uncompleted pick.  Completed picks are preserved: their pairing
        records where the pick was actually placed, which downstream
        verification and ``_currently_occupied_target_names`` rely on.
        """
        for other_name, other_tgt in self._pairings_by_pick_name.items():
            if (other_tgt == target_name
                    and other_name != except_pick
                    and other_name not in self._completed_picks):
                self._pairings_by_pick_name[other_name] = None

    # -------------------------------------------------------------------
    # Overrides
    # -------------------------------------------------------------------

    def add_incremental_targets(self, new_objs: list) -> None:
        """Append new targets without rebuilding the pairing map.

        The base ``MultiPickStrategy.add_incremental_targets`` calls
        ``recompute_pairings``, which rebuilds ``_pairings_by_pick_name``
        from the sequential default.  For JIT-selecting strategies this
        is harmful: it clobbers completed picks' pairings (which record
        the target they were actually placed on), corrupting the
        ``occupied`` set and causing the next pick's JIT to lock onto a
        target that's already filled.

        JIT selection automatically picks up new targets on the next
        ``get_placing_target_name`` call, so no pairing rebuild is
        required — we only need to extend ``_target_objs`` and reset
        the ``_targets_exhausted`` latch.
        """
        self._extend_target_objs(new_objs)
        self._targets_exhausted = False

    def get_placing_target_name(self, pick_name: str) -> Optional[str]:
        """JIT target selection with latch-based stability.

        - Completed picks: return the stable pairing (the target actually
          placed on).  Never re-select — other callers (e.g.
          ``task_controller._build_pick_observations``) query this every
          tick for all picks, and rewriting a completed pick's pairing
          would pollute the ``occupied`` set used by subsequent picks'
          JIT selection.
        - If a latched target exists and is still reachable/unoccupied →
          return it (mid-place stability).
        - If the latched target has become invalid, clear the latch,
          run JIT re-selection, and re-latch to the new winner so the
          place-phase behaviours keep tracking a stable target.
        - Otherwise (no latch — pre-grasp), run JIT every call for
          maximum responsiveness to new arrivals / losses.
        """
        if pick_name in self._completed_picks:
            return self._pairings_by_pick_name.get(pick_name)

        latched_name = self._latched_target_by_pick.get(pick_name)
        if latched_name is not None:
            still_valid = (
                latched_name not in self._permanently_unreachable_targets
                and not self._is_target_occupied(latched_name, exclude_pick=pick_name)
                and self.is_target_reachable(latched_name)
            )
            if still_valid:
                self._clear_stale_uncompleted_pairings_to(latched_name, pick_name)
                self._pairings_by_pick_name[pick_name] = latched_name
                return latched_name
            # Latched target died — clear, re-select, re-latch
            old_latched = latched_name
            del self._latched_target_by_pick[pick_name]
            new_name = self._jit_select(pick_name)
            if new_name is None:
                self._pairings_by_pick_name[pick_name] = None
                logger.info(
                    "Latched target '%s' for '%s' became invalid; "
                    "no reachable replacement available",
                    old_latched, pick_name,
                )
                return None
            self._latched_target_by_pick[pick_name] = new_name
            self._clear_stale_uncompleted_pairings_to(new_name, pick_name)
            self._pairings_by_pick_name[pick_name] = new_name
            logger.info(
                "Re-latched '%s': '%s' -> '%s' (previous latch invalid)",
                pick_name, old_latched, new_name,
            )
            return new_name

        # No latch (pre-grasp or between picks): fresh JIT selection
        prev_name = self._pairings_by_pick_name.get(pick_name)
        new_name = self._jit_select(pick_name)
        if new_name is None:
            self._pairings_by_pick_name[pick_name] = None
            return None
        self._clear_stale_uncompleted_pairings_to(new_name, pick_name)
        self._pairings_by_pick_name[pick_name] = new_name
        if prev_name != new_name:
            logger.info(
                "Proximity re-pair: '%s' '%s' -> '%s'",
                pick_name, prev_name, new_name,
            )
        return new_name

    def _try_retarget(
        self, pick_name: str, old_target_name: Optional[str],
    ) -> Optional[str]:
        """JIT-based retarget used by ``advance_pick_index`` / ``_has_target``.

        Returns the most-urgent reachable alternative, or None.  The
        ``old_target_name`` argument is ignored — the JIT scan already
        excludes permanently unreachable targets, which is what the base
        class uses ``old_target_name`` to skip.
        """
        return self._jit_select(pick_name)

    def _has_target(self, pick_name: str) -> bool:
        """Override base check to query JIT instead of the static pairing dict.

        The base ``MultiPickStrategy._has_target`` consults
        ``_pairings_by_pick_name``, which the JIT path actively invalidates
        via ``_clear_stale_uncompleted_pairings_to`` whenever another pick
        claims a target.  Under sequential init + stacking reassignment
        that leaves later picks with ``None`` pairings and makes
        ``_scan_for_available_pick`` falsely skip them, even though JIT
        would gladly find them a target.  Asking JIT directly is the
        single source of truth for "is a target available for this pick".
        """
        return self._jit_select(pick_name) is not None

    # -------------------------------------------------------------------
    # Latch API (overrides base no-ops)
    # -------------------------------------------------------------------

    def latch_current_target(self, pick_name: str) -> None:
        """Pin *pick_name*'s current target for the duration of the place phase.

        Called by ``LatchPlacementTarget`` (BT behaviour) as the first step
        of the place phase, after the gripper has closed.  Uses the most
        recent JIT selection as the latch value.  If no current selection
        exists, runs a fresh JIT scan.
        """
        tgt_name = self._pairings_by_pick_name.get(pick_name)
        if tgt_name is None:
            tgt_name = self._jit_select(pick_name)
        if tgt_name is None:
            logger.warning(
                "latch_current_target('%s'): no reachable target to latch",
                pick_name,
            )
            return
        self._latched_target_by_pick[pick_name] = tgt_name
        self._clear_stale_uncompleted_pairings_to(tgt_name, pick_name)
        self._pairings_by_pick_name[pick_name] = tgt_name
        logger.debug("Latched '%s' -> '%s'", pick_name, tgt_name)

    def clear_target_latch(self, pick_name: str) -> None:
        self._latched_target_by_pick.pop(pick_name, None)

    def clear_all_target_latches(self) -> None:
        self._latched_target_by_pick.clear()

    # -------------------------------------------------------------------
    # Introspection helpers (for tests / diagnostics)
    # -------------------------------------------------------------------

    @property
    def latched_target_by_pick(self) -> Dict[str, str]:
        return dict(self._latched_target_by_pick)

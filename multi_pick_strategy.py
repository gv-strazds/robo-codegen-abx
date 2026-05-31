"""MultiPickStrategy: encapsulates pick-to-target pairing logic.

Owns pairing computation, pick iteration, target/placing resolution,
and completion tracking.  TaskContext delegates to the strategy for
these concerns while remaining the facade used by py_trees behaviours.

Strategy subclasses override `pair_picks_with_targets()` (and optionally
`valid_targets_for_pick` / `placement_constraints_satisfied`) to
implement colour-matching, type-based sorting, etc.
"""
import logging
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from typing_extensions import override

import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat

logger = logging.getLogger(__name__)


# Drop orientation shared by strategies that place items on their side (e.g.
# bottles laid horizontally on a conveyor): pi/2 rotation around X.  Used by
# ``BottlePickStrategy`` and ``DynamicTopPickStrategy`` instead of duplicating
# the literal quaternion in each override.
HORIZONTAL_DROP_QUAT = euler_angles_to_quat(np.array([np.pi / 2, 0, 0]))


# ---------------------------------------------------------------------------
# Stacking utilities
# ---------------------------------------------------------------------------


def compute_stacking_map(
    pick_objs, xy_tolerance: float = 0.01
) -> Dict[str, List[str]]:
    """Compute stacking relationships from pick object positions.

    Groups objects by XY proximity (within tolerance). Within each column,
    sorts by Z and maps each item to the item directly above it.

    Args:
        pick_objs: List of pick objects with ``get_local_pose()`` method.
        xy_tolerance: Maximum XY distance to consider two items in the same column.

    Returns:
        Dict mapping pick_name → [names of items directly above it].
        Items with nothing above them do not appear as keys.
    """
    # Collect positions
    positions = {}
    for obj in pick_objs:
        pos, _ = obj.get_local_pose()
        positions[obj.name] = pos

    # Group into columns by XY proximity
    columns: List[List[tuple]] = []  # each column: [(name, pos), ...]
    for name, pos in positions.items():
        xy = pos[:2]
        placed = False
        for column in columns:
            ref_xy = column[0][1][:2]
            if np.linalg.norm(xy - ref_xy) <= xy_tolerance:
                column.append((name, pos))
                placed = True
                break
        if not placed:
            columns.append([(name, pos)])

    # Build stacking map: lower item → [item directly above it]
    stacking_map: Dict[str, List[str]] = {}
    for column in columns:
        if len(column) <= 1:
            continue
        # Sort by Z ascending (lowest first)
        column.sort(key=lambda entry: entry[1][2])
        for i in range(len(column) - 1):
            lower_name = column[i][0]
            upper_name = column[i + 1][0]
            stacking_map.setdefault(lower_name, []).append(upper_name)

    return stacking_map


def pair_by_target_columns(
    pick_objs,
    target_objs,
    *,
    x_tolerance: float = 1e-3,
    secondary_key: Callable[[np.ndarray], float] = lambda pos: pos[1],
) -> Iterator[Tuple[int, Optional[int]]]:
    """Yield pick→target pairings that fill grid columns highest-x first.

    Targets are grouped into columns by x (within ``x_tolerance``), columns
    are sorted by x descending, and within each column targets are sorted by
    ``secondary_key(position)`` (default: y ascending). Picks are then paired
    sequentially against the resulting target order; surplus picks yield
    ``(i, None)``.

    Args:
        pick_objs: List of pick objects (only their count is used here).
        target_objs: List of target objects with ``get_local_pose()`` method.
        x_tolerance: Maximum x distance to consider two targets in the same column.
        secondary_key: Sort key for within-column ordering. Default y ascending;
            pass ``lambda pos: -pos[1]`` for y descending.
    """
    # Collect (target_idx, position) and bucket into columns by x proximity.
    target_positions: List[Tuple[int, np.ndarray]] = []
    for j, tgt in enumerate(target_objs):
        pos, _ = tgt.get_local_pose()
        target_positions.append((j, pos))

    columns: List[List[Tuple[int, np.ndarray]]] = []
    for entry in target_positions:
        _, pos = entry
        placed = False
        for col in columns:
            ref_x = col[0][1][0]
            if abs(pos[0] - ref_x) <= x_tolerance:
                col.append(entry)
                placed = True
                break
        if not placed:
            columns.append([entry])

    # Sort columns by representative x descending, then targets within each
    # column by secondary_key(position).
    columns.sort(key=lambda col: -col[0][1][0])
    sorted_target_indices: List[int] = []
    for col in columns:
        col.sort(key=lambda entry: secondary_key(entry[1]))
        sorted_target_indices.extend(idx for idx, _ in col)

    n = min(len(pick_objs), len(sorted_target_indices))
    for i in range(n):
        yield (i, sorted_target_indices[i])
    for i in range(n, len(pick_objs)):
        yield (i, None)


def build_bin_geometry_check(bin_geometry: dict) -> Callable:
    """Build a spatial check fn from a bin_geometry dict.

    Returns a callable with signature
    ``(pick_obj, target_obj=None, bb_cache=None, obj_scale=None) -> bool``
    that checks whether *pick_obj* is within the bin bounds using
    ``is_within_box_geometry()``.
    """
    bg = dict(bin_geometry)

    def _bin_check(pick_obj, target_obj=None, bb_cache=None, obj_scale=None,
                   log_failure=False):
        from task_verification import is_within_box_geometry

        kwargs = {}
        if "z_tol" in bg:
            kwargs["z_tol"] = bg["z_tol"]
        return is_within_box_geometry(
            pick_obj,
            box_center_xy=bg["center_xy"],
            box_inner_size=bg["inner_size"],
            box_floor_z=bg["floor_z"],
            box_height=bg["height"],
            bb_cache=bb_cache,
            obj_scale=obj_scale,
            log_failure=log_failure,
            **kwargs,
        )

    return _bin_check


class MultiPickStrategy:
    """Default sequential pick-to-target pairing strategy.

    Args:
        pick_objs: List of pick (source) objects.
        target_objs: List of target (destination) objects.
    """

    # After this many ``defer_pick`` calls for the same pick *without* an
    # intervening ``mark_pick_complete``, the pick is promoted to
    # permanently-unreachable.  Bounds livelock when an item has drifted
    # past the kinematic envelope (e.g. tipped onto its side near the
    # workspace edge) and no successful neighbour completion is going to
    # change that.  Threshold of 3 preserves the legitimate "deferred and
    # then unblocked by a sibling completion" cases (e.g. stacking) — the
    # counter resets on any successful ``mark_pick_complete``.
    MAX_DEFERS_BEFORE_PERMANENT: int = 3

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        stacking_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._pick_objs = pick_objs
        self._target_objs = target_objs
        self._stacking_map: Dict[str, List[str]] = stacking_map or {}

        # Pairing state — populated by initialize_pairings().  The single
        # source of truth is the name→name dict; index translations happen
        # at call sites that genuinely need them.
        self._pairings_by_pick_name: Dict[str, Optional[str]] = {}
        self._picking_order_item_names: List[str] = []

        # Pick iteration state
        self._current_pick_index: int = 0
        self._completed_picks: set = set()
        self._targets_exhausted: bool = False
        self._task_finished: bool = False

        # Incremental generation flags
        self._more_items_expected: bool = False
        self._more_targets_expected: bool = False

        # Frozen-target callback: the task verifier owns the per-pick frozen
        # snapshot state; the strategy queries it through this thunk when
        # picking a retarget so it skips targets already claimed by a passing
        # frozen check.  Defaults to "no frozen targets" (single-cycle tasks).
        self._frozen_target_names_fn: Optional[Callable[[], set]] = None

        # Target reachability state.  When a target_reachable_fn is set,
        # targets that fail the check are excluded from pairing.  Targets
        # that were once reachable but later fail (e.g. fell off conveyor)
        # are marked permanently unreachable and skipped by advance_pick_index.
        self._target_reachable_fn: Optional[Callable] = None
        self._target_was_reachable: Dict[str, bool] = {}
        self._permanently_unreachable_targets: set = set()

        # Pick-commit stash (race mitigation for pick latching). Set by
        # CortexMoveToPick on each successful command compute; consumed
        # by LatchCurrentPick after grasp; cleared on mark_pick_complete.
        self._committed_pick_name: Optional[str] = None

        # Temporarily-excluded picks.  When a pick attempt fails its
        # retry budget (see DeferPickAndRelease in the cortex BT) it is
        # added here and skipped by selection for the current pass.
        # Cleared by mark_pick_complete (any successful completion
        # changes the scene, so deferred picks get another chance) and
        # by the second-chance pass in _scan_for_available_pick
        # (fallback livelock guard when all candidates are deferred).
        self._deferred_picks: set = set()

        # Per-pick consecutive-defer counter.  Incremented by ``defer_pick``
        # and reset by ``mark_pick_complete`` (any successful completion
        # changes the scene, so prior defers don't count toward livelock).
        # When a pick's count reaches ``MAX_DEFERS_BEFORE_PERMANENT`` it is
        # promoted to ``_permanently_unreachable_picks``.  Survives the
        # second-chance pass in ``_scan_for_available_pick`` (which clears
        # ``_deferred_picks`` but not this dict) so a livelock between two
        # unreachable picks can't reset the counter forever.
        self._defer_counts: Dict[str, int] = {}

        # Permanently-excluded picks.  Mirrors _permanently_unreachable_targets
        # but for picks: items that physics has put out of reach for the rest
        # of the run (e.g. dropped below the pick-side z-floor after falling
        # off the conveyor).  Unlike _deferred_picks, this set is NOT cleared
        # by mark_pick_complete or by the second-chance pass — once a pick is
        # marked permanent it stays permanent until reset().
        self._permanently_unreachable_picks: set = set()

        # Cycles since the last successful mark_pick_complete.  Bumped by
        # CheckCycleProgress at the head of do_one_pick_place; reset on
        # mark_pick_complete.  Used as a livelock safety net when neither
        # the per-pick z-floor permanent-flag nor the strategy's normal
        # exit conditions catch a no-progress loop.
        self._cycles_since_last_completion: int = 0

        # Build name -> obj lookup tables
        self._pick_objs_by_name = {obj.name: obj for obj in self._pick_objs}
        self._target_objs_by_name = {obj.name: obj for obj in self._target_objs}

    # -------------------------------------------------------------------
    # Incremental pick generation
    # -------------------------------------------------------------------

    @property
    def more_items_expected(self) -> bool:
        """True when more pick items will be added via incremental generation."""
        return self._more_items_expected

    @more_items_expected.setter
    def more_items_expected(self, value: bool) -> None:
        self._more_items_expected = value

    @property
    def more_targets_expected(self) -> bool:
        """True when more target objects will be added via incremental generation."""
        return self._more_targets_expected

    @more_targets_expected.setter
    def more_targets_expected(self, value: bool) -> None:
        self._more_targets_expected = value

    def _extend_pick_objs(self, additional_objs: list) -> int:
        """Append objects to the pick list and update the name lookup.

        Returns the starting index of the newly added picks.
        """
        start_idx = len(self._pick_objs)
        for obj in additional_objs:
            self._pick_objs.append(obj)
            self._pick_objs_by_name[obj.name] = obj
        return start_idx

    def add_incremental_picks(self, new_objs: list) -> None:
        """Add new pick objects and recompute pairings to include them.

        Preserves completed picks and current pick position.  New items
        are appended at the end of the picking order.  If stacking is
        enabled, the stacking map is recomputed from the updated pick
        list before rebuilding pairings.
        """
        self._extend_pick_objs(new_objs)
        if self._stacking_map:
            self._stacking_map = compute_stacking_map(self._pick_objs)
        self.recompute_pairings(preserve_current=True, respect_completed=True)
        if self._stacking_map:
            self._apply_stacking_order()
            self._reassign_targets_by_picking_order()

    # -------------------------------------------------------------------
    # Target-list manipulation (for stacking strategies)
    # -------------------------------------------------------------------

    def _extend_target_objs(self, additional_objs: list) -> int:
        """Append objects to the target list and update the name lookup.

        Returns the starting index of the newly added targets.
        """
        start_idx = len(self._target_objs)
        for obj in additional_objs:
            self._target_objs.append(obj)
            self._target_objs_by_name[obj.name] = obj
        return start_idx

    def add_incremental_targets(self, new_objs: list) -> None:
        """Add new target objects and recompute pairings to include them.

        Mirrors ``add_incremental_picks``: appends to the target list, clears
        the ``targets_exhausted`` latch (new targets may unblock picks that
        had been paired with None), and recomputes pairings while preserving
        the current pick and already-completed picks.
        """
        self._extend_target_objs(new_objs)
        self._targets_exhausted = False
        self.recompute_pairings(preserve_current=True, respect_completed=True)
        if self._stacking_map:
            self._apply_stacking_order()
            self._reassign_targets_by_picking_order()

    def _truncate_target_objs(self, count: int) -> None:
        """Remove targets beyond index *count*, cleaning up the name lookup."""
        removed = self._target_objs[count:]
        del self._target_objs[count:]
        for obj in removed:
            self._target_objs_by_name.pop(obj.name, None)

    # -------------------------------------------------------------------
    # Pairing computation (overridable)
    # -------------------------------------------------------------------

    def initialize_pairings(self) -> None:
        """Compute pairings, build lookup dicts, set default picking order."""
        try:
            pairings = list(self.pair_picks_with_targets())
        except Exception:
            pairings = list(self._default_sequential_pairings())

        self._pairings_by_pick_name = {}
        for pick_idx, tgt_idx in pairings:
            pick_name = self._pick_objs[pick_idx].name
            tgt_name = (
                self._target_objs[tgt_idx].name if tgt_idx is not None else None
            )
            self._pairings_by_pick_name[pick_name] = tgt_name

        # Default picking order: all pick names in pairing order
        self._picking_order_item_names = [
            self._pick_objs[p_idx].name for p_idx, _ in pairings
        ]

        self._apply_stacking_order()
        self._reassign_targets_by_picking_order()

    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Overridable strategy for pairing picks to targets.

        Yields (pick_index, Optional[target_index]) tuples.
        Default: sequential pairing.
        """
        yield from self._default_sequential_pairings()

    def _default_sequential_pairings(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Yield sequential pairings: pick[i] -> target[i] or None."""
        n_picks = len(self._pick_objs)
        n_targets = len(self._target_objs)
        n = min(n_picks, n_targets)
        for i in range(n):
            yield (i, i)
        for i in range(n, n_picks):
            yield (i, None)

    # -------------------------------------------------------------------
    # Pick iteration
    # -------------------------------------------------------------------

    def get_current_pick_name(self) -> Optional[str]:
        if self._targets_exhausted:
            return None
        # When every pick has been completed or flagged permanently
        # unreachable, there is no "current pick" — return None even if
        # the cursor still points at the last-returned (now-completed)
        # pick.  Mirrors the historical "cursor walked off the end"
        # behaviour without depending on cursor over-advancement during
        # incremental-spawn idle ticks (see ``all_picks_done``).
        if self.all_picks_done:
            return None
        if self._current_pick_index >= len(self._picking_order_item_names):
            if self._stacking_map:
                return self._scan_for_available_pick()
            return None
        name = self._picking_order_item_names[self._current_pick_index]
        if self._stacking_map:
            # Skip if stacking-blocked or has no target.
            # Note: do NOT check `name in self._completed_picks` here —
            # get_current_pick_name() is called multiple times per tick
            # (e.g. by ContextMonitor), and the scan side-effect would
            # double-advance the index before advance_pick_index() runs.
            if not self._is_pick_available(name) or not self._has_target(name):
                return self._scan_for_available_pick()
        return name

    def advance_pick_index(self) -> Optional[str]:
        """Increment current pick index. Return new pick name or None.

        Skips temporarily-deferred picks (see ``defer_pick``) and
        permanently-unreachable picks (see
        ``mark_pick_permanently_unreachable``) in addition to
        permanently-unreachable *targets*.  When the tail of the list is
        all-deferred, falls through to ``_scan_for_available_pick`` so
        the second-chance pass can fire.

        The cursor is only consumed when a candidate is actually returned;
        repeated calls during the "waiting for incremental items" idle
        period therefore do NOT race the cursor past
        ``len(_picking_order_item_names)``.  This protects ``all_picks_done``
        from a false-positive when an incremental scheduler adds the last
        item after the BT has been idling for many ticks.
        """
        if self._stacking_map:
            # Always use scan (with wrap-around) when stacking is active —
            # the picking order may not align with source stacking constraints,
            # so previously-blocked items may now be available behind the cursor.
            return self._scan_for_available_pick()
        # Skip picks that have no viable target (permanently unreachable
        # AND no re-pairing possible) *and* picks temporarily deferred
        # this pass *and* picks permanently flagged unreachable.  Don't
        # skip for unreachable targets if _try_retarget can find an
        # alternative — let CheckTargetAvailable handle that.
        n = len(self._picking_order_item_names)
        idx = self._current_pick_index + 1
        while idx < n:
            name = self._picking_order_item_names[idx]
            if name in self._completed_picks:
                idx += 1
                continue
            if name in self._permanently_unreachable_picks:
                logger.debug(
                    "advance_pick_index: skipping '%s' (permanently unreachable)", name,
                )
                idx += 1
                continue
            if name in self._deferred_picks:
                logger.debug(
                    "advance_pick_index: skipping '%s' (deferred this pass)", name,
                )
                idx += 1
                continue
            tgt_name = self._pairings_by_pick_name.get(name)
            if tgt_name is not None and tgt_name in self._permanently_unreachable_targets:
                # Check if re-pairing is possible before skipping
                alt = self._try_retarget(name, tgt_name)
                if alt is None:
                    logger.debug(
                        "advance_pick_index: skipping '%s' (target permanently unreachable, "
                        "no alternative)", name,
                    )
                    idx += 1
                    continue
                # Re-pairing possible — don't skip, let CheckTargetAvailable handle
            self._current_pick_index = idx
            return name
        # Tail exhausted.  Preserve legacy no-wrap-around semantics in
        # the common case; fall through to the scan only when deferred
        # picks are present so the second-chance pass can fire.
        # Note: we deliberately do NOT advance _current_pick_index past the
        # last valid pick — see the docstring for why.
        if self._deferred_picks:
            return self._scan_for_available_pick()
        return None

    @property
    def all_picks_done(self) -> bool:
        """True when every pick in the order is completed or permanently unreachable.

        Defined by set membership rather than cursor position so it stays
        correct under (a) incremental growth of ``_picking_order_item_names``
        via ``add_incremental_picks`` and (b) non-sequential pick selection
        in JIT strategies that may complete picks out of cursor order.

        Implemented as an O(1) cardinality check: ``mark_pick_complete`` and
        ``mark_pick_permanently_unreachable`` are the only writers and only
        accept names already in the picking order, so under the invariant
        ``_completed_picks ∪ _permanently_unreachable_picks ⊆ set(names)``
        the union's cardinality matches the (duplicate-free) picking-order
        length iff every name is covered.  The empty-picking-order case
        is vacuously True (``0 == 0``) — matches the historical positional
        ``cursor=0 >= len=0`` semantic.
        """
        return (
            len(self._completed_picks | self._permanently_unreachable_picks)
            == len(self._picking_order_item_names)
        )

    @property
    def picking_order_item_names(self) -> List[str]:
        return self._picking_order_item_names

    @property
    def targets_exhausted(self) -> bool:
        return self._targets_exhausted

    @targets_exhausted.setter
    def targets_exhausted(self, value: bool) -> None:
        self._targets_exhausted = value

    @property
    def task_finished(self) -> bool:
        return self._task_finished

    @task_finished.setter
    def task_finished(self, value: bool) -> None:
        self._task_finished = value

    # -------------------------------------------------------------------
    # Stacking constraint helpers
    # -------------------------------------------------------------------

    def _has_target(self, pick_name: str) -> bool:
        """Return True if *pick_name* has (or can get) a reachable target."""
        tgt_name = self._pairings_by_pick_name.get(pick_name)
        if tgt_name is None:
            return False
        if tgt_name not in self._permanently_unreachable_targets:
            return True
        # Current target is unreachable — check if re-pairing is possible
        return self._try_retarget(pick_name, tgt_name) is not None

    def _is_pick_available(self, pick_name: str) -> bool:
        """Check whether all items directly above *pick_name* have been completed."""
        if not self._stacking_map:
            return True
        above = self._stacking_map.get(pick_name, [])
        return all(name in self._completed_picks for name in above)

    def _scan_for_available_pick(self) -> Optional[str]:
        """Scan forward from current index to find the next available pick.

        Skips items that are completed, permanently unreachable,
        stacking-blocked, have no target, or have been temporarily
        deferred via ``defer_pick``.  If the forward scan exhausts the
        list, wraps around from index 0 to find previously-blocked items
        that are now available.  If that too fails *and* at least one
        pick was skipped solely because it is deferred, clears the
        deferred set and scans once more (the "second-chance" pass;
        livelock guard).  Permanently-unreachable picks are skipped on
        every pass — including the second-chance pass — so a fallen
        item can never come back via the livelock-clearing path.
        """
        n = len(self._picking_order_item_names)

        def _candidate(name: str) -> bool:
            return (name not in self._completed_picks
                    and name not in self._permanently_unreachable_picks
                    and self._is_pick_available(name)
                    and self._has_target(name))

        # Forward scan from current position (skipping deferred picks).
        while self._current_pick_index < n:
            name = self._picking_order_item_names[self._current_pick_index]
            if name not in self._deferred_picks and _candidate(name):
                return name
            self._current_pick_index += 1

        # Wrap-around pass.
        for i in range(n):
            name = self._picking_order_item_names[i]
            if name not in self._deferred_picks and _candidate(name):
                self._current_pick_index = i
                return name

        # Second-chance pass: if there are deferred picks that were
        # filtered out above, clear the deferred set and rescan once.
        # Only fires when the non-deferred wrap-around produced None —
        # i.e. the task is otherwise out of candidates — so this is a
        # last-ditch retry before the tree terminates.  Permanent picks
        # are still excluded — _candidate() filters them out.
        if self._deferred_picks:
            logger.info(
                "Second-chance pass: clearing %d deferred pick(s) and rescanning",
                len(self._deferred_picks),
            )
            self._deferred_picks.clear()
            for i in range(n):
                name = self._picking_order_item_names[i]
                if _candidate(name):
                    self._current_pick_index = i
                    return name

        return None

    def _apply_stacking_order(self) -> None:
        """Reorder picking list so top items come first (stable sort by stacking depth).

        Depth 0 = topmost (nothing above), depth increases for lower items.
        Only acts when a stacking_map has been provided.
        """
        if not self._stacking_map:
            return

        depths: Dict[str, int] = {}

        def _depth(name: str) -> int:
            if name in depths:
                return depths[name]
            above = self._stacking_map.get(name, [])
            if not above:
                depths[name] = 0
            else:
                depths[name] = 1 + max(_depth(a) for a in above)
            return depths[name]

        for name in self._picking_order_item_names:
            _depth(name)

        self._picking_order_item_names.sort(key=lambda n: depths.get(n, 0))

    def _reassign_targets_by_picking_order(self) -> None:
        """Redistribute targets so items first in stacking order get them.

        When ``target_count < pick_count`` and stacking reorders picks,
        items that were originally paired with targets may now be late in
        picking order while early items (top of stacks) have no target.
        This method collects all assigned target indices and redistributes
        them in ``_picking_order_item_names`` order.

        Only acts when a stacking_map is present and some picks lack targets.
        """
        if not self._stacking_map:
            return

        # Check if any picks lack targets — if all have targets, nothing to do
        has_none = any(
            self._pairings_by_pick_name.get(name) is None
            for name in self._picking_order_item_names
        )
        if not has_none:
            return

        # Collect all assigned target names (preserving original assignment order).
        available_targets = [
            tgt_name for name in self._picking_order_item_names
            for tgt_name in [self._pairings_by_pick_name.get(name)]
            if tgt_name is not None
        ]

        # Redistribute in picking order.
        target_iter = iter(available_targets)
        new_pairings_by_name: Dict[str, Optional[str]] = {}
        for name in self._picking_order_item_names:
            new_pairings_by_name[name] = next(target_iter, None)

        # Carry over any names not in picking order (None pairing).
        for name in self._pairings_by_pick_name:
            if name not in new_pairings_by_name:
                new_pairings_by_name[name] = None

        self._pairings_by_pick_name = new_pairings_by_name

        logger.debug(
            "Reassigned targets by picking order: %s",
            {n: t for n, t in self._pairings_by_pick_name.items() if t is not None},
        )

    # -------------------------------------------------------------------
    # Target / placing info
    # -------------------------------------------------------------------

    def get_picking_position(self, pick_name: str) -> Optional[np.ndarray]:
        obj = self._pick_objs_by_name.get(pick_name)
        if obj is None:
            return None
        pos, _ = obj.get_local_pose()
        return pos

    def _currently_occupied_target_names(self) -> set:
        """Return names of targets that cannot be re-assigned.

        A target is "occupied" when it carries (or was verified carrying)
        a completed pick.  Sources:

        * Pairings whose pick name is in ``_completed_picks``.
        * Targets claimed by passing frozen snapshots in the task verifier
          (queried through ``_frozen_target_names_fn``) — authoritative even
          after the target has fallen off the conveyor.
        """
        occupied = {
            tgt for name, tgt in self._pairings_by_pick_name.items()
            if tgt is not None and name in self._completed_picks
        }
        if self._frozen_target_names_fn is not None:
            occupied |= set(self._frozen_target_names_fn())
        return occupied

    def _try_retarget(
        self, pick_name: str, old_target_name: Optional[str],
    ) -> Optional[str]:
        """Try to find an alternative reachable target for *pick_name*.

        Excludes targets that already have a completed pick on them and
        targets that are permanently unreachable.  Prefers targets not
        assigned to other uncompleted picks; falls back to stealing from
        the least-progressed pick if needed.

        Returns the new target name, or None if no reachable target exists.
        """
        occupied = self._currently_occupied_target_names()
        # Targets assigned to other uncompleted picks (not yet occupied)
        taken = {
            tgt for name, tgt in self._pairings_by_pick_name.items()
            if tgt is not None and name != pick_name
            and name not in self._completed_picks
        }
        # Prefer unassigned reachable targets
        for tgt_obj in self._target_objs:
            tgt_name = tgt_obj.name
            if (tgt_name not in self._permanently_unreachable_targets
                    and tgt_name not in occupied
                    and tgt_name not in taken
                    and self.is_target_reachable(tgt_name)):
                return tgt_name
        # Fallback: steal from the furthest-from-completion uncompleted pick
        for tgt_obj in reversed(self._target_objs):
            tgt_name = tgt_obj.name
            if (tgt_name not in self._permanently_unreachable_targets
                    and tgt_name not in occupied
                    and tgt_name != old_target_name
                    and self.is_target_reachable(tgt_name)):
                return tgt_name
        return None

    # -------------------------------------------------------------------
    # Target-latch hooks (no-ops in base class; overridden by strategies
    # that do JIT target selection and need mid-place stability).
    # -------------------------------------------------------------------

    def latch_current_target(self, pick_name: str) -> None:
        """Snapshot the currently selected target for *pick_name*.

        Base class: no-op.  Subclasses that re-select targets on every
        ``get_placing_target_name`` call override this to pin the target
        for the duration of the place phase (so RMPFlow isn't chased by
        a moving goal mid-descent).
        """
        pass

    def clear_target_latch(self, pick_name: str) -> None:
        """Clear any latched target for *pick_name*.  Base: no-op."""
        pass

    def clear_all_target_latches(self) -> None:
        """Clear all latched targets.  Base: no-op."""
        pass

    # -------------------------------------------------------------------
    # Pick-latch hooks (no-ops in base class; overridden by strategies
    # that do JIT pick selection and need post-grasp stability).
    # -------------------------------------------------------------------

    def latch_current_pick(self, pick_name: str) -> None:
        """Pin *pick_name* as the current pick through lift/place.

        Base class: no-op.  JIT pick strategies override to prevent a
        newly-arrived higher bottle from redirecting the arm *after*
        the gripper has closed around the committed pick.
        """
        pass

    def clear_pick_latch(self, pick_name: Optional[str] = None) -> None:
        """Clear the current pick latch.  Base: no-op."""
        pass

    def clear_all_pick_latches(self) -> None:
        """Clear all pick latches.  Base: no-op."""
        pass

    def poll_pick_positions(self) -> None:
        """Sample pick positions once per tick.

        Mirrors ``poll_target_reachability``.  Base: no-op.  JIT pick
        strategies override to feed a settled-detection window.
        """
        pass

    # -------------------------------------------------------------------
    # Committed-pick-name stash (race mitigation for pick latching).
    # Set by ``CortexMoveToPick`` on every tick it successfully computes
    # a command; read by ``LatchCurrentPick`` after grasp so the latch
    # pins the name the arm was actually approaching when the gripper
    # closed — not whatever JIT would return fresh at latch time.
    # -------------------------------------------------------------------

    @property
    def committed_pick_name(self) -> Optional[str]:
        return getattr(self, "_committed_pick_name", None)

    @committed_pick_name.setter
    def committed_pick_name(self, value: Optional[str]) -> None:
        self._committed_pick_name = value

    def _is_target_occupied(
        self, target_name: str, exclude_pick: Optional[str] = None,
    ) -> bool:
        """Return True if *target_name* already has a different completed pick on it."""
        for name, tgt in self._pairings_by_pick_name.items():
            if tgt == target_name and name != exclude_pick and name in self._completed_picks:
                return True
        # Also check frozen passing snapshots in the task verifier.
        if self._frozen_target_names_fn is not None:
            if target_name in self._frozen_target_names_fn():
                return True
        return False

    def get_placing_target_name(self, pick_name: str) -> Optional[str]:
        """Return the target name for the given pick, or None if unavailable.

        Returns None when the paired target is unreachable, occupied by
        another completed pick, or no re-pairing is possible.
        """
        tgt_name = self._pairings_by_pick_name.get(pick_name)
        if tgt_name is None:
            return None
        unreachable = not self.is_target_reachable(tgt_name)
        if unreachable or self._is_target_occupied(tgt_name, exclude_pick=pick_name):
            new_tgt_name = self._try_retarget(pick_name, tgt_name)
            if new_tgt_name is None:
                return None
            logger.info(
                "Re-paired '%s': target '%s' -> '%s'",
                pick_name, tgt_name, new_tgt_name,
            )
            self._pairings_by_pick_name[pick_name] = new_tgt_name
            return new_tgt_name
        return tgt_name

    def get_pick_name_for_target(self, target_name: str) -> Optional[str]:
        """Return the name of the pick paired to ``target_name``, or None.

        Prefers a *completed* pick when multiple are paired to the same
        target — JIT pick strategies can leave stale entries pointing at
        the same target, and the completed one is the pick physically
        riding it.  Returns the first encountered match otherwise.
        """
        fallback: Optional[str] = None
        for pick_name, paired_name in self._pairings_by_pick_name.items():
            if paired_name != target_name:
                continue
            if pick_name in self._completed_picks:
                return pick_name
            if fallback is None:
                fallback = pick_name
        return fallback

    def get_end_effector_orientation(self, pick_name: str) -> np.ndarray:
        """Return the EE orientation quaternion for picking the given item.

        Default: gripper pointing down (pi/2 rotation around Y).
        Subclasses can override for per-item orientation.
        """
        return euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))

    def get_end_effector_orientation_for_drop(
        self, pick_name: str, target_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Return the EE orientation for dropping the given pick item.

        Default returns None (fall back to pick orientation).
        Subclasses can override for custom drop orientations.
        """
        return None

    # -------------------------------------------------------------------
    # Completion tracking
    # -------------------------------------------------------------------

    def mark_pick_complete(self, pick_name: str) -> None:
        """Mark the given pick as completed and call on_pair_completed hook.

        Also clears the deferred-pick set — any pick that was skipped
        earlier in this pass gets another chance on the next cycle now
        that the scene has changed (one fewer obstacle, possibly freeing
        an earlier-ungraspable item).  The permanent-unreachable set is
        NOT cleared (those items are dead for the run).  Resets the
        cycles-without-progress safety-net counter.
        """
        self._completed_picks.add(pick_name)
        tgt_name = self._pairings_by_pick_name.get(pick_name)
        self.clear_target_latch(pick_name)
        self.clear_pick_latch(pick_name)
        if self._committed_pick_name == pick_name:
            self._committed_pick_name = None
        if self._deferred_picks:
            logger.debug(
                "mark_pick_complete: clearing %d deferred pick(s) after '%s'",
                len(self._deferred_picks), pick_name,
            )
            self._deferred_picks.clear()
        # Reset per-pick defer counts: a successful completion changes the
        # scene, so prior defers no longer count toward the livelock guard.
        self._defer_counts.clear()
        self._cycles_since_last_completion = 0
        self.on_pair_completed(pick_name, tgt_name)

    # -------------------------------------------------------------------
    # Pick deferral (temporary exclusion — the retry-exhausted handler)
    # -------------------------------------------------------------------

    def defer_pick(self, pick_name: str) -> None:
        """Mark *pick_name* as temporarily excluded from selection.

        ``SelectNextPick`` / ``advance_pick_index`` /
        ``_scan_for_available_pick`` skip deferred picks for the current
        pass.  Cleared on ``mark_pick_complete`` (successful completion
        of any other pick changes the scene) or by the second-chance
        pass in ``_scan_for_available_pick`` (livelock guard).  For the
        permanent flavour (item dropped below the z-floor and is not
        coming back), use ``mark_pick_permanently_unreachable`` instead —
        that flag survives ``mark_pick_complete`` and the second-chance
        pass.

        Also bumps a per-pick consecutive-defer counter.  When the count
        reaches ``MAX_DEFERS_BEFORE_PERMANENT`` the pick is promoted to
        permanently-unreachable: a livelock guard for items that have
        drifted past the kinematic envelope (e.g. tipped boxes near the
        workspace edge) and won't be unblocked by a sibling completion.
        """
        if not pick_name:
            return
        if pick_name in self._permanently_unreachable_picks:
            # No-op: pick is already terminally excluded.  This branch
            # fires when ``CheckGraspPoseReachable`` (or any other gate)
            # has just promoted the pick to permanent and the wrapping
            # ``DeferPickAndRelease`` then runs as a side effect of the
            # pick_attempt FAILURE before ``IsPickReachableGuard`` gets
            # a chance to short-circuit at the next tick.  Counting it
            # would be double-bookkeeping and produce misleading
            # "deferred 1/3" log lines.
            return
        new_count = self._defer_counts.get(pick_name, 0) + 1
        self._defer_counts[pick_name] = new_count
        if new_count >= self.MAX_DEFERS_BEFORE_PERMANENT:
            logger.info(
                "defer_pick: '%s' deferred %d times without progress "
                "(threshold=%d); promoting to permanently unreachable",
                pick_name, new_count, self.MAX_DEFERS_BEFORE_PERMANENT,
            )
            self.mark_pick_permanently_unreachable(pick_name)
            return
        self._deferred_picks.add(pick_name)
        logger.info("defer_pick: '%s' deferred (pass total: %d, attempt %d/%d)",
                    pick_name, len(self._deferred_picks),
                    new_count, self.MAX_DEFERS_BEFORE_PERMANENT)

    def is_pick_deferred(self, pick_name: str) -> bool:
        """Return True if *pick_name* is currently deferred."""
        return pick_name in self._deferred_picks

    def clear_deferred_picks(self) -> None:
        """Un-defer all picks (called on completion or livelock fallback)."""
        self._deferred_picks.clear()

    @property
    def deferred_picks(self) -> set:
        """Return a copy of the currently-deferred pick-name set."""
        return set(self._deferred_picks)

    # -------------------------------------------------------------------
    # Pick permanent-unreachable (run-lifetime exclusion)
    # -------------------------------------------------------------------

    def mark_pick_permanently_unreachable(self, pick_name: str) -> None:
        """Permanently exclude *pick_name* from selection for this run.

        Mirrors ``_permanently_unreachable_targets`` on the pick side:
        used when an item has dropped below the pick-side z-floor (off
        the conveyor / table) — physics has put it permanently out of
        reach, so deferring temporarily is not enough.  Unlike
        ``defer_pick``, this flag is NOT cleared by ``mark_pick_complete``
        or by the second-chance pass in ``_scan_for_available_pick``;
        the pick stays out for the rest of the run (until ``reset()``).

        Idempotent.  Also adds the name to the temporary deferred set
        for the current pass so the pick cursor advances past it on the
        next ``advance_pick_index`` call.
        """
        if not pick_name:
            return
        if pick_name not in self._permanently_unreachable_picks:
            self._permanently_unreachable_picks.add(pick_name)
            logger.info(
                "mark_pick_permanently_unreachable: '%s' (total: %d)",
                pick_name, len(self._permanently_unreachable_picks),
            )
        # Keep in-pass cursor advancement in lockstep with permanent flag.
        self._deferred_picks.add(pick_name)
        # Drop any defer-count bookkeeping for this pick — it's permanent now.
        self._defer_counts.pop(pick_name, None)

    def is_pick_permanently_unreachable(self, pick_name: str) -> bool:
        """Return True if *pick_name* is permanently excluded for this run."""
        return pick_name in self._permanently_unreachable_picks

    @property
    def permanently_unreachable_picks(self) -> set:
        """Return a copy of the permanently-unreachable pick-name set."""
        return set(self._permanently_unreachable_picks)

    # -------------------------------------------------------------------
    # Cycle-progress safety net
    # -------------------------------------------------------------------

    def increment_cycle_count(self) -> int:
        """Bump the no-progress cycle counter and return the new value.

        Reset to 0 by ``mark_pick_complete``.  Used by
        ``CheckCycleProgress`` at the head of ``do_one_pick_place`` as a
        last-line livelock guard for failure modes the per-pick permanent
        flag does not cover (e.g. an item that keeps slipping but stays
        above the z-floor).
        """
        self._cycles_since_last_completion += 1
        return self._cycles_since_last_completion

    @property
    def cycles_since_last_completion(self) -> int:
        """Cycles elapsed since the last successful ``mark_pick_complete``."""
        return self._cycles_since_last_completion

    def on_pair_completed(
        self, pick_name: str, target_name: Optional[str],
    ) -> None:
        """Hook invoked when a pick->target pairing completes.

        Subclasses can override to record metrics or trigger re-planning.
        """
        pass

    # -------------------------------------------------------------------
    # Frozen-target injection (queried by retargeting logic)
    # -------------------------------------------------------------------

    def set_frozen_target_names_fn(self, fn: Optional[Callable[[], set]]) -> None:
        """Provide a callback returning target names already claimed by frozen
        passing snapshots.  Called by the task verifier at wiring time.

        When unset (the default), the strategy treats no targets as
        frozen-claimed — appropriate for single-cycle tasks without a
        conveyor fall-off monitor.
        """
        self._frozen_target_names_fn = fn

    # -------------------------------------------------------------------
    # Target reachability
    # -------------------------------------------------------------------

    def set_target_reachable_fn(self, fn: Optional[Callable]) -> None:
        """Set a predicate ``fn(target_obj) -> bool`` for target reachability.

        When set, ``get_placing_target_name`` returns None for targets that
        fail the check.  Targets that were once reachable but later fail are
        marked permanently unreachable and skipped by ``advance_pick_index``.
        """
        self._target_reachable_fn = fn

    def is_target_reachable(self, target_name: str) -> bool:
        """Check whether *target_name* is currently reachable.

        Side-effects: updates ``_target_was_reachable`` and, when a
        previously-reachable target fails, adds it to
        ``_permanently_unreachable_targets``.
        """
        if self._target_reachable_fn is None:
            return True
        if target_name in self._permanently_unreachable_targets:
            return False
        target_obj = self._target_objs_by_name.get(target_name)
        if target_obj is None:
            return False
        reachable = bool(self._target_reachable_fn(target_obj))
        if reachable:
            self._target_was_reachable[target_name] = True
        elif self._target_was_reachable.get(target_name, False):
            # Was reachable once, now unreachable → permanent
            self._permanently_unreachable_targets.add(target_name)
            logger.info(
                "Target '%s' permanently unreachable", target_name,
            )
        return reachable

    def is_target_permanently_unreachable(self, target_name: str) -> bool:
        """Return True if *target_name* was reachable once but is no longer."""
        return target_name in self._permanently_unreachable_targets

    def poll_target_reachability(self) -> None:
        """Poll all targets to detect reachability transitions.

        Call once per tick (e.g. from ContextMonitor) so that permanently-
        unreachable targets are detected even when they are not currently
        being queried by ``get_placing_target_name``.
        """
        if self._target_reachable_fn is None:
            return
        for t in self._target_objs:
            if t.name not in self._permanently_unreachable_targets:
                self.is_target_reachable(t.name)

    # -------------------------------------------------------------------
    # Reordering / update
    # -------------------------------------------------------------------

    def reorder_picks(self, new_order_names: List[str], current_pick_name: Optional[str] = None) -> None:
        self._picking_order_item_names = list(new_order_names)
        if current_pick_name is not None and current_pick_name in self._picking_order_item_names:
            self._current_pick_index = self._picking_order_item_names.index(current_pick_name)
        else:
            self._current_pick_index = 0

    def update_pairings(self, pairings_by_pick_name: dict) -> None:
        self._pairings_by_pick_name = dict(pairings_by_pick_name)

    def recompute_pairings(self, preserve_current: bool = True, respect_completed: bool = True) -> None:
        """Recompute pairings mid-run and rebuild picking order.

        Args:
            preserve_current: Keep the current active pick name as the active one.
            respect_completed: Keep already completed picks at the front.
        """
        current_name = self.get_current_pick_name() if preserve_current else None

        try:
            pairings = list(self.pair_picks_with_targets())
        except Exception:
            pairings = list(self._default_sequential_pairings())

        self._pairings_by_pick_name = {}
        for pick_idx, tgt_idx in pairings:
            pick_name = self._pick_objs[pick_idx].name
            tgt_name = (
                self._target_objs[tgt_idx].name if tgt_idx is not None else None
            )
            self._pairings_by_pick_name[pick_name] = tgt_name

        # Build new picking order
        completed_names = set()
        if respect_completed:
            completed_names.update(self._completed_picks)
            completed_names.update(
                self._picking_order_item_names[:self._current_pick_index]
            )

        current_sequence = list(self._picking_order_item_names)
        front_completed = [n for n in current_sequence if n in completed_names]

        pairing_picks_in_order = [self._pick_objs[p].name for p, _ in pairings]
        remaining = [
            n for n in pairing_picks_in_order
            if n not in completed_names and (not preserve_current or n != current_name)
        ]

        if preserve_current and current_name and current_name not in completed_names:
            new_order = front_completed + [current_name] + remaining
        else:
            new_order = front_completed + remaining

        self.reorder_picks(new_order, current_pick_name=current_name)

    def reset(self, picking_order_item_names: Optional[List[str]] = None) -> None:
        self._current_pick_index = 0
        self._targets_exhausted = False
        self._task_finished = False
        self._completed_picks.clear()
        self._target_was_reachable.clear()
        self._permanently_unreachable_targets.clear()
        self._deferred_picks.clear()
        self._defer_counts.clear()
        self._permanently_unreachable_picks.clear()
        self._cycles_since_last_completion = 0
        if picking_order_item_names is not None:
            self._picking_order_item_names = list(picking_order_item_names)

    # -------------------------------------------------------------------
    # Verification hooks
    # -------------------------------------------------------------------

    def valid_targets_for_pick(self, pick_name: str) -> List[str]:
        """Return target names considered valid for the given pick.

        Default: all targets. Subclasses may override (e.g., color matching).
        """
        return [tgt.name for tgt in self._target_objs]

    def is_pick_expected(self, pick_name: str) -> bool:
        """Return True if *pick_name* is part of the task's intended placement set.

        Overflow picks (more picks than targets) have no assigned target and
        return False so verification short-circuits them as "not expected to
        be placed".
        """
        return self._pairings_by_pick_name.get(pick_name) is not None

    def placement_constraints_satisfied(self, pick_name: str, target_name: str) -> tuple:
        """Check additional constraints for a pick placed on a target.

        Returns:
            (bool, str): (passed, reason). reason is empty on success,
            contains a failure description on failure.
        """
        return (True, "")

    def get_spatial_check_fn(self):
        """Return a custom spatial check function, or None to use the task default."""
        return None

    def get_recommended_ee_height(self, prim_geometry=None) -> Optional[float]:
        """Return a recommended ee_height_for_move based on current state, or None.

        Strategies that build growing stacks can override this to dynamically
        raise the transport height as the destination stack grows.
        """
        return None

    # -------------------------------------------------------------------
    # Stacking helpers (shared by stacking strategies)
    # -------------------------------------------------------------------

    def _stack_clearance_height(
        self,
        stacks,
        prim_geometry: Optional[dict] = None,
        margin: float = 0.03,
    ) -> Optional[float]:
        """Transport height that clears the tallest completed destination stack.

        Each *stacks* entry is an iterable of pick names in bottom→top order.
        For each stack, finds the highest completed pick and reads its top_z
        from world-pose plus geometry. Returns ``max_top_z + carried_item_rest_height
        + margin``, or ``None`` if no stack has any completed picks.
        """
        if not self._completed_picks:
            return None

        geom_cache = prim_geometry or {}
        max_top_z = 0.0
        for stack in stacks:
            stack_list = list(stack)
            for name in reversed(stack_list):
                if name in self._completed_picks:
                    obj = self._pick_objs_by_name.get(name)
                    geom = geom_cache.get(name)
                    if obj is not None and geom is not None:
                        pos, _ = obj.get_world_pose()
                        max_top_z = max(max_top_z, pos[2] + geom.top_surface_height)
                    break

        if max_top_z <= 0.0:
            return None

        current_pick_name = self.get_current_pick_name()
        current_rest_height = 0.0
        if current_pick_name:
            geom = geom_cache.get(current_pick_name)
            if geom is not None:
                current_rest_height = geom.rest_height

        return max_top_z + current_rest_height + margin

    def _make_marker_or_position_check(self):
        """Return a check fn that uses ``_base_check_fn`` for base-marker targets
        and an XY-proximity + Z-ordering check for dynamic stacking targets.

        Subclasses must set ``_base_check_fn`` and ``_base_target_count``.
        Returns ``None`` if ``_base_check_fn`` is not configured.

        Position-based check avoids false negatives from AABB expansion when
        boxes tilt in a settled stack.
        """
        base_fn = getattr(self, "_base_check_fn", None)
        if base_fn is None:
            return None

        base_target_count = getattr(self, "_base_target_count", 0)
        base_target_names = set(
            obj.name for obj in self._target_objs[:base_target_count]
        )

        def _check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
            if target_obj.name in base_target_names:
                return base_fn(
                    pick_obj, target_obj,
                    bb_cache=bb_cache, obj_scale=obj_scale,
                )
            pick_pos, _ = pick_obj.get_world_pose()
            tgt_pos, _ = target_obj.get_world_pose()
            xy_dist = float(np.linalg.norm(pick_pos[:2] - tgt_pos[:2]))
            z_above = float(pick_pos[2]) > float(tgt_pos[2])
            return xy_dist < 0.05 and z_above

        return _check

    # -------------------------------------------------------------------
    # Read-only accessors
    # -------------------------------------------------------------------

    @property
    def pairings_by_pick_name(self) -> Dict[str, Optional[str]]:
        return self._pairings_by_pick_name

    @property
    def pick_objs(self) -> list:
        return self._pick_objs

    @property
    def pick_objs_by_name(self) -> dict:
        return self._pick_objs_by_name

    @property
    def target_objs(self) -> list:
        return self._target_objs

    @property
    def target_objs_by_name(self) -> dict:
        return self._target_objs_by_name

    @property
    def completed_picks(self) -> set:
        return self._completed_picks


# ---------------------------------------------------------------------------
# ColorMatchStrategy
# ---------------------------------------------------------------------------


class ColorMatchStrategy(MultiPickStrategy):
    """Pair picks to targets by matching semantic color labels.

    Used by TableTaskColors1, TableTaskColorBinSort, TableTaskColorShapes.

    Args:
        pick_objs: List of pick objects.
        target_objs: List of target objects.
        color_palette: List of color names to match against.
        has_color_fn: Callable(obj, color_name) -> bool.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        color_palette: List[str],
        has_color_fn=None,
    ) -> None:
        super().__init__(pick_objs, target_objs)
        self._color_palette = color_palette
        if has_color_fn is None:
            from asset_utils import has_color
            has_color_fn = has_color
        self._has_color = has_color_fn

    @override
    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Pair by semantic color label. Each target used at most once."""
        color_to_targets: dict = {}
        for j, tgt in enumerate(self._target_objs):
            for color_name in self._color_palette:
                if self._has_color(tgt, color_name):
                    color_to_targets.setdefault(color_name, []).append(j)
                    break

        used_targets: set = set()
        for i, pick in enumerate(self._pick_objs):
            pick_color = None
            for color_name in self._color_palette:
                if self._has_color(pick, color_name):
                    pick_color = color_name
                    break
            if pick_color is None:
                yield (i, None)
                continue

            candidates = color_to_targets.get(pick_color, [])
            match_idx = None
            for j in candidates:
                if j not in used_targets:
                    match_idx = j
                    break
            if match_idx is not None:
                used_targets.add(match_idx)
                yield (i, match_idx)
            else:
                yield (i, None)

    def initialize_pairings(self) -> None:
        """After computing pairings, filter picking order to matched picks only."""
        super().initialize_pairings()
        matched = [
            name for name in self._picking_order_item_names
            if self._pairings_by_pick_name.get(name) is not None
        ]
        if matched:
            self._picking_order_item_names = matched
            self._current_pick_index = 0

    def valid_targets_for_pick(self, pick_name: str) -> List[str]:
        """Restrict valid targets to those matching the pick's color."""
        pick = self._pick_objs_by_name.get(pick_name)
        if pick is None:
            return []
        pick_color = None
        for cname in self._color_palette:
            if self._has_color(pick, cname):
                pick_color = cname
                break
        if pick_color is None:
            return []
        return [tgt.name for tgt in self._target_objs if self._has_color(tgt, pick_color)]


# ---------------------------------------------------------------------------
# TypeBasedStrategy
# ---------------------------------------------------------------------------


class TypeBasedStrategy(MultiPickStrategy):
    """Route picks to targets based on item type.

    Targets are pre-grouped by type via ``target_indices_by_type``
    (``{"cube": [0,1,2,3], "ball": [4,5,6,7]}`` or arbitrary type keys like
    ``"cracker_box"`` / ``"sugar_box"``).  Each pick consumes the next unused
    target index from its type's list; picks whose type has no (or exhausted)
    target bucket receive ``None``.

    Per-pick type resolution order (first non-None wins):

    1. ``source_types[pick_index]`` if ``source_types`` was provided.
    2. ``type_detect_fn(pick_obj)`` if ``type_detect_fn`` was provided.
    3. Default: name-prefix match against keys of ``target_indices_by_type``
       (keys tested longest-first so ``"cracker_box"`` wins over ``"box"``).

    Args:
        pick_objs: List of pick objects.
        target_objs: List of target objects.
        target_indices_by_type: Mapping of type string -> list of target indices
            allocated to that type.  Indices are consumed in-order.
        source_types: Optional explicit per-pick type list (length == ``len(pick_objs)``).
        type_detect_fn: Optional callable ``(pick_obj) -> Optional[str]`` used
            when ``source_types`` is not provided.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        target_indices_by_type: Dict[str, List[int]],
        source_types: Optional[List[str]] = None,
        type_detect_fn: Optional[Callable[[object], Optional[str]]] = None,
    ) -> None:
        super().__init__(pick_objs, target_objs)
        self._target_indices_by_type: Dict[str, List[int]] = {
            k: list(v) for k, v in target_indices_by_type.items()
        }
        self._source_types: Optional[List[str]] = (
            list(source_types) if source_types is not None else None
        )
        self._type_detect_fn = type_detect_fn
        # Longest keys first so "cracker_box" wins over "box" in name-prefix match.
        self._known_types: List[str] = sorted(
            self._target_indices_by_type.keys(), key=len, reverse=True
        )

    def _type_for_pick(self, pick_name: str) -> Optional[str]:
        pick_obj = self._pick_objs_by_name.get(pick_name)
        if pick_obj is None:
            return None
        if self._source_types is not None:
            try:
                idx = self._pick_objs.index(pick_obj)
            except ValueError:
                return None
            if 0 <= idx < len(self._source_types):
                return self._source_types[idx]
            return None
        if self._type_detect_fn is not None:
            return self._type_detect_fn(pick_obj)
        for t in self._known_types:
            if pick_name.startswith(t):
                return t
        return None

    @override
    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Route each pick to the next unused target indexed under its type."""
        if not self._target_indices_by_type:
            yield from self._default_sequential_pairings()
            return
        iters = {t: iter(indices) for t, indices in self._target_indices_by_type.items()}
        for idx, pick_obj in enumerate(self._pick_objs):
            t = self._type_for_pick(pick_obj.name)
            tgt_idx = next(iters.get(t, iter(())), None) if t else None
            yield (idx, tgt_idx)

    def valid_targets_for_pick(self, pick_name: str) -> List[str]:
        """Restrict valid targets to the pick's type bucket."""
        t = self._type_for_pick(pick_name)
        if t is None:
            return super().valid_targets_for_pick(pick_name)
        return [self._target_objs[i].name
                for i in self._target_indices_by_type.get(t, [])]


# ---------------------------------------------------------------------------
# BottlePickStrategy
# ---------------------------------------------------------------------------


class BottlePickStrategy(MultiPickStrategy):
    """Strategy for bottle pick-and-place tasks.

    Overrides drop orientation (bottles are placed on their side) and
    computes the drop EE offset from the pick item's top_surface_height.
    """

    _BOTTLE_RADIUS_FALLBACK = 0.03

    def get_end_effector_orientation_for_drop(
        self, pick_name: str, target_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Bottles are placed on their side: pi/2 rotation around X."""
        return HORIZONTAL_DROP_QUAT.copy()


# ---------------------------------------------------------------------------
# BottleGridColumnFillStrategy
# ---------------------------------------------------------------------------


class BottleGridColumnFillStrategy(BottlePickStrategy):
    """BottlePickStrategy that fills a target grid column-by-column.

    Targets are paired in highest-x → lowest-x column order so the arm
    never has to reach over a previously placed bottle to a target at the
    same y but higher x. Within-column ordering is customizable via
    ``secondary_key`` (default: y ascending).

    Args:
        pick_objs: List of pick objects.
        target_objs: List of target objects (grid carrier pads).
        secondary_key: Optional sort key for within-column ordering. ``None``
            uses the helper default (y ascending); pass e.g.
            ``lambda pos: -pos[1]`` for y descending.
        x_tolerance: Maximum x distance to consider two targets in the same column.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        *,
        secondary_key: Optional[Callable[[np.ndarray], float]] = None,
        x_tolerance: float = 1e-3,
    ) -> None:
        super().__init__(pick_objs, target_objs)
        self._secondary_key = secondary_key
        self._x_tolerance = x_tolerance

    @override
    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        kw = {"x_tolerance": self._x_tolerance}
        if self._secondary_key is not None:
            kw["secondary_key"] = self._secondary_key
        yield from pair_by_target_columns(self._pick_objs, self._target_objs, **kw)


# ---------------------------------------------------------------------------
# ColorSortStackBase
# ---------------------------------------------------------------------------


class ColorSortStackBase(MultiPickStrategy):
    """Shared logic for color-sorted stacking strategies.

    Provides classification by color, layer-completeness gating, dynamic
    stacking-target readiness checks, and combined spatial verification
    (base markers + position-based stacking). Subclasses define how picks
    map onto stacks (uniform `_stacks_per_box` vs per-color counts) and
    override `pair_picks_with_targets`, `initialize_pairings`, and
    `get_recommended_ee_height`.

    Args:
        pick_objs: List of pick objects.
        target_objs: List of base-layer target objects (ordered by sort_color
            then stack position).
        sort_colors: Colors to sort.
        skip_colors: Colors to ignore (e.g. distractors). Default empty.
        base_check_fn: Spatial check for bottom-layer placement.
        stacking_map: Source-stacking relationships.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        sort_colors: List[str],
        skip_colors: Optional[List[str]] = None,
        base_check_fn: Optional[Callable] = None,
        stacking_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(pick_objs, target_objs, stacking_map=stacking_map)
        self._sort_colors = list(sort_colors)
        self._skip_colors = list(skip_colors or [])
        self._base_target_count = len(target_objs)
        self._base_check_fn = base_check_fn

        from asset_utils import has_color
        self._has_color = has_color

        # Set by subclass pair_picks_with_targets (per-color list of stacks;
        # each stack is a bottom-to-top sequence of pick names).
        self._color_stacks: Dict[str, List[List[str]]] = {}
        self._max_layers = 0

        # Set by _build_layer_info (called from subclass initialize_pairings)
        self._pick_layer_info: Dict[str, Tuple[str, int]] = {}
        self._color_layer_picks: Dict[str, Dict[int, List[str]]] = {}

    def _classify_pick(self, pick) -> Optional[str]:
        """Return the sort color for a pick, or None if skip/unrecognized."""
        all_colors = self._sort_colors + self._skip_colors
        for c in all_colors:
            if self._has_color(pick, c):
                return c if c in self._sort_colors else None
        return None

    def _build_layer_info(self) -> None:
        """Build pick_name → (color, layer) and color → layer → [pick_names] lookups.

        Used by _is_pick_available to enforce layer-completeness: all stacks of
        a color must complete layer N before any stack advances to layer N+1.
        """
        self._pick_layer_info = {}
        self._color_layer_picks = {}
        for c in self._sort_colors:
            self._color_layer_picks[c] = {}
            for stack in self._color_stacks[c]:
                for layer, name in enumerate(stack):
                    self._pick_layer_info[name] = (c, layer)
                    self._color_layer_picks[c].setdefault(layer, []).append(name)

    def _reassign_targets_by_picking_order(self) -> None:
        """No-op: color-based target assignments must not be redistributed."""
        pass

    def _is_pick_available(self, pick_name: str) -> bool:
        """Check source stacking, destination readiness, AND layer completeness.

        A pick is available only if:
        1. All source items above it are completed (base class check).
        2. Its destination target is ready — base markers are always ready;
           dynamic targets (previously placed cubes) must be completed first.
        3. Layer completeness — all stacks of the same color must complete
           layer N-1 before any stack advances to layer N.  This prevents the
           scanner from skipping a source-blocked cube and advancing other
           stacks, which would leave a gap that the robot must later fill by
           reaching down between tall stacks (causing collisions).
        """
        if not super()._is_pick_available(pick_name):
            return False

        # Check destination readiness for dynamic stacking targets.
        # Dynamic targets are pick objects re-registered as targets via
        # _extend_target_objs; their names appear in _pick_objs_by_name.
        tgt_name = self._pairings_by_pick_name.get(pick_name)
        if (tgt_name is not None
                and tgt_name in self._pick_objs_by_name
                and tgt_name not in self._completed_picks):
            return False

        # Layer completeness: don't advance to layer N until ALL stacks
        # of the same color have completed layer N-1
        info = self._pick_layer_info.get(pick_name)
        if info is not None:
            color, layer = info
            if layer > 0:
                prev_picks = self._color_layer_picks.get(color, {}).get(layer - 1, [])
                if not all(p in self._completed_picks for p in prev_picks):
                    return False

        return True

    def valid_targets_for_pick(self, pick_name: str) -> List[str]:
        """Only the assigned target is valid (prevents false occupancy matches)."""
        tgt_name = self._pairings_by_pick_name.get(pick_name)
        if tgt_name is None:
            return []
        return [tgt_name]

    def get_spatial_check_fn(self):
        """Base check for bottom markers, position-based for stacked cubes."""
        return self._make_marker_or_position_check()


# ---------------------------------------------------------------------------
# LayeredStackStrategy
# ---------------------------------------------------------------------------


class LayeredStackStrategy(MultiPickStrategy):
    """Stack objects in layers based on an ordered list of property values.

    Generalizes the color-stacking pattern: bottom-layer objects go onto marker
    targets, each successive layer targets the objects from the layer below.

    Args:
        pick_objs: List of pick objects.
        target_objs: List of target objects (markers for the bottom layer).
        layer_order: Property values from bottom to top, e.g. ``["blue", "green", "red"]``.
        max_stacks: Maximum number of stack positions available.
        classify_fn: ``obj -> Optional[str]`` returning the property value for an object.
            When *None*, a default color-based classifier is built from *all_values*.
        skip_values: Property values whose objects should be ignored (no target assigned).
        all_values: All recognized values (used to build a default color-based *classify_fn*).
        bin_geometry: Optional bin geometry dict for bottom-layer containment checks.
            Convenience shorthand — converted to a check fn via ``build_bin_geometry_check()``.
        base_check_fn: Optional callable for bottom-layer spatial verification.
            Signature: ``(pick_obj, target_obj=None, bb_cache=None, obj_scale=None) -> bool``.
            When provided, takes priority over *bin_geometry*.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        layer_order: List[str],
        max_stacks: int = 3,
        classify_fn: Optional[Callable] = None,
        skip_values: Optional[List[str]] = None,
        all_values: Optional[List[str]] = None,
        bin_geometry: Optional[dict] = None,
        base_check_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(pick_objs, target_objs)
        self._layer_order = list(layer_order)
        self._max_stacks = max_stacks
        self._skip_values = list(skip_values) if skip_values else []
        self._bin_geometry = bin_geometry

        # Resolve base-layer check: explicit fn > bin_geometry > None
        if base_check_fn is not None:
            self._base_check_fn = base_check_fn
        elif bin_geometry is not None:
            self._base_check_fn = build_bin_geometry_check(bin_geometry)
        else:
            self._base_check_fn = None

        # Build classify_fn
        if classify_fn is not None:
            self._classify_fn = classify_fn
        else:
            # Default: color-based classifier using all_values
            recognized = list(all_values) if all_values else (self._layer_order + self._skip_values)
            from asset_utils import has_color
            _has_color = has_color

            def _default_classify(obj, _colors=recognized, _hc=_has_color):
                for cname in _colors:
                    if _hc(obj, cname):
                        return cname
                return None

            self._classify_fn = _default_classify

        self._num_complete_stacks = 0
        self._base_target_count = len(target_objs)
        # Built by pair_picks_with_targets: per-layer list of target names
        # (one entry per stack at that layer, in stack order).
        self._layer_target_names: List[List[str]] = []

    def _classify_picks(self) -> dict:
        """Group pick indices by property value.

        Returns:
            dict mapping value -> list of pick indices.
        """
        value_picks: dict = {}
        for i, pick in enumerate(self._pick_objs):
            value = self._classify_fn(pick)
            if value is not None:
                value_picks.setdefault(value, []).append(i)
        return value_picks

    @override
    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Pair objects to targets for layered stacking.

        Bottom-layer objects are placed onto marker targets. Each upper layer
        targets the objects from the layer below, added dynamically so that
        ``get_placing_info()`` queries live positions.
        """
        # Reset target list to base markers if re-entering
        self._truncate_target_objs(self._base_target_count)

        value_picks = self._classify_picks()

        # How many complete stacks can we form?
        counts = [len(value_picks.get(v, [])) for v in self._layer_order]
        self._num_complete_stacks = min(*counts, self._max_stacks) if counts else 0
        N = self._num_complete_stacks

        logger.info(
            "LayeredStackStrategy: %d complete stacks from layers %s",
            N,
            {v: len(value_picks.get(v, [])) for v in self._layer_order},
        )

        if N == 0:
            for i in range(len(self._pick_objs)):
                yield (i, None)
            return

        # Build layer target names:
        # Layer 0 uses marker targets (the first N base targets)
        # Layer k (k>=1) uses pick objects from layer k-1 as targets
        self._layer_target_names = [
            [self._target_objs[s].name for s in range(N)]
        ]
        for k in range(1, len(self._layer_order)):
            prev_value = self._layer_order[k - 1]
            prev_picks = [self._pick_objs[value_picks[prev_value][s]] for s in range(N)]
            start = self._extend_target_objs(prev_picks)
            self._layer_target_names.append(
                [self._target_objs[start + s].name for s in range(N)]
            )

        # Pair objects for each complete stack.  The base-class
        # pair_picks_with_targets contract still requires index pairs; we
        # walk a local name→idx map once rather than per-yield.
        target_name_to_idx = {t.name: i for i, t in enumerate(self._target_objs)}
        used_picks: set = set()
        for stack_idx in range(N):
            for layer_idx, value in enumerate(self._layer_order):
                pick_idx = value_picks[value][stack_idx]
                used_picks.add(pick_idx)
                target_name = self._layer_target_names[layer_idx][stack_idx]
                yield (pick_idx, target_name_to_idx.get(target_name))

        # Remaining picks (skipped values + excess) get no target
        for i in range(len(self._pick_objs)):
            if i not in used_picks:
                yield (i, None)

    def initialize_pairings(self) -> None:
        """After pairing, set picking order: within each stack, bottom to top."""
        super().initialize_pairings()

        value_picks = self._classify_picks()
        ordered: list = []
        for stack_idx in range(self._num_complete_stacks):
            for value in self._layer_order:
                pick_obj = self._pick_objs[value_picks[value][stack_idx]]
                ordered.append(pick_obj.name)

        self._picking_order_item_names = ordered
        self._current_pick_index = 0

    def valid_targets_for_pick(self, pick_name: str) -> List[str]:
        """Only the targets matching this pick's layer are valid."""
        pick = self._pick_objs_by_name.get(pick_name)
        if pick is None:
            return []
        pick_value = self._classify_fn(pick)

        if pick_value is None or pick_value not in self._layer_order:
            return []

        if not self._layer_target_names:
            return []

        layer_idx = self._layer_order.index(pick_value)
        return list(self._layer_target_names[layer_idx])

    def get_spatial_check_fn(self):
        """Return a dispatch: base_check_fn for bottom layer, is_on_top for stacked layers."""
        if self._base_check_fn is None:
            return None

        from task_verification import is_on_top

        base_target_names = set(
            obj.name for obj in self._target_objs[:self._base_target_count]
        )
        base_fn = self._base_check_fn

        def _check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
            if target_obj.name in base_target_names:
                return base_fn(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale)
            else:
                return is_on_top(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale)

        return _check


# ---------------------------------------------------------------------------
# SingleStackStrategy
# ---------------------------------------------------------------------------


class SingleStackStrategy(MultiPickStrategy):
    """Place all picks into a single growing stack at one target location.

    Respects source-side stacking constraints (from ``stacking_map``) to
    determine the picking order.  Items are placed in picking order: first
    picked = bottom of destination stack.

    Args:
        pick_objs: Pick objects.
        target_objs: Single marker target (base of the stack).
        stacking_map: Source-side stacking relationships.
        bin_geometry: Optional bin geometry for bottom-layer spatial checks.
            Convenience shorthand — converted to a check fn via ``build_bin_geometry_check()``.
        base_check_fn: Optional callable for bottom-layer spatial verification.
            Signature: ``(pick_obj, target_obj=None, bb_cache=None, obj_scale=None) -> bool``.
            When provided, takes priority over *bin_geometry*.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        stacking_map: Optional[Dict[str, List[str]]] = None,
        bin_geometry: Optional[dict] = None,
        base_check_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(pick_objs, target_objs, stacking_map)
        self._bin_geometry = bin_geometry
        self._base_target_count = len(target_objs)
        self._stacking_order: List[str] = []

        # Resolve base-layer check: explicit fn > bin_geometry > None
        if base_check_fn is not None:
            self._base_check_fn = base_check_fn
        elif bin_geometry is not None:
            self._base_check_fn = build_bin_geometry_check(bin_geometry)
        else:
            self._base_check_fn = None

    def _compute_source_constrained_order(self) -> List[str]:
        """Order pick names respecting source stacking: topmost items first."""
        if not self._stacking_map:
            return [obj.name for obj in self._pick_objs]

        depths: Dict[str, int] = {}

        def _depth(name: str) -> int:
            if name in depths:
                return depths[name]
            above = self._stacking_map.get(name, [])
            if not above:
                depths[name] = 0
            else:
                depths[name] = 1 + max(_depth(a) for a in above)
            return depths[name]

        for obj in self._pick_objs:
            _depth(obj.name)

        names = [obj.name for obj in self._pick_objs]
        names.sort(key=lambda n: depths.get(n, 0))
        return names

    @override
    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Chain all picks into a single stack: first -> marker, rest -> previous pick."""
        self._truncate_target_objs(self._base_target_count)
        self._stacking_order = self._compute_source_constrained_order()

        if not self._stacking_order:
            return

        def _idx(name: str) -> int:
            return self._pick_objs.index(self._pick_objs_by_name[name])

        # First item -> base marker (index 0)
        yield (_idx(self._stacking_order[0]), 0)

        # Each subsequent item -> previous item (dynamically added as target)
        for k in range(1, len(self._stacking_order)):
            prev_pick = self._pick_objs_by_name[self._stacking_order[k - 1]]
            tgt_start = self._extend_target_objs([prev_pick])
            yield (_idx(self._stacking_order[k]), tgt_start)

    def initialize_pairings(self) -> None:
        """Set picking order to source-stacking-constrained order."""
        super().initialize_pairings()
        if self._stacking_order:
            self._picking_order_item_names = list(self._stacking_order)
            self._current_pick_index = 0

    def valid_targets_for_pick(self, pick_name: str) -> List[str]:
        """Only the assigned target in the stacking chain is valid for each pick.

        In a single-stack scenario, each pick is paired with exactly one target
        (the item below it).  Returning all targets would let the exclusive
        occupancy check in PlacementChecker match picks to wrong targets when
        physics settling shifts Z positions within tolerance.
        """
        tgt_name = self._pairings_by_pick_name.get(pick_name)
        if tgt_name is None:
            return []
        return [tgt_name]

    def get_recommended_ee_height(self, prim_geometry=None) -> Optional[float]:
        """Transport height to clear the current destination stack top."""
        if not self._stacking_order:
            return None
        return self._stack_clearance_height([self._stacking_order], prim_geometry)

    def get_spatial_check_fn(self):
        """Base check for bottom (marker) target, position-based for stacked items."""
        return self._make_marker_or_position_check()

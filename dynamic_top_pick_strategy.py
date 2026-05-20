"""DynamicTopPickStrategy: JIT pick + target selection for bottle tasks.

Combines two JIT mechanisms:

*Pick side (this class)*: ``get_current_pick_name`` scans all
uncompleted picks and returns the one with the highest world-frame Z
that is *settled* (max XY and Z drift across a sliding window of
samples fall below tolerances).  ``CortexMoveToPick`` re-queries every
tick, so a newly-arrived higher bottle redirects the arm — but only
once the bottle has stopped dropping/rolling.  After the gripper
closes, ``LatchCurrentPick`` pins the pick name so the carried item is
not abandoned mid-cycle.

*Target side (inherited from ``ConveyorProximityStrategy``)*:
``get_placing_target_name`` returns the reachable, unoccupied target
with the smallest distance to the conveyor edge, using ``conveyor_axis``
/ ``conveyor_sign``.  Targets are latched during the place phase via
``LatchPlacementTarget``; if the latched target becomes permanently
unreachable (falls off the belt) the latch is cleared and JIT
re-selects the next most-urgent target so the carried item is
smoothly redirected rather than dropped.

The bottles-specific drop orientation (bottle on its side,
``pi/2`` rotation around X) is applied here so the strategy stays a
single-inheritance chain through ``ConveyorProximityStrategy`` rather
than pulling in ``BottlePickStrategy`` via multiple inheritance.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from conveyor_proximity_strategy import ConveyorProximityStrategy
from multi_pick_strategy import HORIZONTAL_DROP_QUAT

logger = logging.getLogger(__name__)


class DynamicTopPickStrategy(ConveyorProximityStrategy):
    """JIT pick selection by highest settled world-frame Z.

    Args:
        pick_objs: Pick (source) objects.
        target_objs: Target (destination) objects.
        settle_window: Number of position samples over which drift is
            measured.  A pick must have ``settle_window + 1`` samples in
            its history before it can be considered settled.
        settle_xy_tol: Maximum horizontal *net* displacement (metres) —
            first-sample to last-sample — across the window.  Sized for
            the ~3-5 mm net drift observed in Isaac Sim stacks of
            contacting bodies (penalty-force chatter, friction noise).
            A falling bottle covers ≥15 mm in a 5-tick window at 60 Hz
            even just after release, so this still rejects real motion.
        settle_z_tol: Maximum vertical *net* displacement (metres) to
            call a pick settled.  Slightly tighter than
            ``settle_xy_tol`` because a bottle that is genuinely falling
            accumulates Z drift fastest.
        min_pick_z: World-frame Z below which a pick is treated as "not
            yet in the pile" and excluded from selection.  ``None`` to
            disable.
        top_z_margin: Picks within this Z of the current tallest
            settled pick are treated as equivalent.  Tie-break: keep
            the ``committed_pick_name`` (the in-flight pick) when it is
            still in the tied set; otherwise the alphabetically-first
            tied name.  This prevents JIT ping-pong both between two
            near-tied bottles (Z jitter flipping which is "highest")
            and when a neighbour merely settles into the tied band
            mid-approach — both would otherwise yank the arm mid-flight.
            Sized to 1 cm so sub-cm settled-Z noise between two bottles
            on the same layer does not trigger a redirect.  A genuinely
            higher new arrival (>1 cm above the committed pick) falls
            outside the tied set and does redirect, preserving the
            mid-flight-redirect feature for the cases where it matters.
        force_settled_after_ticks: Watchdog that treats an uncompleted
            pick as settled once it has been polled for this many ticks
            even if its net displacement still exceeds the tolerances.
            Prevents deadlock when physics noise or a slow contact
            shuffle leaves a bottle perpetually "unsettled" in the bin.
            Default 60 (~1 s at 60 Hz — long enough for a freshly
            dropped bottle to truly settle, short enough that the task
            never visibly hangs).  Pass ``None`` to disable the watchdog.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        *,
        conveyor_axis: str = "y",
        conveyor_sign: int = -1,
        conveyor_end: Optional[float] = None,
        settle_window: int = 5,
        settle_xy_tol: float = 0.008,
        settle_z_tol: float = 0.006,
        min_pick_z: Optional[float] = None,
        top_z_margin: float = 0.010,
        pick_region: Optional[object] = None,
        force_settled_after_ticks: Optional[int] = 60,
    ) -> None:
        super().__init__(
            pick_objs, target_objs,
            conveyor_axis=conveyor_axis,
            conveyor_sign=conveyor_sign,
            conveyor_end=conveyor_end,
        )
        self._settle_window = int(settle_window)
        self._settle_xy_tol = float(settle_xy_tol)
        self._settle_z_tol = float(settle_z_tol)
        self._min_pick_z = float(min_pick_z) if min_pick_z is not None else None
        self._top_z_margin = float(top_z_margin)
        self._force_settled_after_ticks = (
            int(force_settled_after_ticks)
            if force_settled_after_ticks is not None else None
        )
        # Axis-aligned XY rectangle (e.g. env_config_values.Region2D with
        # attributes min_x/max_x/min_y/max_y).  Picks whose latest sampled
        # position lies outside the rectangle are excluded from JIT
        # selection and do not hold the task open via more_items_expected.
        # None disables the filter (any XY accepted).
        self._pick_region = pick_region

        self._latched_pick_name: Optional[str] = None
        self._position_history: Dict[str, Deque[np.ndarray]] = {}
        self._settled_since_tick: Dict[str, int] = {}
        self._first_seen_tick: Dict[str, int] = {}
        self._tick_counter: int = 0
        self._last_jit_returned: Optional[str] = None
        # Names logged as "out of pick region" so we only log once per transition.
        self._out_of_region_logged: set = set()
        # Names we've already logged the force-settled fallback for.
        self._force_settled_logged: set = set()
        # Cached set of unsettled-blocker names from the last diagnostic —
        # re-logs only when the set changes, to avoid per-tick spam.
        self._last_blocker_set: Optional[frozenset] = None

    # -------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------

    def initialize_pairings(self) -> None:
        """Chain to super() but zero the picking order — JIT doesn't use it.

        Keeps ``_pick_objs_by_name`` (needed for name lookup) but empties
        the list-based iteration state and the pairing map — both get
        populated on demand by JIT.
        """
        super().initialize_pairings()
        self._picking_order_item_names = []
        self._current_pick_index = 0
        self._pairings_by_pick_name = {}

    # -------------------------------------------------------------------
    # Incremental picks — no pairing rebuild
    # -------------------------------------------------------------------

    def add_incremental_picks(self, new_objs: list) -> None:
        """Append new picks without rebuilding pairings, stacking, or order.

        JIT selection automatically picks up new arrivals via
        ``_jit_select_top_settled``.  Mirrors
        ``ConveyorProximityStrategy.add_incremental_targets``.
        """
        self._extend_pick_objs(new_objs)

    # -------------------------------------------------------------------
    # Settled detection
    # -------------------------------------------------------------------

    def poll_pick_positions(self) -> None:
        """Sample world-frame positions for all uncompleted picks.

        Skipped while a pick is latched: during the grasp / lift / place
        / lift phases, JIT pick selection is disabled (see
        ``get_current_pick_name`` — the latched name wins unconditionally),
        so per-tick pose queries and settle-state accumulation for the
        remaining bottles is wasted work.  Polling resumes automatically
        when the latch clears on the next cycle (``advance_pick_index`` /
        ``mark_pick_complete``), and the rolling window re-fills within
        ``settle_window + 1`` ticks before the next JIT decision — well
        under the ``force_settled_after_ticks`` watchdog horizon.

        Skipping the tick counter increment (not just the sampling loop)
        means ``_first_seen_tick`` → watchdog math still measures elapsed
        *polling* opportunities, which is the intended semantic: a pick
        held in the bin during a grasp of some *other* pick didn't get
        60 chances to settle.
        """
        if self._latched_pick_name is not None:
            return
        self._tick_counter += 1
        for obj in self._pick_objs:
            name = obj.name
            if name in self._completed_picks:
                self._position_history.pop(name, None)
                self._settled_since_tick.pop(name, None)
                self._first_seen_tick.pop(name, None)
                self._force_settled_logged.discard(name)
                continue
            try:
                p, _ = obj.get_world_pose()
            except Exception:
                continue
            hist = self._position_history.get(name)
            if hist is None:
                hist = deque(maxlen=self._settle_window + 1)
                self._position_history[name] = hist
                self._first_seen_tick.setdefault(name, self._tick_counter)
            hist.append(np.asarray(p, dtype=float).copy())
            if self._is_settled(name):
                self._settled_since_tick.setdefault(name, self._tick_counter)
            else:
                self._settled_since_tick.pop(name, None)

    def _is_in_pick_region(self, pick_name: str) -> bool:
        """True if *pick_name*'s most recent position is inside ``pick_region``.

        Uses the last sample in ``_position_history`` when available;
        falls back to a live ``get_world_pose()`` query otherwise.  When
        ``pick_region`` is None, always returns True.

        A pick transitioning from inside to outside is logged once (info
        level) so displacement events during run time are visible.
        """
        if self._pick_region is None:
            return True
        hist = self._position_history.get(pick_name)
        if hist is not None and len(hist) > 0:
            pos = hist[-1]
        else:
            obj = self._pick_objs_by_name.get(pick_name)
            if obj is None:
                return False
            try:
                pos, _ = obj.get_world_pose()
            except Exception:
                return True  # fail-open: don't permanently exclude on transient errors
        x = float(pos[0])
        y = float(pos[1])
        r = self._pick_region
        inside = (r.min_x <= x <= r.max_x) and (r.min_y <= y <= r.max_y)
        if not inside and pick_name not in self._out_of_region_logged:
            logger.info(
                "DynamicTopPickStrategy: pick '%s' at (x=%.3f, y=%.3f) is "
                "outside pick_region — excluding from future selection",
                pick_name, x, y,
            )
            self._out_of_region_logged.add(pick_name)
        elif inside and pick_name in self._out_of_region_logged:
            # Rare: pick returned to the bin (e.g. bounced back in).
            self._out_of_region_logged.discard(pick_name)
        return inside

    def _is_settled(self, pick_name: str) -> bool:
        """Return True if *pick_name* has near-zero *net* displacement
        across the most recent ``settle_window + 1`` samples, OR if the
        force-settled watchdog has elapsed.

        Uses net (first-to-last) displacement rather than min-to-max
        range so that a bottle oscillating in place — which is the norm
        in a stack of bodies with contact and numerical noise — is not
        mis-classified as moving.  A bottle that is actually falling or
        sliding accumulates net displacement in one direction and fails
        the check.

        Watchdog fallback: once a pick has been in the bin for more
        than ``force_settled_after_ticks`` ticks, it is treated as
        settled even if the drift check still fails — prevents deadlock
        when a bottle is perpetually nudged by contacts but is actually
        pickable.  ``min_pick_z`` still applies, so a bottle that is
        genuinely still falling below the tabletop is rejected.
        """
        hist = self._position_history.get(pick_name)
        if hist is None or len(hist) <= self._settle_window:
            return False
        # Only the first and last samples drive the net-displacement check;
        # stacking the full deque was pure overhead (called every tick per
        # uncompleted pick, so it dominated the per-tick cost for tasks
        # with many bottles in flight).
        first_sample = hist[0]
        last_sample = hist[-1]
        if self._min_pick_z is not None and float(last_sample[2]) < self._min_pick_z:
            return False
        net = last_sample - first_sample
        xy_net = float(np.hypot(net[0], net[1]))
        z_net = float(abs(net[2]))
        if xy_net <= self._settle_xy_tol and z_net <= self._settle_z_tol:
            return True
        # Watchdog fallback: the pick has been resident long enough;
        # treat as settled to unblock selection.
        if self._force_settled_after_ticks is not None:
            first = self._first_seen_tick.get(pick_name)
            if (first is not None
                    and self._tick_counter - first >= self._force_settled_after_ticks):
                if pick_name not in self._force_settled_logged:
                    logger.info(
                        "DynamicTopPickStrategy: force-settling '%s' after "
                        "%d ticks (xy_net=%.4f, z_net=%.4f exceed tolerances "
                        "%.4f/%.4f)",
                        pick_name, self._tick_counter - first,
                        xy_net, z_net,
                        self._settle_xy_tol, self._settle_z_tol,
                    )
                    self._force_settled_logged.add(pick_name)
                return True
        return False

    # -------------------------------------------------------------------
    # JIT pick selection
    # -------------------------------------------------------------------

    def _jit_select_top_settled(self) -> Optional[str]:
        """Return the highest-Z settled uncompleted pick name, or None."""
        candidates: List[Tuple[str, float]] = []
        for obj in self._pick_objs:
            name = obj.name
            if name in self._completed_picks:
                continue
            if not self._is_settled(name):
                continue
            if not self._is_in_pick_region(name):
                continue
            try:
                p, _ = obj.get_world_pose()
            except Exception:
                continue
            candidates.append((name, float(p[2])))
        if not candidates:
            self._log_blocker_diagnostic()
            return None
        self._last_blocker_set = None  # reset so next stall re-logs
        # Fix A: deterministic tie-break among candidates within
        # ``top_z_margin`` of the max.  The previous implementation sorted
        # by (-z, name) and returned the first within margin, which always
        # resolved to the instantaneously-highest candidate — physics
        # jitter flipping which of two near-tied bottles has the larger
        # Z (by ~1 mm between ticks) then caused catastrophic pick
        # ping-pong mid-approach.  Alphabetical tie-break among "tied"
        # picks produces a stable choice independent of per-tick Z noise.
        candidates.sort(key=lambda nz: -nz[1])
        top_z = candidates[0][1]
        tied = sorted(
            name for name, z in candidates if top_z - z <= self._top_z_margin
        )
        # Fix B: sticky committed pick.  If the in-flight pick
        # (``committed_pick_name``, stashed every tick by
        # ``CortexMoveToPreGrasp`` / ``CortexExecuteApproach``) is still
        # a tied candidate, keep it — a freshly-settled neighbour that
        # *ties* the current choice must not redirect the arm mid-flight.
        # A genuinely higher arrival (>``top_z_margin`` above the
        # committed pick) falls outside ``tied`` and does redirect,
        # preserving the original mid-flight-redirect feature for the
        # cases where it matters.
        committed = self.committed_pick_name
        if committed is not None and committed in tied:
            return committed
        return tied[0]

    def _log_blocker_diagnostic(self) -> None:
        """Log (once per transition) which uncompleted in-region picks are
        preventing selection — their drift and how long they have been
        resident.  Called when ``_jit_select_top_settled`` returns None.
        """
        blockers = []  # (name, xy_net, z_net, age_ticks, reason)
        for obj in self._pick_objs:
            name = obj.name
            if name in self._completed_picks:
                continue
            if not self._is_in_pick_region(name):
                continue
            hist = self._position_history.get(name)
            if hist is None or len(hist) <= self._settle_window:
                blockers.append((name, None, None,
                                 len(hist) if hist else 0, "warming-up"))
                continue
            first_sample = hist[0]
            last_sample = hist[-1]
            net = last_sample - first_sample
            xy_net = float(np.hypot(net[0], net[1]))
            z_net = float(abs(net[2]))
            first = self._first_seen_tick.get(name, self._tick_counter)
            age = self._tick_counter - first
            if self._min_pick_z is not None and float(last_sample[2]) < self._min_pick_z:
                reason = "below min_pick_z"
            else:
                reason = "drift"
            blockers.append((name, xy_net, z_net, age, reason))
        if not blockers:
            self._last_blocker_set = None
            return
        current_set = frozenset(b[0] for b in blockers)
        if current_set == self._last_blocker_set:
            return
        self._last_blocker_set = current_set
        details = ", ".join(
            f"{n}(xy={xy:.4f},z={z:.4f},age={age}t,{r})"
            if xy is not None else f"{n}(age={age}t,{r})"
            for n, xy, z, age, r in blockers
        )
        logger.info(
            "DynamicTopPickStrategy: no settled pick; blockers: %s "
            "(settle_tol xy=%.4f z=%.4f, force_after=%s)",
            details, self._settle_xy_tol, self._settle_z_tol,
            self._force_settled_after_ticks,
        )

    def get_current_pick_name(self) -> Optional[str]:
        """Return the latched pick if set, otherwise the JIT top-settled pick."""
        if self._latched_pick_name is not None:
            if (self._latched_pick_name in self._pick_objs_by_name
                    and self._latched_pick_name not in self._completed_picks):
                return self._latched_pick_name
            # Latched name is stale (removed or already completed); clear.
            self._latched_pick_name = None
        name = self._jit_select_top_settled()
        if name != self._last_jit_returned:
            if self._last_jit_returned is not None and name is not None:
                logger.info(
                    "DynamicTopPickStrategy: JIT pick '%s' -> '%s'",
                    self._last_jit_returned, name,
                )
            self._last_jit_returned = name
        return name

    def advance_pick_index(self) -> Optional[str]:
        """Clear latch and return the next JIT selection."""
        self._latched_pick_name = None
        self._last_jit_returned = None
        return self._jit_select_top_settled()

    @property
    def all_picks_done(self) -> bool:
        """All picks completed AND no incremental spawn pending."""
        if self._more_items_expected:
            return False
        for obj in self._pick_objs:
            if obj.name not in self._completed_picks:
                return False
        return True

    # -------------------------------------------------------------------
    # Pick-latch API
    # -------------------------------------------------------------------

    def latch_current_pick(self, pick_name: str) -> None:
        if pick_name is None:
            return
        self._latched_pick_name = pick_name
        logger.debug("DynamicTopPickStrategy: latched pick '%s'", pick_name)

    def clear_pick_latch(self, pick_name: Optional[str] = None) -> None:
        if pick_name is None or self._latched_pick_name == pick_name:
            self._latched_pick_name = None

    def clear_all_pick_latches(self) -> None:
        self._latched_pick_name = None

    # -------------------------------------------------------------------
    # Target pairing
    #
    # ``get_placing_target_name``, ``_jit_select``, ``latch_current_target``,
    # ``clear_target_latch``, ``_try_retarget``, and ``add_incremental_targets``
    # are inherited from ``ConveyorProximityStrategy``.  A target is selected
    # by smallest distance to the conveyor edge (``conveyor_axis`` /
    # ``conveyor_sign``) and latched at start of the place phase by
    # ``LatchPlacementTarget`` in the cortex tree.
    # -------------------------------------------------------------------

    def _any_reachable_target_available(self) -> bool:
        """True if at least one target is neither permanently unreachable,
        occupied by a completed pick, nor failing the reachability check.

        Used by ``more_items_expected`` to short-circuit the
        "wait-for-settle" state when no placement slot remains.  Does NOT
        consider targets pre-assigned to other uncompleted picks — the
        looser check reflects "is the task still viable at all".
        """
        occupied = self._currently_occupied_target_names()
        for t in self._target_objs:
            if t.name in self._permanently_unreachable_targets:
                continue
            if t.name in occupied:
                continue
            if not self.is_target_reachable(t.name):
                continue
            return True
        return False

    # -------------------------------------------------------------------
    # Drop orientation (bottle on side)
    # -------------------------------------------------------------------

    def get_end_effector_orientation_for_drop(
        self, pick_name: str, target_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Bottles are placed on their side — shared ``HORIZONTAL_DROP_QUAT``.

        Matches ``BottlePickStrategy`` without pulling it in via multiple
        inheritance; this class extends ``ConveyorProximityStrategy``
        directly.
        """
        return HORIZONTAL_DROP_QUAT.copy()

    # -------------------------------------------------------------------
    # more_items_expected override — stay RUNNING while unsettled.
    # -------------------------------------------------------------------

    @property
    def more_items_expected(self) -> bool:
        """True while the task has any realistic chance of placing another pick.

        - Incremental spawn still pending: always True.
        - No reachable, unoccupied target remains AND no more targets
          are expected: False.  Waiting for an unsettled bottle to
          stabilise is pointless when there is nothing to place it on;
          without this short-circuit, a leftover jostling bottle in the
          bin combined with a permanently-unreachable last target would
          trap ``SelectNextPick`` in an infinite RUNNING loop.
        - Otherwise: True iff any uncompleted pick lacks a full settle
          window of samples or is currently unsettled.
        """
        if self._more_items_expected:
            return True
        if (not self._more_targets_expected
                and not self._any_reachable_target_available()):
            return False
        required = self._settle_window + 1
        for obj in self._pick_objs:
            if obj.name in self._completed_picks:
                continue
            # Displaced bottles (outside pick_region) are unreachable and
            # don't block task termination, even if their positions still
            # wobble from physics.
            if not self._is_in_pick_region(obj.name):
                continue
            hist = self._position_history.get(obj.name)
            if hist is None or len(hist) < required:
                return True
            if not self._is_settled(obj.name):
                return True
        return False

    @more_items_expected.setter
    def more_items_expected(self, value: bool) -> None:
        self._more_items_expected = bool(value)

    # -------------------------------------------------------------------
    # Introspection (tests / diagnostics)
    # -------------------------------------------------------------------

    @property
    def latched_pick_name(self) -> Optional[str]:
        return self._latched_pick_name

"""ConveyorFalloffMonitor: snapshot verification for targets leaving a moving conveyor.

Triggering is based on the **leading edge** (smallest Y of the AABB) of either
the target or the pick already placed on it. This is what matters physically —
once the leading edge of the pair crosses the conveyor's -Y boundary the
combined object starts to tilt/fall and the spatial relationship is lost. The
snapshot runs immediately before that happens so any pick already placed is
verified against a still-valid belt-top geometry.

If no pick has been placed on a target that crosses the threshold, the monitor
calls ``on_available_lost`` instead — informational only, surfaced in the
final task report as "target was available but not filled in time".

Once a target passes beyond the edge entirely, the monitor hides the prim via
``set_visibility(False)``. The paired pick (if any) is hidden alongside the
target so cans don't remain visible floating past the belt.

Pure Python — no Isaac Sim or USD imports — so the monitor can be exercised
from unit tests with ``LightweightObj``-style stand-ins.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Set

logger = logging.getLogger(__name__)


def _default_hide(prim: Any) -> None:
    """Default hide callback: ``prim.set_visibility(False)`` if available."""
    try:
        prim.set_visibility(False)
    except Exception:
        logger.debug(
            "ConveyorFalloffMonitor: set_visibility(False) failed on '%s'",
            getattr(prim, "name", "?"),
            exc_info=True,
        )


def _default_half_extent_y(prim: Any) -> float:
    """Default half-extent-Y lookup.

    Tries ``prim._local_half_extents[1]`` (works for LightweightObj mocks and
    any wrapper that exposes axis-aligned half-extents). Falls back to 0.0,
    which degrades the leading-edge check to a center-only check — safe but
    not preferred. Callers with access to a geometry cache should pass their
    own ``get_half_extent_y`` for accurate triggering.
    """
    local_he = getattr(prim, "_local_half_extents", None)
    if local_he is not None:
        try:
            return float(local_he[1])
        except Exception:
            pass
    return 0.0


class ConveyorFalloffMonitor:
    """Polls target positions and fires snapshot/hide callbacks near the conveyor edge.

    Args:
        strategy: The task's MultiPickStrategy. Used to resolve pick pairings
            and to check which picks have completed.
        target_objs_ref: Live reference to the strategy's ordered target list.
            Re-read every ``poll()`` so newly spawned targets are picked up
            automatically.
        conveyor_end_y: Y coordinate of the conveyor's -Y edge. A target or
            its paired pick triggers the snapshot when its leading edge
            (position.y minus half-extent-y) drops below
            ``conveyor_end_y + snapshot_margin``. Hide fires when the target's
            leading edge drops below ``conveyor_end_y``.
        snapshot_margin: Distance (m) before the edge at which the snapshot
            should fire. Larger margins snapshot earlier (more time for the
            pick to still be correctly on its target); smaller margins let
            the snapshot capture state closer to fall-off. The leading-edge
            trigger already accounts for the object extent, so this margin
            can be modest.
        hide_after: If True, hide the target (and its paired pick, if any)
            once the target's leading edge drops below ``conveyor_end_y``.
        on_snapshot: ``Callable[[pick_name, target_name, sim_time], None]``.
            Invoked once per target whose paired pick has completed and
            whose leading edge has crossed the snapshot threshold.
        on_available_lost: ``Callable[[target_name, sim_time], None]``.
            Invoked once per target that crosses the snapshot threshold
            without a completed pick on it.
        on_hide: Optional ``Callable[[prim], None]``. Defaults to
            ``prim.set_visibility(False)``. Applied to both target and
            paired pick.
        get_half_extent_y: Optional ``Callable[[prim], float]`` returning the
            Y half-extent of a prim's AABB. Defaults to ``_default_half_extent_y``
            which reads ``_local_half_extents[1]`` when available.
    """

    def __init__(
        self,
        strategy,
        target_objs_ref: List,
        conveyor_end_y: float,
        snapshot_margin: float,
        hide_after: bool,
        on_snapshot: Callable[[str, str, float], None],
        on_available_lost: Callable[[str, float], None],
        on_hide: Optional[Callable[[Any], None]] = None,
        get_half_extent_y: Optional[Callable[[Any], float]] = None,
    ) -> None:
        self._strategy = strategy
        self._target_objs_ref = target_objs_ref
        self._conveyor_end_y = float(conveyor_end_y)
        self._snapshot_margin = float(snapshot_margin)
        self._hide_after = bool(hide_after)
        self._on_snapshot = on_snapshot
        self._on_available_lost = on_available_lost
        self._on_hide = on_hide or _default_hide
        self._get_half_extent_y = get_half_extent_y or _default_half_extent_y

        self._snapshotted_target_names: Set[str] = set()
        self._hidden_target_names: Set[str] = set()
        self._hidden_pick_names: Set[str] = set()

    # ------------------------------------------------------------------
    # Introspection helpers (exposed for tests)
    # ------------------------------------------------------------------

    @property
    def snapshotted_target_names(self) -> Set[str]:
        return set(self._snapshotted_target_names)

    @property
    def hidden_target_names(self) -> Set[str]:
        return set(self._hidden_target_names)

    @property
    def hidden_pick_names(self) -> Set[str]:
        return set(self._hidden_pick_names)

    @property
    def snapshot_threshold(self) -> float:
        """The Y coordinate at which the leading-edge triggers a snapshot."""
        return self._conveyor_end_y + self._snapshot_margin

    # ------------------------------------------------------------------
    # Core polling loop
    # ------------------------------------------------------------------

    def poll(self, simulation_time: float) -> None:
        """Poll targets and fire callbacks when their leading edge nears the conveyor edge.

        For each target:
          1. Compute ``target_leading_y = target.pos.y - target_half_y``.
          2. If a paired pick exists, also compute ``pick_leading_y`` and use
             whichever is further ahead (smaller Y) as the trigger position.
          3. Snapshot fires when the trigger Y drops below
             ``conveyor_end_y + snapshot_margin`` — once per target.
          4. Hide fires (on target and paired pick) when the target's leading
             edge drops below ``conveyor_end_y`` — once per target.

        Targets are matched to pairings by object identity so incremental
        spawning (which can reorder the target list) does not cause
        mis-attribution.
        """
        snap_threshold = self._conveyor_end_y + self._snapshot_margin
        completed_picks = self._strategy.completed_picks

        targets = list(self._target_objs_ref)  # snapshot the list
        for target in targets:
            name = getattr(target, "name", None)
            if name is None:
                continue

            try:
                pos, _ = target.get_world_pose()
            except Exception:
                logger.debug(
                    "ConveyorFalloffMonitor: get_world_pose failed on '%s'",
                    name, exc_info=True,
                )
                continue

            target_center_y = float(pos[1])
            target_half_y = self._get_half_extent_y(target)
            target_leading_y = target_center_y - target_half_y

            # Resolve paired pick (if any). Only a *completed* pick —
            # physically placed on the target and riding the belt with it —
            # contributes to the leading-edge trigger and gets hidden
            # alongside the target. An uncompleted paired pick (still in the
            # bin or being carried by the robot) is ignored: its world pose
            # is unrelated to the target's position, and hiding it would
            # cause cans to vanish mid-air or from the source bin.
            tgt_idx = self._find_target_index(target)
            placed_pick_name: Optional[str] = None
            placed_pick_obj = None
            pick_leading_y: Optional[float] = None
            if tgt_idx is not None:
                paired_pick_name = self._strategy.get_pick_name_for_target(name)
                if paired_pick_name is not None and paired_pick_name in completed_picks:
                    paired_pick_obj = self._strategy._pick_objs_by_name.get(
                        paired_pick_name
                    )
                    if paired_pick_obj is not None:
                        try:
                            p_pos, _ = paired_pick_obj.get_world_pose()
                            # Only treat as co-located if the pick is near
                            # the target in Y. A pick released elsewhere
                            # (e.g. recovery after a failed placement) would
                            # be far from the target and should be ignored.
                            proximity = abs(float(p_pos[1]) - target_center_y)
                            if proximity < max(target_half_y, 0.05) * 3:
                                placed_pick_name = paired_pick_name
                                placed_pick_obj = paired_pick_obj
                                p_half_y = self._get_half_extent_y(placed_pick_obj)
                                pick_leading_y = float(p_pos[1]) - p_half_y
                        except Exception:
                            logger.debug(
                                "ConveyorFalloffMonitor: get_world_pose failed on placed pick '%s'",
                                getattr(placed_pick_obj, "name", "?"),
                                exc_info=True,
                            )

            # Effective leading edge is whichever is further ahead.
            if pick_leading_y is not None:
                trigger_y = min(target_leading_y, pick_leading_y)
            else:
                trigger_y = target_leading_y

            # Snapshot trigger (one-shot per target)
            if (trigger_y < snap_threshold
                    and name not in self._snapshotted_target_names):
                logger.debug(
                    "ConveyorFalloffMonitor snapshot trigger '%s' at t=%.2fs: "
                    "target_center_y=%.3f target_leading_y=%.3f pick_leading_y=%s "
                    "trigger_y=%.3f threshold=%.3f end_y=%.3f margin=%.3f",
                    name, simulation_time, target_center_y, target_leading_y,
                    f"{pick_leading_y:.3f}" if pick_leading_y is not None else "None",
                    trigger_y, snap_threshold, self._conveyor_end_y, self._snapshot_margin,
                )
                if tgt_idx is None:
                    logger.debug(
                        "ConveyorFalloffMonitor: target '%s' not in current list; skipping snapshot",
                        name,
                    )
                else:
                    if placed_pick_name is not None:
                        try:
                            self._on_snapshot(placed_pick_name, name, simulation_time)
                        except Exception:
                            logger.exception(
                                "ConveyorFalloffMonitor: on_snapshot callback failed for pick='%s' tgt='%s'",
                                placed_pick_name, name,
                            )
                    else:
                        try:
                            self._on_available_lost(name, simulation_time)
                        except Exception:
                            logger.exception(
                                "ConveyorFalloffMonitor: on_available_lost callback failed for tgt='%s'",
                                name,
                            )
                self._snapshotted_target_names.add(name)

            # Hide trigger (one-shot per target). Only the target is hidden
            # here unless a *placed* pick is riding along; an uncompleted
            # pick (in the bin or mid-transport) stays visible.
            if (self._hide_after
                    and target_leading_y < self._conveyor_end_y
                    and name not in self._hidden_target_names):
                logger.debug(
                    "ConveyorFalloffMonitor hide trigger '%s' at t=%.2fs: target_leading_y=%.3f end_y=%.3f",
                    name, simulation_time, target_leading_y, self._conveyor_end_y,
                )
                try:
                    self._on_hide(target)
                except Exception:
                    logger.exception(
                        "ConveyorFalloffMonitor: on_hide callback failed for tgt='%s'", name,
                    )
                self._hidden_target_names.add(name)
                # Mark the target permanently unreachable immediately so
                # that re-pairing happens on the same tick — before the
                # placement behaviors see a stale/falling target position.
                unreachable_set = getattr(self._strategy, '_permanently_unreachable_targets', None)
                was_reachable = getattr(self._strategy, '_target_was_reachable', None)
                if unreachable_set is not None and name not in unreachable_set:
                    unreachable_set.add(name)
                    if was_reachable is not None:
                        was_reachable[name] = True  # ensure it counts as "was reachable"
                    logger.debug(
                        "ConveyorFalloffMonitor: marked target '%s' permanently unreachable at hide",
                        name,
                    )

                if (placed_pick_obj is not None
                        and placed_pick_obj.name not in self._hidden_pick_names):
                    try:
                        self._on_hide(placed_pick_obj)
                    except Exception:
                        logger.exception(
                            "ConveyorFalloffMonitor: on_hide callback failed for placed pick '%s'",
                            placed_pick_obj.name,
                        )
                    self._hidden_pick_names.add(placed_pick_obj.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_target_index(self, target: Any) -> Optional[int]:
        """Return the current index of ``target`` in the live target list (object identity)."""
        for i, obj in enumerate(self._target_objs_ref):
            if obj is target:
                return i
        return None


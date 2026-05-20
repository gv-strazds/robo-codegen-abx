"""TaskVerifier — long-lived owner of multi-pick-place verification state.

Decouples verification orchestration (PlacementChecker construction,
incremental & final verification, snapshot freezing, lost-target tracking,
merge logic) from MultiPickStrategy and UR10MultiPickPlaceTask.  The
verifier consumes the strategy only through its name-based policy
surface (``valid_targets_for_pick``, ``placement_constraints_satisfied``,
``is_pick_expected``).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TaskVerifier:
    """Owns verification orchestration over a task's lifetime.

    Constructs a :class:`PlacementChecker` (per-call engine) for every
    incremental, snapshot, or final verification pass — choosing
    box-containment vs marker mode based on the task's configuration.

    Holds the snapshot state populated by the conveyor fall-off monitor:
    frozen placement checks (authoritative per-pick verdicts captured at
    the moment a target nears the belt edge) and the list of targets that
    rolled off without any pick on them.
    """

    def __init__(
        self,
        *,
        pick_objs: list,
        strategy,
        bb_cache_factory: Callable,
        spatial_check_fn: Optional[Callable] = None,
        placement_constraints_fn: Optional[Callable] = None,
        containment_check: bool = False,
        box_verification_info: Optional[dict] = None,
        adjust_box_specs_fn: Optional[Callable] = None,
        on_incremental_check_fail: Optional[Callable] = None,
    ) -> None:
        self._pick_objs = pick_objs
        self._strategy = strategy
        self._bb_cache_factory = bb_cache_factory
        self._task_spatial_check_fn = spatial_check_fn
        self._task_placement_constraints_fn = placement_constraints_fn
        self._containment_check = containment_check
        self._box_verification_info = box_verification_info
        self._adjust_box_specs_fn = adjust_box_specs_fn
        self._on_incremental_check_fail = on_incremental_check_fail

        # Snapshot state owned by the verifier.  Frozen checks are keyed by
        # pick name (the strategy never sees them directly).
        self._frozen_checks: dict = {}
        # (target_name, sim_time) tuples for unfilled targets that crossed
        # the fall-off threshold.
        self._lost_available_targets: list = []

    # ------------------------------------------------------------------
    # Frozen / lost-target state (owned)
    # ------------------------------------------------------------------

    def freeze_check(self, check) -> None:
        """Record a PlacementCheck as authoritative for its pick.

        Subsequent live verification skips this pick; the frozen verdict is
        merged into the final report.  Re-freezing the same pick is a no-op.
        """
        if check.pick_name in self._frozen_checks:
            logger.warning(
                "freeze_check: pick '%s' already frozen; ignoring re-freeze",
                check.pick_name,
            )
            return
        self._frozen_checks[check.pick_name] = check

    def frozen_pick_names(self) -> set:
        """Return the set of pick names with frozen snapshot results."""
        return set(self._frozen_checks.keys())

    def frozen_pick_indices(self) -> set:
        """Return frozen pick indices for index-based callers (merge, live filter)."""
        return {check.pick_index for check in self._frozen_checks.values()}

    def frozen_checks_ordered(self) -> list:
        """Return frozen PlacementChecks ordered by pick_index."""
        return sorted(self._frozen_checks.values(), key=lambda c: c.pick_index)

    def frozen_target_names(self) -> set:
        """Return target names occupied by frozen passing snapshots.

        Used by the strategy's retargeting logic (set via
        ``MultiPickStrategy.set_frozen_target_names_fn``) to avoid
        reassigning a target that's already been counted as filled.
        """
        return {
            check.target_name for check in self._frozen_checks.values()
            if check.passed and check.target_name is not None
        }

    def lost_available_targets(self) -> list:
        """Return ``(target_name, sim_time)`` pairs in arrival order."""
        return list(self._lost_available_targets)

    def _drop_lost_target(self, target_name: str) -> None:
        """Remove any lost-available entry for *target_name* (retroactive snapshot)."""
        self._lost_available_targets = [
            entry for entry in self._lost_available_targets
            if entry[0] != target_name
        ]

    # ------------------------------------------------------------------
    # Resolution helpers (spatial check, placement constraints, valid targets)
    # ------------------------------------------------------------------

    def _resolve_spatial_check_fn(self) -> Callable:
        from task_verification import is_on_top
        return (
            self._strategy.get_spatial_check_fn()
            or self._task_spatial_check_fn
            or is_on_top
        )

    def _index_based_adapters(self):
        from task_verification import make_index_based_strategy_adapters
        return make_index_based_strategy_adapters(
            self._strategy, self._pick_objs, self._strategy.target_objs,
        )

    def _resolve_placement_constraints_fn(self) -> Callable:
        if self._task_placement_constraints_fn is not None:
            return self._task_placement_constraints_fn
        _, fn = self._index_based_adapters()
        return fn

    def _wrap_valid_targets_fn(self, base_fn: Callable) -> Callable:
        """Filter targets already claimed by frozen passing checks.

        Without this, a later pick's snapshot sees earlier (already retired
        off-belt) targets as still available, producing misleading "N valid
        targets available" failure diagnostics.
        """
        claimed = {
            c.target_index for c in self._frozen_checks.values()
            if c.passed and c.target_index is not None
        }
        if not claimed:
            return base_fn
        return lambda pick_idx: [t for t in base_fn(pick_idx) if t not in claimed]

    # ------------------------------------------------------------------
    # PlacementChecker construction
    # ------------------------------------------------------------------

    def _build_box_checker(self):
        if self._box_verification_info is None:
            return None
        from task_verification import PlacementChecker, build_box_verification_hooks

        box_specs = self._box_verification_info["box_specs"]
        if self._adjust_box_specs_fn is not None:
            box_specs = self._adjust_box_specs_fn(box_specs)
        box_targets, spatial_fn, valid_fn = build_box_verification_hooks(
            box_specs, self._pick_objs,
            is_pick_expected=self._strategy.is_pick_expected,
            extra_pick_check=self._box_verification_info.get("extra_pick_check"),
        )
        return PlacementChecker(
            pick_objs=self._pick_objs,
            target_objs=box_targets,
            spatial_check_fn=spatial_fn,
            valid_targets_fn=self._wrap_valid_targets_fn(valid_fn),
            placement_constraints_fn=self._resolve_placement_constraints_fn(),
            bb_cache_factory=self._bb_cache_factory,
            containment_mode=True,
        )

    def _build_marker_checker(self):
        from task_verification import PlacementChecker

        valid_targets_fn, _ = self._index_based_adapters()
        return PlacementChecker(
            pick_objs=self._pick_objs,
            target_objs=list(self._strategy.target_objs),
            spatial_check_fn=self._resolve_spatial_check_fn(),
            valid_targets_fn=self._wrap_valid_targets_fn(valid_targets_fn),
            placement_constraints_fn=self._resolve_placement_constraints_fn(),
            bb_cache_factory=self._bb_cache_factory,
            containment_mode=self._containment_check,
        )

    def _build_checker(self) -> tuple:
        box = self._build_box_checker()
        if box is not None:
            return box, "box"
        return self._build_marker_checker(), "marker"

    # ------------------------------------------------------------------
    # Verification API
    # ------------------------------------------------------------------

    def _names_to_indices(self, pick_names: list) -> list:
        """Translate pick names to per-call PlacementChecker indices."""
        if not pick_names:
            return []
        name_to_idx = {obj.name: i for i, obj in enumerate(self._pick_objs)}
        return [name_to_idx[n] for n in pick_names if n in name_to_idx]


    def verify_incremental(
        self, pick_names: list, *, simulation_time: float = 0.0,
    ) -> None:
        """Run incremental verification for *pick_names* and log per-item results.

        Successful checks whose target was prematurely recorded as "lost"
        (the snapshot trigger fired moments before MarkPickComplete landed)
        are retroactively frozen so they won't be re-verified live.
        """
        checker, _ = self._build_checker()
        result = checker.verify(pick_indices=self._names_to_indices(pick_names))
        for check in result.checks:
            if check.passed:
                logger.debug(
                    "Incremental check OK: '%s' -> '%s'",
                    check.pick_name, check.target_name,
                )
                self._maybe_retroactive_snapshot(check)
            else:
                logger.warning(
                    "Incremental check FAIL: '%s': %s",
                    check.pick_name, check.detail,
                )
                if self._on_incremental_check_fail is not None:
                    try:
                        self._on_incremental_check_fail(check, simulation_time)
                    except Exception as e:
                        logger.warning(
                            "on_incremental_check_fail callback raised: %s", e,
                        )

    def _maybe_retroactive_snapshot(self, check) -> None:
        """Freeze a passing incremental check if its target was prematurely recorded as lost."""
        if check.target_name is None:
            return
        for tgt_name, sim_time in self._lost_available_targets:
            if tgt_name == check.target_name:
                check.source = f"snapshot@{sim_time:.1f}s(retroactive)"
                self.freeze_check(check)
                self._drop_lost_target(tgt_name)
                logger.info(
                    "Retroactive snapshot for pick '%s' -> target '%s' "
                    "(target was recorded as lost at t=%.1fs but placement completed)",
                    check.pick_name, tgt_name, sim_time,
                )
                break

    def run_snapshot_verification(
        self, pick_name: str, target_name: str, simulation_time: float,
    ) -> None:
        """Freeze a placement check at current scene state for *pick_name*.

        Invoked by ConveyorFalloffMonitor when a target carrying a completed
        pick nears the conveyor edge.  The frozen check is authoritative —
        final verification skips the pick and merges this verdict in.

        ``target_name`` parameter mirrors the monitor's callback signature
        and surfaces in the failure log below when the strategy's resolved
        pairing disagrees with what the monitor expected.
        """
        pick_indices = self._names_to_indices([pick_name])
        if not pick_indices:
            logger.warning(
                "Snapshot verification: unknown pick_name=%r", pick_name,
            )
            return
        checker, _ = self._build_checker()
        result = checker.verify(pick_indices=pick_indices)
        if not result.checks:
            logger.warning(
                "Snapshot verification produced no check for pick=%r",
                pick_name,
            )
            return
        check = result.checks[0]
        check.source = f"snapshot@{simulation_time:.1f}s"
        self.freeze_check(check)
        verdict = "PASSED" if check.passed else "FAILED"
        target_label = check.target_name if check.target_name is not None else "(none)"
        if (check.target_name is not None
                and target_name != check.target_name):
            logger.warning(
                "Snapshot for pick '%s': monitor reported target '%s' but "
                "strategy resolved '%s'", pick_name, target_name, check.target_name,
            )
        logger.info(
            "Snapshot for pick '%s' -> target '%s': %s (t=%.1fs)",
            check.pick_name, target_label, verdict, simulation_time,
        )

    def record_available_target_lost(
        self, target_name: str, simulation_time: float,
    ) -> None:
        """Record an unfilled target crossing the fall-off edge (informational)."""
        self._lost_available_targets.append((target_name, simulation_time))
        logger.info(
            "Target '%s' was available but not filled in time (t=%.1fs)",
            target_name, simulation_time,
        )

    def verify_final(self) -> tuple:
        """Ground-truth final verification with snapshot merge.

        Returns:
            (success, failures) tuple — same shape as the legacy
            ``check_groundtruth_task_success`` API.
        """
        if not self._pick_objs or not self._strategy.target_objs:
            logger.warning(
                "Ground-truth check skipped: picks or targets are missing.",
            )
            return True, []

        frozen_idx = self.frozen_pick_indices()
        total_picks = len(self._pick_objs)
        live_idx = [i for i in range(total_picks) if i not in frozen_idx]

        checker, _mode = self._build_checker()
        live_result = checker.verify(pick_indices=live_idx)

        # Targets claimed by frozen passing snapshots OR by passing live
        # checks; anything else recorded as lost stays in the info lines.
        passed_target_names = self.frozen_target_names()
        for c in live_result.checks:
            if c.passed and c.target_name is not None:
                passed_target_names.add(c.target_name)
        info_lines = [
            f"Target '{name}' was available but not filled in time (t={t:.1f}s)"
            for (name, t) in self._lost_available_targets
            if name not in passed_target_names
        ]

        if not frozen_idx and not info_lines:
            if not live_result.success:
                logger.info(live_result.summary())
            return (live_result.success, live_result.failures)

        from task_verification import merge_verification_results
        merged = merge_verification_results(
            frozen_checks=self.frozen_checks_ordered(),
            live_result=live_result,
            pick_count=total_picks,
            info_lines=info_lines,
        )
        if not merged.success or info_lines:
            logger.info(merged.summary())
        return (merged.success, merged.failures)

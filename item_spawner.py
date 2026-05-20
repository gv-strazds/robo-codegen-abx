"""ItemSpawner — per-task owner of item-spawning schedulers and prim creation.

Encapsulates the pick + target schedulers (``IncrementalItemScheduler`` /
``SpatialTriggeredItemScheduler``) and the dispatch to a ``PrimFactory``
that materialises new prims into the live scene.  Returns new prims via
:class:`SpawnResult`; the caller (typically
``UR10MultiPickPlaceTask.pre_step``) is responsible for notifying
``MultiPickStrategy.add_incremental_picks`` / ``add_incremental_targets``.
Keeping the strategy notification on the caller preserves the
configurator-spawns / strategy-policy boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class SpawnResult:
    """Outcome of a single :meth:`ItemSpawner.tick` call.

    ``new_picks`` / ``new_targets`` are the prims that were just spawned
    (empty when the schedulers had nothing to release this tick).
    ``all_*_released`` reflects whether the corresponding scheduler has
    handed out its full quota — readers use it to clear the strategy's
    ``more_*_expected`` flags.
    """

    new_picks: list = field(default_factory=list)
    new_targets: list = field(default_factory=list)
    all_picks_released: bool = False
    all_targets_released: bool = False


class PrimFactory(Protocol):
    """Creates scene prims from ``ItemSpec`` lists for the live scene.

    Implementations must create the prims, cache any geometry they
    need, and return the list of new prim objects.  They must NOT
    append the new prims to any shared list — that responsibility
    lives with the spawner's caller, which has the right context to
    coordinate the strategy notification.
    """

    def create_picks(self, items: list, scene) -> list: ...

    def create_targets(self, items: list, scene) -> list: ...


class IsaacPrimFactory:
    """Self-contained ``PrimFactory`` for the live Isaac Sim scene.

    Owns the per-task name counters and the assets-root / geometry-cache
    state required to materialise picks and targets, so that the same
    factory instance can be used for both the initial-batch setup (called
    from ``SimulationConfigurator``) and runtime replenishment (called
    from ``ItemSpawner.tick`` after the configurator has been discarded).

    Decoupling from :class:`SimulationConfigurator` keeps the runtime
    free of back-channel references into the setup-only configurator,
    and lets the strategy be the single owner of pick / target lists.
    """

    def __init__(
        self,
        *,
        prim_geometry: dict,
        assets_root_path: Optional[str],
        bb_cache_factory: Callable,
        pick_name_seq: int = 0,
        target_name_seq: int = 0,
    ) -> None:
        self._prim_geometry = prim_geometry
        self._assets_root_path = assets_root_path
        self._bb_cache_factory = bb_cache_factory
        self._pick_name_seq = pick_name_seq
        self._target_name_seq = target_name_seq

    @property
    def pick_name_seq(self) -> int:
        return self._pick_name_seq

    @property
    def target_name_seq(self) -> int:
        return self._target_name_seq

    def create_picks(self, items: list, scene) -> list:
        """Materialise pick prims for *items* in *scene* and cache geometry.

        Returned prims are NOT appended to any external list — callers
        (setup-time configurator or runtime spawner) are responsible for
        registering them with whichever owner needs them.
        """
        import numpy as np
        from asset_utils import add_asset, get_or_compute_prim_geometry

        new_prims = []
        prim_asset_info: dict = {}

        for item in items:
            asset_type = item.asset_type
            obj_name = item.name if item.name else f"{asset_type}_{self._pick_name_seq}"
            self._pick_name_seq += 1

            color = item.color
            if color is None:
                color = np.random.uniform(size=(3,))

            prim = add_asset(
                scene,
                asset_type=asset_type,
                obj_name=obj_name,
                position=item.position,
                orientation=item.orientation,
                scale=item.scale,
                scene_path_root="/World/",
                assets_root_path=self._assets_root_path,
                color=color,
                visible=not item.hidden,
            )
            prim_asset_info[prim.name] = (asset_type, item.scale, item.orientation)
            new_prims.append(prim)

        cache = self._bb_cache_factory()
        for prim in new_prims:
            self._prim_geometry[prim.name] = get_or_compute_prim_geometry(
                prim,
                asset_type_default=None,
                prim_asset_info=prim_asset_info.get(prim.name),
                bb_cache=cache,
            )

        logger.debug("create_picks: spawned %d objects", len(new_prims))
        return new_prims

    def create_targets(self, items: list, scene) -> list:
        """Materialise target prims for *items* in *scene* and cache geometry."""
        import numpy as np
        from asset_utils import add_asset, get_or_compute_prim_geometry

        new_prims = []
        prim_asset_info: dict = {}

        for item in items:
            asset_type = item.asset_type
            obj_name = item.name if item.name else f"target_{self._target_name_seq}"
            self._target_name_seq += 1

            color = (
                item.color if item.color is not None else np.array([0, 0, 1])
            )  # Default blue

            prim = add_asset(
                scene,
                asset_type=asset_type,
                obj_name=obj_name,
                prim_path="/Targets/" + obj_name,
                position=item.position,
                orientation=item.orientation,
                scale=item.scale,
                assets_root_path=self._assets_root_path,
                color=color,
                visible=not item.hidden,
            )
            prim_asset_info[prim.name] = (asset_type, item.scale, item.orientation)
            new_prims.append(prim)

        cache = self._bb_cache_factory()
        for prim in new_prims:
            self._prim_geometry[prim.name] = get_or_compute_prim_geometry(
                prim,
                asset_type_default=None,
                prim_asset_info=prim_asset_info.get(prim.name),
                bb_cache=cache,
            )

        logger.debug("create_targets: spawned %d objects", len(new_prims))
        return new_prims


def _live_xy(prims) -> List[tuple]:
    """Snapshot live world-frame (x, y) for the given prims.

    Returns a list of ``(x, y)`` tuples, skipping prims whose pose
    query raises.  Used by spatial-trigger schedulers for region
    predicates.
    """
    out: List[tuple] = []
    for p in prims:
        try:
            pos, _ = p.get_world_pose()
            out.append((float(pos[0]), float(pos[1])))
        except Exception:
            pass
    return out


class ItemSpawner:
    """Per-task owner of pick + target schedulers and the spawn flow.

    The spawner does NOT know about ``MultiPickStrategy``.  Its
    :meth:`tick` returns a :class:`SpawnResult` carrying the newly
    spawned prims; the caller dispatches them to the strategy.
    """

    def __init__(
        self,
        *,
        pick_scheduler=None,
        target_scheduler=None,
        prim_factory: PrimFactory,
        scene=None,
        conveyor_speed_fn: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self._pick_scheduler = pick_scheduler
        self._target_scheduler = target_scheduler
        self._prim_factory = prim_factory
        self._scene = scene
        self._conveyor_speed_fn = conveyor_speed_fn

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def pick_scheduler(self):
        return self._pick_scheduler

    @pick_scheduler.setter
    def pick_scheduler(self, scheduler) -> None:
        self._pick_scheduler = scheduler

    @property
    def target_scheduler(self):
        return self._target_scheduler

    @target_scheduler.setter
    def target_scheduler(self, scheduler) -> None:
        self._target_scheduler = scheduler

    @property
    def has_pick_scheduler(self) -> bool:
        return self._pick_scheduler is not None

    @property
    def has_target_scheduler(self) -> bool:
        return self._target_scheduler is not None

    @property
    def more_picks_expected(self) -> bool:
        """True iff a pick scheduler is configured and has items pending."""
        return (
            self._pick_scheduler is not None
            and not self._pick_scheduler.all_released
        )

    @property
    def more_targets_expected(self) -> bool:
        """True iff a target scheduler is configured and has items pending."""
        return (
            self._target_scheduler is not None
            and not self._target_scheduler.all_released
        )

    # ------------------------------------------------------------------
    # Scheduler-type helpers
    # ------------------------------------------------------------------

    def is_spatial(self, scheduler) -> bool:
        from item_generation import SpatialTriggeredItemScheduler
        return isinstance(scheduler, SpatialTriggeredItemScheduler)

    def is_spatial_scheduler_inert(self, scheduler) -> bool:
        """A spatial scheduler is inert when the conveyor is stationary.

        Replenishment will never fire, so callers must stop waiting on
        ``more_*_expected`` or the BT idles forever.
        """
        if not self.is_spatial(scheduler):
            return False
        speed = self._conveyor_speed_fn() if self._conveyor_speed_fn else None
        return speed is None or float(speed) == 0.0

    # ------------------------------------------------------------------
    # BT-start gate
    # ------------------------------------------------------------------

    def bt_should_start(self, simulation_time: float) -> bool:
        """Return True iff at least one configured scheduler says BT can start.

        With both schedulers present we OR their readiness signals — BT
        starts as soon as either side is ready; per-cycle stalls are
        handled by the ``more_*_expected`` flags.  With one scheduler,
        the gate reduces to that scheduler's threshold.  With neither
        scheduler configured the spawner should not be invoked at all
        (caller short-circuits before construction); the True fallback
        is defensive.
        """
        checks: List[bool] = []
        if self._pick_scheduler is not None:
            checks.append(self._pick_scheduler.bt_should_start(simulation_time))
        if self._target_scheduler is not None:
            checks.append(self._target_scheduler.bt_should_start(simulation_time))
        return any(checks) if checks else True

    # ------------------------------------------------------------------
    # Per-tick spawn
    # ------------------------------------------------------------------

    def tick(
        self,
        simulation_time: float,
        *,
        live_picks: Optional[list] = None,
        live_targets: Optional[list] = None,
    ) -> SpawnResult:
        """Tick both schedulers; spawn any released items; return a SpawnResult.

        ``live_picks`` / ``live_targets`` are the currently-alive prim
        lists, used only by spatial-trigger schedulers for region
        predicates.  Time-based schedulers ignore them.
        """
        result = SpawnResult()
        if self._pick_scheduler is not None:
            if not self._pick_scheduler.all_released:
                new_items = self._tick_one(
                    self._pick_scheduler, simulation_time, live_picks or [],
                )
                if new_items:
                    result.new_picks = self._prim_factory.create_picks(
                        new_items, self._scene,
                    )
            result.all_picks_released = bool(self._pick_scheduler.all_released)
        if self._target_scheduler is not None:
            if not self._target_scheduler.all_released:
                new_items = self._tick_one(
                    self._target_scheduler, simulation_time, live_targets or [],
                )
                if new_items:
                    result.new_targets = self._prim_factory.create_targets(
                        new_items, self._scene,
                    )
            result.all_targets_released = bool(self._target_scheduler.all_released)
        return result

    def _tick_one(self, scheduler, simulation_time, live_prims) -> list:
        """Dispatch ``scheduler.tick`` with the right args for its type."""
        if self.is_spatial(scheduler):
            speed = self._conveyor_speed_fn() if self._conveyor_speed_fn else None
            return scheduler.tick(
                simulation_time,
                current_xy=_live_xy(live_prims),
                conveyor_speed=speed,
            )
        return scheduler.tick(simulation_time)

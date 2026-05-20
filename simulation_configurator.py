"""SimulationConfigurator: builds simulation scene state for the task.

Setup-phase factory.  During ``set_up_scene`` / ``post_reset`` it
constructs pick/target object lists, geometry caches, and the
``TaskVerifier``.  Those services are then transferred onto
``TaskContext`` for the runtime phase, at which point the
configurator is discarded.

Owns during the setup window:
- Pick and target object lists
- PrimGeometry cache (shared with the prim factory)
- An :class:`IsaacPrimFactory` used for both setup-time spawn and
  later runtime replenishment via ``ItemSpawner``.
- Object generation (``add_source_objects``, ``add_target_objects``)
- ``build_verifier`` factory (returns a ``TaskVerifier``; the verifier
  itself is then owned by ``TaskContext``).
"""
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def _create_fresh_bb_cache():
    """Return a fresh bounding-box cache.

    Free-function factory so the verifier doesn't capture a configurator
    reference through its closure.  Mock tests monkey-patch
    ``task_verification.create_bbox_cache`` to swap in their stub.
    """
    from isaacsim.core.utils.bounds import create_bbox_cache
    return create_bbox_cache()


class SimulationConfigurator:
    """Manages scene objects, geometry cache, and verification for pick-place tasks.

    Args:
        pick_generation_strategy: ItemGenerator (or duck-typed) for pick objects.
        target_generation_strategy: ItemGenerator (or duck-typed) for target objects.
        pick_count: Number of pick objects (int, tuple range, or None).
        target_count: Number of target objects (int, tuple range, or None).
        seed: Random seed for reproducible generation.
        assets_root_path: Root path for USD assets.
    """

    def __init__(
        self,
        pick_generation_strategy=None,
        target_generation_strategy=None,
        pick_count: Optional[Union[int, tuple]] = None,
        target_count: Optional[Union[int, tuple]] = None,
        seed: Optional[int] = None,
        assets_root_path: Optional[str] = None,
    ):
        self.pick_generation_strategy = pick_generation_strategy
        self.target_generation_strategy = target_generation_strategy
        self._pick_count = pick_count
        self._target_count = target_count
        self._seed = seed
        self._assets_root_path = assets_root_path

        self._pick_objs: list = []
        self._target_objs: list = []
        self._prim_geometry: dict = {}
        self._staged_spawner = None
        # The factory owns prim-creation, name counters, and the bb-cache
        # factory closure.  Same instance is used for setup-time spawn
        # (add_source_objects / add_target_objects / initial-batch helpers)
        # and is later handed to ItemSpawner for runtime replenishment, so
        # name counters survive the configurator being dropped.
        from item_spawner import IsaacPrimFactory
        self._prim_factory = IsaacPrimFactory(
            prim_geometry=self._prim_geometry,
            assets_root_path=self._assets_root_path,
            bb_cache_factory=_create_fresh_bb_cache,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pick_objs(self) -> list:
        return self._pick_objs

    @property
    def target_objs(self) -> list:
        return self._target_objs

    @property
    def prim_geometry(self) -> dict:
        return self._prim_geometry

    @property
    def prim_factory(self):
        return self._prim_factory

    def build_verifier(
        self,
        *,
        strategy,
        spatial_check_fn=None,
        placement_constraints_fn=None,
        containment_check: bool = False,
        box_verification_info=None,
        adjust_box_specs_fn=None,
        on_incremental_check_fail=None,
    ):
        """Build a TaskVerifier for this task and return it.

        The caller is responsible for wiring the verifier's
        ``frozen_target_names`` callback into the strategy
        (``strategy.set_frozen_target_names_fn(...)``) so retargeting
        skips targets already claimed by a passing snapshot.
        """
        from task_verifier import TaskVerifier
        return TaskVerifier(
            pick_objs=self._pick_objs,
            strategy=strategy,
            bb_cache_factory=_create_fresh_bb_cache,
            spatial_check_fn=spatial_check_fn,
            placement_constraints_fn=placement_constraints_fn,
            containment_check=containment_check,
            box_verification_info=box_verification_info,
            adjust_box_specs_fn=adjust_box_specs_fn,
            on_incremental_check_fail=on_incremental_check_fail,
        )

    def stage_spawner_from(
        self,
        *,
        pick_scheduler,
        target_scheduler,
        scene,
        conveyor_speed_fn=None,
    ):
        """Wrap pre-built schedulers in an ``ItemSpawner`` and stash it.

        Either scheduler may be ``None``; if both are ``None`` no
        spawner is staged.  Returns the staged spawner (or ``None``);
        ``UR10MultiPickPlaceTask.post_reset`` reads ``_staged_spawner``
        to transfer onto the runtime context.
        """
        if pick_scheduler is None and target_scheduler is None:
            self._staged_spawner = None
            return None
        from item_spawner import ItemSpawner
        self._staged_spawner = ItemSpawner(
            pick_scheduler=pick_scheduler,
            target_scheduler=target_scheduler,
            prim_factory=self._prim_factory,
            scene=scene,
            conveyor_speed_fn=conveyor_speed_fn,
        )
        return self._staged_spawner

    # ------------------------------------------------------------------
    # Object generation
    # ------------------------------------------------------------------

    def add_source_objects(self, scene) -> None:
        """Add source (pickable) objects to the scene using the generation strategy."""
        if not self.pick_generation_strategy:
            logger.warning(
                "No pick_generation_strategy provided. Skipping source object generation."
            )
            return

        items = self.pick_generation_strategy.generate(
            count_range=self._pick_count, seed=self._seed
        )
        self._pick_objs = list(self._prim_factory.create_picks(items, scene))

    def add_target_objects(self, scene) -> None:
        """Add target/drop-off objects to the scene using the generation strategy."""
        if not self.target_generation_strategy:
            return

        items = self.target_generation_strategy.generate(
            count_range=self._target_count, seed=self._seed
        )
        self._target_objs = list(self._prim_factory.create_targets(items, scene))


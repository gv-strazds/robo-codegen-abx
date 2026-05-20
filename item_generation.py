import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
import numpy as np
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ItemSpec:
    """Specification for a generated item."""
    asset_type: str
    position: np.ndarray
    orientation: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    color: Optional[Union[str, np.ndarray]] = None
    name: Optional[str] = None
    hidden: bool = False

def resolve_count(count_range, capacity=None, seed=None):
    """Resolve a count_range to a concrete integer.

    Args:
        count_range: int, (min, max) tuple, or None.
        capacity: Fallback when count_range is None or max is None.
        seed: Random seed for sampling from a range.

    Returns:
        Concrete integer count, or None if count_range is None and no capacity.
    """
    if count_range is None:
        return capacity
    if isinstance(count_range, int):
        return count_range
    min_count, max_count = count_range
    if max_count is None:
        max_count = capacity if capacity is not None else min_count
    if max_count < min_count:
        max_count = min_count
    return random.Random(seed).randint(min_count, max_count)


class AttributeStrategy(ABC):
    @abstractmethod
    def get_value(self, index: int, total_items: int, seed: Optional[int] = None) -> Any:
        pass

class FixedValue(AttributeStrategy):
    def __init__(self, value):
        self._value = value
    
    def get_value(self, index: int, total_items: int, seed: Optional[int] = None) -> Any:
        return self._value

class RandomChoice(AttributeStrategy):
    def __init__(self, options: List[Any]):
        self._options = options
        
    def get_value(self, index: int, total_items: int, seed: Optional[int] = None) -> Any:
        rng = random.Random(seed + index if seed is not None else None)
        return rng.choice(self._options)

class SequentialChoice(AttributeStrategy):
    def __init__(self, options: List[Any], loop: bool = True):
        self._options = options
        self._loop = loop
        
    def get_value(self, index: int, total_items: int, seed: Optional[int] = None) -> Any:
        if not self._options:
            return None
        i = index % len(self._options) if self._loop else min(index, len(self._options) - 1)
        return self._options[i]

class MixedScaleStrategy(AttributeStrategy):
    """Strategy that scales primitives while keeping USD assets at identity scale.
    
    Args:
        types: List of asset type strings corresponding to each item.
        default_scale: The scale to apply to primitive assets.
    """
    def __init__(self, types: List[str], default_scale: np.ndarray):
        self.types = types
        self.default_scale = default_scale

    def get_value(self, index: int, total: int, seed: Optional[int] = None) -> Any:
        from asset_data_utils import PRIM_TYPES
        asset_type = self.types[index % len(self.types)]
        if asset_type in ["cylinder", "cone"]:
            return np.array([
                self.default_scale[0] * 0.5,
                self.default_scale[1] * 0.5,
                self.default_scale[2] * 2.0
            ])
        if asset_type in PRIM_TYPES:
            return self.default_scale
        return np.array([1.0, 1.0, 1.0])

class MixedOrientationStrategy(AttributeStrategy):
    """Strategy that applies a -90 degree X-axis rotation to boxes, bottles, and cans.

    Args:
        types: List of asset type strings corresponding to each item.
    """
    def __init__(self, types: List[str]):
        self.types = types

    def get_value(self, index: int, total: int, seed: Optional[int] = None) -> Any:
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        asset_type = self.types[index % len(self.types)]
        if asset_type in ["cracker_box", "sugar_box", "mustard_bottle", "soup_can"]:
            return euler_angles_to_quat(np.array([-np.pi / 2.0, 0, 0]))
        return None


class PositionGenerator(ABC):
    @abstractmethod
    def get_positions(self, count: int, seed: Optional[int] = None) -> List[np.ndarray]:
        pass

    @property
    def capacity(self) -> Optional[int]:
        """Return the maximum number of positions this generator can provide, or None if infinite/variable."""
        return None

class GridPositionGenerator(PositionGenerator):
    def __init__(self, center: np.ndarray, rows: int, cols: int, spacing_x: float, spacing_y: float, z_offset: float = 0.0, randomize: bool = True):
        self.center = np.array(center)
        self.rows = rows
        self.cols = cols
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.z_offset = z_offset
        self.randomize = randomize

    @property
    def capacity(self) -> Optional[int]:
        return self.rows * self.cols

    def get_positions(self, count: int, seed: Optional[int] = None) -> List[np.ndarray]:
        positions = []
        # Calculate grid origin (top-left or centered)
        # Assuming centered grid
        total_width = (self.cols - 1) * self.spacing_x
        total_length = (self.rows - 1) * self.spacing_y
        
        start_x = self.center[0] - total_width / 2
        start_y = self.center[1] - total_length / 2
        
        all_possible_indices = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        rng = random.Random(seed)

        if self.randomize:
            # If randomize is True, we shuffle the available slots and pick the first 'count' ones.
            # This ensures we pick random slots even if count < capacity.
            rng.shuffle(all_possible_indices)
            selected_indices = all_possible_indices[:count]
        else:
            # If randomize is False, we pick slots sequentially.
            selected_indices = all_possible_indices[:count]

        for r, c in selected_indices:
            x = start_x + c * self.spacing_x
            y = start_y + r * self.spacing_y
            z = self.center[2] + self.z_offset
            positions.append(np.array([x, y, z]))

        return positions

class CircularPositionGenerator(PositionGenerator):
    def __init__(self, center: np.ndarray, radius: float, z_offset: float = 0.0, count: Optional[int] = None, randomize: bool = False):
        self.center = np.array(center)
        self.radius = radius
        self.z_offset = z_offset
        self._count = count
        self.randomize = randomize

    @property
    def capacity(self) -> Optional[int]:
        return self._count

    def get_positions(self, count: int, seed: Optional[int] = None) -> List[np.ndarray]:
        positions = []
        rng = random.Random(seed)

        if self.randomize:
            # If randomize is True, we treat the circle as having discrete slots.
            # The number of slots is defined by self._count (capacity).
            # If self._count is None, we default to the requested count (which effectively disables slot randomness, 
            # but still might randomize ordering if we shuffled, but here we define 'slots' first).
            # The user requirement says: "max_count (capacity) should by default be used to set the number of slots".
            
            num_slots = self._count
            if num_slots is None:
                 # Fallback: if no capacity defined, the circle has 'count' slots, so we just use that.
                 num_slots = count
            
            # Generate all slot indices
            all_slots = list(range(num_slots))
            
            # We want to pick `count` unique slots from `num_slots`.
            # If count > num_slots, we can wrap around? Or simpler: max out at num_slots.
            # Usually count <= capacity.
            
            if count > num_slots:
                # If we need more than available slots, we can resample or just limit.
                # Standard behavior is usually limited by capacity.
                # But let's assume we pick 'count' indices from the available slots.
                # If we need strict unique slots, we can't exceed num_slots.
                # If we allow reuse, we use choices.
                # Given "choose which slots to fill randomly (leaving some empty)", it implies unique filling of available slots.
                # So we cap at num_slots.
                count_to_pick = min(count, num_slots)
            else:
                count_to_pick = count

            # Shuffle slots and pick
            rng.shuffle(all_slots)
            selected_indices = all_slots[:count_to_pick]
            
            for i in selected_indices:
                angle = 2 * np.pi * i / num_slots
                x = self.center[0] + self.radius * np.cos(angle)
                y = self.center[1] + self.radius * np.sin(angle)
                z = self.center[2] + self.z_offset
                positions.append(np.array([x, y, z]))
                
        else:
            # Original behavior: dynamic spacing based on requested count (or sequential filling if we interpreted it that way, 
            # but the original code did '2 * np.pi * i / count', which means standard even spacing for THAT count)
            for i in range(count):
                angle = 2 * np.pi * i / count
                x = self.center[0] + self.radius * np.cos(angle)
                y = self.center[1] + self.radius * np.sin(angle)
                z = self.center[2] + self.z_offset
                positions.append(np.array([x, y, z]))

        return positions

class ConveyorPositionGenerator(PositionGenerator):
    def __init__(self, center_x: float, center_y: float, z: float, spacing: float, jitter_x: float = 0.0, jitter_y: float = 0.0):
        self.center_x = center_x
        self.center_y = center_y
        self.z = z
        self.spacing = spacing
        self.jitter_x = jitter_x
        self.jitter_y = jitter_y

    def get_positions(self, count: int, seed: Optional[int] = None) -> List[np.ndarray]:
        rng = random.Random(seed)
        start_y = self.center_y - self.spacing * (count - 1) / 2
        positions = []
        for i in range(count):
            jx = rng.uniform(-self.jitter_x, self.jitter_x) if self.jitter_x > 0 else 0
            jy = rng.uniform(-self.jitter_y, self.jitter_y) if self.jitter_y > 0 else 0
            pos = np.array([self.center_x + jx, start_y + i * self.spacing + jy, self.z])
            positions.append(pos)
        return positions

class LayeredPositionGenerator(PositionGenerator):
    """Wraps a base PositionGenerator and replicates its positions across multiple Z layers.

    Fills bottom-up: layer 0 first, then layer 1, etc. Same XY positions in each layer,
    offset by layer_idx * layer_height in Z. Supports partial top layers when
    count < capacity.

    Note: For stacked items in a real physics sim, picks should be done top-down
    (reverse order) to avoid pulling from under a stack. This is a separate concern
    from position generation — the pick strategy would handle reordering.
    """

    def __init__(self, base_generator: PositionGenerator, num_layers: int, layer_height: float):
        self._base = base_generator
        self._num_layers = num_layers
        self._layer_height = layer_height

    @property
    def capacity(self) -> Optional[int]:
        base_cap = self._base.capacity
        return base_cap * self._num_layers if base_cap is not None else None

    def get_positions(self, count: int, seed: Optional[int] = None) -> List[np.ndarray]:
        base_cap = self._base.capacity
        if base_cap is not None:
            count = min(count, base_cap * self._num_layers)

        # Number of positions per full layer
        per_layer = base_cap if base_cap is not None else count
        # Get one full layer of base positions
        base_positions = self._base.get_positions(per_layer, seed)

        positions = []
        remaining = count
        for layer_idx in range(self._num_layers):
            if remaining <= 0:
                break
            n_this_layer = min(remaining, len(base_positions))
            for i in range(n_this_layer):
                pos = base_positions[i].copy()
                pos[2] += layer_idx * self._layer_height
                positions.append(pos)
            remaining -= n_this_layer

        return positions


class ItemGenerator:
    def __init__(
        self,
        position_generator: PositionGenerator,
        asset_type_strategy: AttributeStrategy = None,
        orientation_strategy: AttributeStrategy = None,
        scale_strategy: AttributeStrategy = None,
        color_strategy: AttributeStrategy = None,
        hidden_strategy: AttributeStrategy = None,
    ):
        self.position_generator = position_generator
        self.asset_type_strategy = asset_type_strategy or FixedValue("cube")
        self.orientation_strategy = orientation_strategy or FixedValue(None)
        self.scale_strategy = scale_strategy or FixedValue(None)
        self.color_strategy = color_strategy or FixedValue(None)
        self.hidden_strategy = hidden_strategy or FixedValue(False)

    def generate(self, count_range: Union[int, tuple, None] = None, seed: Optional[int] = None) -> List[ItemSpec]:
        rng = random.Random(seed)
        
        if count_range is None:
            # Default to capacity if available, otherwise 1
            cap = self.position_generator.capacity
            count = cap if cap is not None else 1
        elif isinstance(count_range, int):
            count = count_range
        else:
            # Tuple range
            min_count = count_range[0]
            max_count = count_range[1]
            if max_count is None:
                cap = self.position_generator.capacity
                # If capacity is None (infinite), we can't really guess a max, so we just use min_count?
                # Or maybe strictly min_count.
                # User request: "max value should be determined by the default capacity for the generators used by the task"
                if cap is not None:
                     max_count = cap
                else:
                     # Fallback if capacity is infinite: just use min_count to be safe, 
                     # or maybe we should default to something else? 
                     # For now, let's treat it as a fixed value of min_count if unlimited.
                     max_count = min_count
            
            # Ensure valid range
            if max_count < min_count:
                max_count = min_count
                
            count = rng.randint(min_count, max_count)
            
        positions = self.position_generator.get_positions(count, seed)
        # Count might be limited by available positions (e.g. grid)
        final_count = min(count, len(positions))
        
        items = []
        for i in range(final_count):
            items.append(ItemSpec(
                asset_type=self.asset_type_strategy.get_value(i, final_count, seed),
                position=positions[i],
                orientation=self.orientation_strategy.get_value(i, final_count, seed),
                scale=self.scale_strategy.get_value(i, final_count, seed),
                color=self.color_strategy.get_value(i, final_count, seed),
                hidden=self.hidden_strategy.get_value(i, final_count, seed),
                name=None # Name is usually assigned by task logic, or we could add a strategy for it
            ))

        return items


# ---------------------------------------------------------------------------
# Incremental generation
# ---------------------------------------------------------------------------


@dataclass
class IncrementalGenerationConfig:
    """Configuration for incremental/time-based item generation.

    Items are spawned in batches of ``items_per_batch`` every
    ``batch_interval`` seconds.  The BT start is gated by a configurable
    threshold so that generation and task execution can overlap:

    - ``bt_start_threshold=None`` (default): wait for all items before
      the BT starts ticking.
    - ``bt_start_threshold=0``: start the BT immediately; items arrive
      while the robot is already processing picks.
    - ``bt_start_threshold=N`` (N>0): start the BT after at least *N*
      items have been spawned.
    - ``bt_start_delay=T``: start the BT *T* seconds after generation
      begins.

    When both ``bt_start_threshold`` and ``bt_start_delay`` are set the
    BT starts when *either* condition is satisfied (whichever comes
    first).
    """

    items_per_batch: int = 1
    batch_interval: float = 0.5
    bt_start_threshold: Optional[int] = None
    bt_start_delay: Optional[float] = None


class IncrementalItemScheduler:
    """Wraps an ItemGenerator to release items in timed batches.

    All ``ItemSpec`` instances are pre-generated at construction time
    (positions are pre-allocated by the ``PositionGenerator``).  The
    scheduler then parcels them out based on elapsed simulation time.

    Usage::

        scheduler = IncrementalItemScheduler(generator, config, count_range, seed)
        initial = scheduler.get_initial_batch()   # spawn these in set_up_scene
        # Each simulation step:
        new_items = scheduler.tick(simulation_time)
        if new_items:
            spawn_and_register(new_items)
    """

    def __init__(
        self,
        generator: "ItemGenerator",
        config: IncrementalGenerationConfig,
        count_range: Union[int, tuple, None] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._all_items: List[ItemSpec] = generator.generate(
            count_range=count_range, seed=seed,
        )
        # Release the first batch immediately.
        first_batch_size = min(config.items_per_batch, len(self._all_items))
        self._released_count: int = first_batch_size
        self._last_batch_time: Optional[float] = None
        self._generation_start_time: Optional[float] = None

    # -- properties ----------------------------------------------------------

    @property
    def config(self) -> IncrementalGenerationConfig:
        return self._config

    @property
    def total_count(self) -> int:
        return len(self._all_items)

    @property
    def released_count(self) -> int:
        return self._released_count

    @property
    def pending_count(self) -> int:
        return max(0, len(self._all_items) - self._released_count)

    @property
    def all_released(self) -> bool:
        return self._released_count >= len(self._all_items)

    # -- batch access --------------------------------------------------------

    def get_initial_batch(self) -> List[ItemSpec]:
        """Return the first batch of items (already counted as released)."""
        return list(self._all_items[: self._released_count])

    def tick(self, current_time: float) -> List[ItemSpec]:
        """Advance the scheduler clock and return any newly released items.

        Args:
            current_time: Current simulation time in seconds.

        Returns:
            List of newly released ``ItemSpec`` instances (may be empty).
        """
        if self.all_released:
            return []

        # Record generation start on first tick.
        if self._generation_start_time is None:
            self._generation_start_time = current_time
            self._last_batch_time = current_time
            return []  # first batch was already released via get_initial_batch

        elapsed_since_last = current_time - self._last_batch_time
        if elapsed_since_last < self._config.batch_interval:
            return []

        # Release next batch.
        start = self._released_count
        end = min(start + self._config.items_per_batch, len(self._all_items))
        self._released_count = end
        self._last_batch_time = current_time

        new_items = self._all_items[start:end]
        if new_items:
            logger.debug(
                "IncrementalItemScheduler: released %d items (%d/%d) at t=%.3f",
                len(new_items), self._released_count, len(self._all_items),
                current_time,
            )
        return new_items

    # -- BT start gate -------------------------------------------------------

    def bt_should_start(self, current_time: float) -> bool:
        """Return True when the BT gate should open.

        The gate opens when *any* of the following is true:

        * All items have been released.
        * ``bt_start_threshold`` is 0 (start immediately).
        * ``released_count >= bt_start_threshold`` (enough items).
        * Elapsed time since generation began >= ``bt_start_delay``.
        * Both threshold and delay are ``None`` and all items released
          (the default — safest behaviour).
        """
        if self.all_released:
            return True

        cfg = self._config

        if cfg.bt_start_threshold is not None:
            if cfg.bt_start_threshold == 0:
                return True
            if self._released_count >= cfg.bt_start_threshold:
                return True

        if cfg.bt_start_delay is not None and self._generation_start_time is not None:
            if (current_time - self._generation_start_time) >= cfg.bt_start_delay:
                return True

        # Default (both None): wait for all items.
        return False

    # -- reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Reset release state (re-release first batch)."""
        first_batch_size = min(
            self._config.items_per_batch, len(self._all_items),
        )
        self._released_count = first_batch_size
        self._last_batch_time = None
        self._generation_start_time = None


# ---------------------------------------------------------------------------
# Spatial-trigger generation
# ---------------------------------------------------------------------------


@dataclass
class SpatialTriggerRegion:
    """Rectangular region in the X/Y plane. Any bound = None is unbounded."""
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None

    def contains(self, x: float, y: float) -> bool:
        """An item is *inside* iff it satisfies every specified bound."""
        if self.min_x is not None and x < self.min_x:
            return False
        if self.max_x is not None and x > self.max_x:
            return False
        if self.min_y is not None and y < self.min_y:
            return False
        if self.max_y is not None and y > self.max_y:
            return False
        return True


@dataclass
class SpatialTriggerConfig:
    """Two-phase spatial-condition incremental spawn config.

    Phase 1 — Initial spawn:
        Release ``initial_count`` items at task start using the side's
        primary ``*_generation_strategy`` (typically the row layout used
        today).  The BT-start gate opens once the initial phase completes.

    Phase 2 — Replenishment (predicate-driven):
        Thereafter, release one batch (size ``items_per_batch``) at a time,
        gated by the spatial predicate evaluated over live world-frame XY
        of currently-spawned items.

        - default: fire iff *no* item is OUTSIDE ``region`` (every item
          satisfies the bounds, or there are no items at all).
        - ``invert=True``: fire iff *no* item is INSIDE ``region``.

        If ``replenishment_generation_strategy`` is provided, replenishment
        items are sourced from it (typically a single-point ``ItemGenerator``
        at the spawn end of the belt with x-jiggle).  Otherwise the primary
        strategy continues to supply positions for replenishment items.

    Cap & gating:
        Total items released is capped by ``count_range`` (the existing
        ``pick_count`` / ``target_count`` field on ``TaskSpec``, which the
        CLI ``--pick-count`` / ``--target-count`` already overrides).  When
        ``conveyor_speed`` (passed in by the caller) is 0.0 — or absent in
        mock execution — the scheduler stops releasing further items so
        ``more_*_expected`` can flip to False promptly.
    """
    region: SpatialTriggerRegion
    initial_count: int = 1
    replenishment_generation_strategy: Optional[Any] = None
    invert: bool = False
    items_per_batch: int = 1
    # Cooldown so we don't spawn every tick after the predicate holds (gives
    # the just-spawned item a moment to enter the trigger-suppressing region).
    min_spawn_interval: float = 0.0


class SpatialTriggeredItemScheduler:
    """Scheduler that releases items based on a spatial predicate.

    Pre-generates two queues of ``ItemSpec``s at construction time:
        * Initial queue — ``initial_count`` items from the primary generator.
        * Replenishment queue — the remaining (count - initial_count) items
          from ``replenishment_generation_strategy`` if provided, else from
          the primary generator.

    ``get_initial_batch()`` releases the entire initial queue in one call.
    ``tick(current_time, current_xy, conveyor_speed)`` evaluates the
    predicate and releases at most ``items_per_batch`` replenishment items
    when it fires (and conveyor is moving and cooldown elapsed).
    """

    def __init__(
        self,
        primary_generator: "ItemGenerator",
        config: SpatialTriggerConfig,
        count_range: Union[int, tuple, None] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config

        # Resolve total count (CLI --pick-count / --target-count flows in here).
        cap = primary_generator.position_generator.capacity
        if config.replenishment_generation_strategy is not None:
            # Replenishment generator may have its own capacity; the cap for
            # ``count_range=None`` is the sum.  In practice, count_range is
            # almost always set by the task author, so this fallback is
            # unlikely to trigger.
            repl_cap = config.replenishment_generation_strategy.position_generator.capacity
            if cap is not None and repl_cap is not None:
                cap = cap + repl_cap
            else:
                cap = None
        total = resolve_count(count_range, capacity=cap, seed=seed)
        if total is None:
            total = config.initial_count
        total = max(total, 0)

        initial_n = min(config.initial_count, total)
        replenish_n = max(0, total - initial_n)

        # Generate initial queue from primary.
        initial_items = primary_generator.generate(
            count_range=initial_n, seed=seed,
        ) if initial_n > 0 else []

        # Generate replenishment queue from replenishment generator (if any),
        # else from the primary generator.  Use a per-queue seed offset so
        # jitter doesn't mirror the initial layout.
        if replenish_n > 0:
            repl_gen = (
                config.replenishment_generation_strategy
                if config.replenishment_generation_strategy is not None
                else primary_generator
            )
            repl_seed = (seed + 1) if seed is not None else None
            replenish_items = repl_gen.generate(
                count_range=replenish_n, seed=repl_seed,
            )
        else:
            replenish_items = []

        self._initial_items: List[ItemSpec] = list(initial_items)
        self._replenish_items: List[ItemSpec] = list(replenish_items)
        self._total_count: int = len(self._initial_items) + len(self._replenish_items)
        self._released_count: int = 0
        self._initial_released: bool = False
        self._last_batch_time: Optional[float] = None

    # -- properties ----------------------------------------------------------

    @property
    def config(self) -> SpatialTriggerConfig:
        return self._config

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def released_count(self) -> int:
        return self._released_count

    @property
    def pending_count(self) -> int:
        return max(0, self._total_count - self._released_count)

    @property
    def all_released(self) -> bool:
        return self._released_count >= self._total_count

    # -- batch access --------------------------------------------------------

    def get_initial_batch(self) -> List[ItemSpec]:
        """Release the entire initial queue in one call."""
        if self._initial_released:
            return []
        self._initial_released = True
        self._released_count += len(self._initial_items)
        return list(self._initial_items)

    def tick(
        self,
        current_time: float,
        current_xy: Optional[List[tuple]] = None,
        conveyor_speed: Optional[float] = None,
    ) -> List[ItemSpec]:
        """Advance the scheduler and return any newly-released replenishment items.

        Args:
            current_time: Current simulation time in seconds.
            current_xy: List of (x, y) for currently-spawned items (live world
                pose). Empty list / None means "no items" (predicate fires by
                default semantics).
            conveyor_speed: Belt speed; 0.0 (or None) suppresses replenishment
                so the BT can complete on items in flight rather than idle.

        Returns:
            List of newly-released ``ItemSpec`` instances (may be empty).
        """
        if self.all_released:
            return []
        if not self._initial_released:
            return []
        # Suppress replenishment when belt is paused / mock-mode.
        if conveyor_speed is None or float(conveyor_speed) == 0.0:
            return []
        # Cooldown.
        if self._last_batch_time is not None and self._config.min_spawn_interval > 0:
            if (current_time - self._last_batch_time) < self._config.min_spawn_interval:
                return []
        # Predicate.
        if not self._trigger_fires(current_xy or []):
            return []
        # Release next batch from replenishment queue.
        repl_offset = self._released_count - len(self._initial_items)
        start = max(0, repl_offset)
        end = min(start + self._config.items_per_batch, len(self._replenish_items))
        if end <= start:
            return []
        new_items = self._replenish_items[start:end]
        self._released_count += len(new_items)
        self._last_batch_time = current_time
        if new_items:
            logger.debug(
                "SpatialTriggeredItemScheduler: released %d items (%d/%d) at t=%.3f",
                len(new_items), self._released_count, self._total_count,
                current_time,
            )
        return new_items

    def _trigger_fires(self, current_xy: List[tuple]) -> bool:
        """Evaluate the spatial predicate over current item XY positions."""
        if self._config.invert:
            # Fire iff no item is INSIDE the region.
            return not any(
                self._config.region.contains(float(x), float(y))
                for x, y in current_xy
            )
        # Fire iff no item is OUTSIDE the region (i.e., all are inside).
        return all(
            self._config.region.contains(float(x), float(y))
            for x, y in current_xy
        )

    # -- BT start gate -------------------------------------------------------

    def bt_should_start(self, current_time: float) -> bool:
        """The BT-start gate opens once the initial batch has been released."""
        return self._initial_released

    # -- reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Reset release state (re-release initial batch on next get_initial_batch)."""
        self._released_count = 0
        self._initial_released = False
        self._last_batch_time = None


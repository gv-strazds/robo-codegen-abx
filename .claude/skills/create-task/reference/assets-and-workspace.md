# Available Assets and Workspace Constants

## Primitive Assets

Built-in shapes created via `add_asset()`. Support `color` and `scale` parameters.

| asset_type   | Shape            | Notes                              |
|-------------|------------------|------------------------------------|
| `"cube"`     | DynamicCuboid    | Most common pick object            |
| `"ball"`     | DynamicSphere    | Rolls — harder for robot to place. For box packing, use reduced scale (0.035m) to prevent bouncing out; add `z_tol: 0.03` to box spec for settling tolerance. |
| `"cylinder"` | DynamicCylinder  | Standard cylinder                  |
| `"disc"`     | DynamicCylinder  | Flat disc (common target)          |
| `"fixed_disc"` | FixedCylinder  | Static disc target. Use when the disc must not move — e.g. when it sits on the conveyor surface and the placed item is tall and narrow (where dynamic-on-kinematic contact-chain perturbations would otherwise transmit jiggle to the placed item). Same `scale_xy = diameter/2` math as `disc` (FixedCylinder also defaults to radius=1.0). |
| `"cone"`     | DynamicCone      | Tapered shape                      |
| `"capsule"`  | DynamicCapsule   | Rounded cylinder                   |
| `"rect"`     | FixedCuboid      | Fixed thin rectangle (target)      |
| `"marker"`   | VisualCuboid     | Invisible placement target         |

## USD Assets

Complex meshes loaded from USD files. Do NOT support `color` parameter (use asset defaults).

**Important**: USD assets have their tall axis along local Y, not Z. Spawn with `-90 deg X` orientation to make them upright in the world.

| asset_type         | Description          | Approx Size (m)    |
|-------------------|----------------------|---------------------|
| `"soup_can"`       | YCB tomato soup can  | 0.068 x 0.102 x 0.068 |
| `"cracker_box"`    | YCB cracker box      | 0.164 x 0.213 x 0.072 |
| `"sugar_box"`      | YCB sugar box        | similar to cracker  |
| `"mustard_bottle"` | YCB mustard bottle   | small bottle        |
| `"madara_bottle"`  | Custom bottle V3     | 0.035 x 0.035 x 0.135 |
| `"madara_pad"`     | Custom carrier pad   | flat pad for bottles|

**Mugs**: `"mug_black"`, `"mug_black_green"`, `"mug_yellow"`, `"mug_blue"`

**Factory parts**: `"factory_bolt_m16"`, `"gear_large"`, `"gear_medium"`, `"gear_small"`, `"gear_base"`, `"factory_hole_8mm"`, `"factory_peg_8mm"`, `"nut_m16_yellow"`, `"nut_m16_green"`

**Bins/containers**: `"KLT_Bin"`, `"sorting_bin_blue"`, `"sorting_bin_black"`, `"sorting_beaker_red"`, `"sorting_bowl_yellow"`

## Robot-Relative Directions

The robot faces the dropzone/conveyor area. From the robot's perspective:

| Direction          | World Axis | Notes                                    |
|-------------------|------------|------------------------------------------|
| **Right**          | +X         | Increasing X                             |
| **Left**           | -X         | Decreasing X                             |
| **Closer to robot**| -Y         | Decreasing Y (toward robot base)         |
| **Away from robot**| +Y         | Increasing Y (away from robot base)      |
| **Up**             | +Z         | Increasing Z                             |

When interpreting user requests like "to the right of the source pile", map to world coordinates:
- "to the right" → +X (greater X than the source region's +X edge)
- "closer to the robot" → -Y (lower Y than the source region's -Y edge)
- "behind" / "further from robot" → +Y (greater Y)
- "in front of" / "between robot and source" → -Y (lower Y)

**Important**: The dropzone surface (Z=0) has limited extent. Objects placed too far in -X (toward the cart/tabletop area at X ≈ -0.8) will fall off the dropzone. Prefer +X for "to the right" placements on the dropzone.

## Workspace Coordinates

All coordinates are in world frame (meters, unless stage units differ).

These constants live in `env_config_values.py` (Isaac-Sim-free) and are also re-exported from `table_setup` for back-compat. Either import works in Isaac Sim tasks; prefer `env_config_values` from `TaskSpec` / tests / mock paths.

### KLT Pick Bin (on cart)
```python
BIN_X_COORD = -0.62
BIN_Y_COORD = 0.6
BIN_Z_COORD = 0.2      # ITEM_SPAWN_REFERENCE_Z + 0.1
BIN_SIZE = [0.270, 0.393, 0.072]  # inner cavity at BIN_SCALE=[1.5, 1.5, 0.5]
ITEM_SPAWN_REFERENCE_Z = 0.1
```
Note: `BIN_SIZE` is computed from `KLT_BIN_INNER_UNSCALED * BIN_SCALE` and represents the inner cavity dimensions, not the outer shell.

### Drop Zone (conveyor/floor area)
```python
DROPZONE_X = 0.25
DROPZONE_Y = 0.38
DROPZONE_Z = 0.0
DROPZONE_CENTER_POINT = [0.04, 0.69, 0.0]
```

### Cart Surface
```python
CART_SURFACE_CENTER = np.array([-0.79373, 0.3584, 0.0573])  # importable from env_config_values (or table_setup re-export)
CART_SURFACE_SIZE = np.array([0.7, 1.09])  # X, Y dims of the cart collision cuboid
```

### Tabletop
```python
TABLETOP_CENTER_POINT = [-0.81, 0.33, 0.1]
```

### Pick Region Utilities

`env_config_values` (re-exported from `table_setup` for back-compat) provides `Region2D` (a namedtuple with `min_x`, `max_x`, `min_y`, `max_y`) and `compute_region_2d()` for creating axis-aligned 2D bounding boxes:

```python
from table_setup import Region2D, compute_region_2d, BIN_INNER_REGION, CART_SURFACE_REGION

# Pre-computed regions:
BIN_INNER_REGION    # Region2D for the KLT bin inner cavity
CART_SURFACE_REGION # Region2D for the cart surface

# Custom region:
my_region = compute_region_2d(center_x, center_y, [width_x, depth_y])
```

## Common Z Calculations

```python
# Primitives in the bin:
pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.02

# USD assets in the bin (e.g., bottles):
pick_z = ITEM_SPAWN_REFERENCE_Z + 0.025 + <asset_half_height>

# Targets on the dropzone floor:
center_grid_z = DROPZONE_Z + 0.001 + <target_half_height>

# Markers inside a box on the cart:
marker_z = cart_z + box_base_thickness + 0.001
```

## Scene Setup

```python
from table_setup import setup_two_tables

# Standard setup (includes bin and standard objects):
self._setup_two_tables(scene, self._assets_root_path)

# Custom setup (no bin, no standard objects):
self._setup_two_tables(scene, self._assets_root_path, standard_objs=False, add_bin=False)
```

## Programmatic Box Creation: spawn_open_box

For tasks that place items into boxes, use `spawn_open_box()` from `table_setup` to create open-top boxes composed of FixedCuboids (a base plate + 4 walls):

```python
from table_setup import spawn_open_box

spawn_open_box(
    scene,                          # Isaac Sim scene to add prims to
    name="cart_box_1",              # Name prefix for generated prims
    center=np.array([x, y, z]),     # Center of the wall region (at mid wall height)
    inner_size=np.array([0.15, 0.22]),  # [width_x, depth_y] inner dimensions
    wall_height=0.12,               # Height of the four walls
    wall_thickness=0.01,            # Thickness of each wall
    base_thickness=0.01,            # Thickness of the base plate
    color=np.array([0.55, 0.43, 0.33]),  # RGB color for all box parts
    base_center_z=None,             # Optional: Z of base plate center
                                    # Default: center[2] - wall_height/2 + base_thickness/2
)
```

The `center` parameter is the center of the wall region at mid wall height — **not** the center of the box floor. For box-packing tasks, you also need to compute `box_floor_z` separately for marker placement:
```python
box_floor_z = cart_z + box_base_thickness + 0.001
```

Typical usage in a workspace setup function:
```python
def _workspace_setup(scene, assets_root):
    setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
    for bspec in box_specs:
        spawn_open_box(
            scene, name=bspec["name"], center=bspec["center"],
            inner_size=box_inner_size, wall_height=box_height,
            wall_thickness=box_wall, base_thickness=box_base_thickness,
            color=bspec["color"],
        )
```

See `TableTaskSoupCanPacking` and `TableTaskColorShapes` for complete examples.

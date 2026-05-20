"""Pure-python workspace geometry constants and helpers.

This module is intentionally free of any Isaac Sim / USD / PhysX dependency so it
can be imported from mock tasks, tests, and `TaskSpec` construction without
starting SimulationApp or relying on `extsMock` shadowing. Scene-construction
functions that do touch Isaac Sim live in `table_setup.py`.
"""

from collections import namedtuple

import numpy as np


GROUND_PLANE_Z_OFFSET = -0.5  # ground plane Z in world coords (baked into ur10_bin_filling.usd)

ASSERT_ORIGINAL_POSITIONS = False   # this can be True only if DELTA_ROBOT, DELTA_CART, and DELTA_DROPZONE are all zeros
DELTA_ROBOT = np.array([-0.35, 0.0, 0.0])  # for adjusting the position of the robot relative to everything else
DELTA_CART = np.array(([-0.10, 0.375, 0]))  # for moving the cart relative to where it was originally
DELTA_DROPZONE = np.array([0.04, 0.0, 0.0])  # for moving the dropzone/conveyor relative to where it was originally

# =============================================================================
# Section A: Root Positions
# =============================================================================
# Three root positions define the workspace. Everything else is derived from these.

# --- Root position #1: Robot base (world origin) ---
UR_COORDS = np.array([0.0, 0.0, 0.0])
UR_Z_COORD_0 = -GROUND_PLANE_Z_OFFSET  # height of robot base above ground plane (0.5m)

# --- Table dimensions (physical) ---
TABLE_THICKNESS = 0.1
# ITEM_SPAWN_REFERENCE_Z is a *nominal* datum plane used as the reference Z
# for spawning items by the usual pattern
#   pick_z = ITEM_SPAWN_REFERENCE_Z + item_half_height + margin
# so the item drops by gravity onto the actual resting surface.  It is NOT
# the physical surface where objects end up resting — the bin settles on
# the cart at CART_SURFACE_CENTER.z (~4 cm below this datum), and items
# inside the bin rest at BIN_INNER_FLOOR_Z (see below).  Use those
# constants, not ITEM_SPAWN_REFERENCE_Z, for reachability / region checks.
ITEM_SPAWN_REFERENCE_Z_OFFSET = 0.1  # spawn-reference datum Z, relative to robot base
ITEM_SPAWN_REFERENCE_Z = UR_COORDS[2] + ITEM_SPAWN_REFERENCE_Z_OFFSET
TABLETOP_HEIGHT = ITEM_SPAWN_REFERENCE_Z - GROUND_PLANE_Z_OFFSET  # datum height above ground
TABLE_LENGTH = 1.2
TABLE_WIDTH = 0.7
TABLE_SIZE = np.array([TABLE_WIDTH, TABLE_LENGTH, TABLE_THICKNESS])

# --- Root position #2: Table surface center ---
TABLETOP_CENTER_POINT = np.array([-0.81, 0.33, ITEM_SPAWN_REFERENCE_Z]) + DELTA_CART - DELTA_ROBOT
TABLE_COORDS = TABLETOP_CENTER_POINT - [0, 0, TABLE_THICKNESS / 2]

# --- Root position #3: Drop zone / conveyor region center ---
DROPZONE_CENTER_POINT = np.array([0.04, 0.69, 0.0]) - DELTA_ROBOT + DELTA_DROPZONE

CAMERA_VIEW1_POS = np.array([0.5, 2.0-0.1, 1.5-0.7]) - DELTA_ROBOT
CAMERA_VIEW1_LOOKAT = TABLETOP_CENTER_POINT + [0.3-0.4, 0-0.7, 0-0.1]

CAMERA_VEC = CAMERA_VIEW1_LOOKAT-CAMERA_VIEW1_POS
CAMERA_EXTRA_DISTANCE = CAMERA_VEC*0.25
CAMERA_VIEW1_POS = CAMERA_VIEW1_POS - CAMERA_EXTRA_DISTANCE
# Wider preset: pull back further along the same view vector and tilt the
# look-at point down so video / wide snapshots show more of the scene.
CAMERA_VIEW1_WIDE_POS = CAMERA_VIEW1_POS - CAMERA_VEC * 0.35
CAMERA_VIEW1_WIDE_LOOKAT = CAMERA_VIEW1_LOOKAT + np.array([0.0, 0.0, -0.3])
# =============================================================================
# Section B: Table-derived positions
# =============================================================================

# KLT picking bin center — offset [+X toward robot, +Y along table, +Z above surface].
# Note: BIN_Z_COORD (0.2) is a reference height for items AT the bin surface level,
# not the bin mesh placement Z (which is ITEM_SPAWN_REFERENCE_Z + 0.05).
BIN_COORDS = TABLETOP_CENTER_POINT + [0.19, 0.20, 0.1]
BIN_X_COORD = BIN_COORDS[0]
BIN_Y_COORD = BIN_COORDS[1]
BIN_Z_COORD = BIN_COORDS[2]  # spawn reference Z for items inside the bin
_BIN_PLACEMENT_Z_OFFSET = 0.05  # bin mesh Z above the ITEM_SPAWN_REFERENCE_Z datum

# Cart surface center — relative to table center
_CART_SURFACE_OFFSET = np.array([0.01627, 0.0284, -0.0427])
CART_SURFACE_CENTER = TABLETOP_CENTER_POINT + _CART_SURFACE_OFFSET
CART_SURFACE_SIZE = np.array([0.7, 1.09])  # X, Y dims of the cart collision cuboid

# Inner floor of the KLT bin in world Z — the surface items rest on when
# settled inside the bin.  The bin spawns above the cart and drops by
# gravity to rest on CART_SURFACE_CENTER.z; the bin's floor thickness is
# negligible compared to item rest_heights, so we use the cart surface Z
# directly.  Items resting in the bin have centre Z ≈ BIN_INNER_FLOOR_Z +
# item_rest_height, so this is also a safe lower bound for "object is
# physically in/on the bin" reachability checks.
BIN_INNER_FLOOR_Z = float(CART_SURFACE_CENTER[2])

# Cart USD model position — relative to cart surface center (model origin is at base/floor)
_CART_PRIM_OFFSET = np.array([0.01619, -0.087, -0.58042])
_CART_POSITION = CART_SURFACE_CENTER + _CART_PRIM_OFFSET

# =============================================================================
# Section C: Dropzone-derived positions
# =============================================================================

DROPZONE_Z = DROPZONE_CENTER_POINT[2]  # Z surface of drop zone (imported by ~38 tasks)

# Corner coordinates of the drop zone region (imported by 9 tasks)
_DROPZONE_HALF_WIDTH = 0.21  # half-extent in X
_DROPZONE_HALF_DEPTH = 0.31  # half-extent in Y
DROPZONE_X = DROPZONE_CENTER_POINT[0] + _DROPZONE_HALF_WIDTH  # = 0.25
DROPZONE_Y = DROPZONE_CENTER_POINT[1] - _DROPZONE_HALF_DEPTH  # = 0.38

# Conveyor belt model position — relative to drop zone center
_CONVEYOR_OFFSET = np.array([0.0927, -0.5369, -0.54184])
_CONVEYOR_POSITION = DROPZONE_CENTER_POINT + _CONVEYOR_OFFSET

# Invisible collision surface on top of conveyor — relative to conveyor model.
# The Z offset is chosen so the surface *top* (center + thickness/2) sits at
# DROPZONE_Z = 0.  Items spawned with their bottom at DROPZONE_Z therefore
# rest directly on the surface with no free-fall gap, and the thick surface
# covers small roller protrusions on the underlying conveyor USD mesh
# (observed to cause brief hesitation as items slid over them when the
# surface was only 1 mm thick with its top ~2.6 mm below DROPZONE_Z).
CONVEYOR_SURFACE_THICKNESS = 0.01
_CONVEYOR_SURFACE_OFFSET = np.array([
    -0.00757, 0.79067,
    -_CONVEYOR_OFFSET[2] - CONVEYOR_SURFACE_THICKNESS / 2,
])
CONVEYOR_SURFACE_CENTER = _CONVEYOR_POSITION + _CONVEYOR_SURFACE_OFFSET
CONVEYOR_SURFACE_TOP_Z = CONVEYOR_SURFACE_CENTER[2] + CONVEYOR_SURFACE_THICKNESS / 2

# Conveyor surface Y half-extent — scale Y is 1.6 (see setup_two_tables), so
# half-extent is 0.8m around CONVEYOR_SURFACE_CENTER[1]. Objects traveling
# along the belt in -Y fall off at CONVEYOR_END_Y.
CONVEYOR_SURFACE_Y_HALF_EXTENT = 0.8
CONVEYOR_END_Y = CONVEYOR_SURFACE_CENTER[1] - CONVEYOR_SURFACE_Y_HALF_EXTENT

# Conveyor surface X half-extent — scale X is 0.7 (see setup_two_tables), so
# half-extent is 0.35m around CONVEYOR_SURFACE_CENTER[0]. Used by the mock
# conveyor's on-belt geometric test.
CONVEYOR_SURFACE_X_HALF_EXTENT = 0.35

# =============================================================================
# Section D: KLT bin geometry & regions
# =============================================================================

# KLT bin inner cavity dimensions at unit scale [1,1,1], measured from mesh vertices.
# The KLT bin USD model (/Isaac/Props/KLT_Bin/small_KLT.usd) has:
#   outer (unscaled): ~0.198 x 0.297 x 0.146 m
#   inner (unscaled): ~0.180 x 0.262 x 0.144 m
#   wall thickness X: ~0.009m, Y: ~0.017m, floor: ~0.002m
KLT_BIN_INNER_UNSCALED = np.array([0.180, 0.262, 0.144])

BIN_SCALE = [1.5, 1.5, 0.5]
BIN_SIZE = (KLT_BIN_INNER_UNSCALED * BIN_SCALE).tolist()  # [0.270, 0.393, 0.072]

Region2D = namedtuple("Region2D", ["min_x", "max_x", "min_y", "max_y"])


def compute_region_2d(center_x, center_y, size_xy):
    """Compute a Region2D (axis-aligned bounding box) centered at (center_x, center_y).

    Args:
        center_x: X center coordinate.
        center_y: Y center coordinate.
        size_xy: Sequence with at least 2 elements [width_x, depth_y].
    """
    return Region2D(
        center_x - size_xy[0] / 2,
        center_x + size_xy[0] / 2,
        center_y - size_xy[1] / 2,
        center_y + size_xy[1] / 2,
    )


def compute_klt_bin_inner_size(scale):
    """Compute inner cavity dimensions for a KLT bin at the given scale.

    Args:
        scale: [sx, sy, sz] scale factors applied to the KLT bin USD asset.

    Returns:
        np.ndarray [inner_x, inner_y, inner_z] in meters.
    """
    return KLT_BIN_INNER_UNSCALED * np.asarray(scale)


BIN_INNER_REGION = compute_region_2d(BIN_X_COORD, BIN_Y_COORD, BIN_SIZE)
CART_SURFACE_REGION = compute_region_2d(CART_SURFACE_CENTER[0], CART_SURFACE_CENTER[1], CART_SURFACE_SIZE)

# =============================================================================
# Value-preservation assertions — catch accidental drift from refactoring
# =============================================================================
if ASSERT_ORIGINAL_POSITIONS:
    assert np.allclose(DROPZONE_X, 0.25), f"DROPZONE_X drift: {DROPZONE_X}"
    assert np.allclose(DROPZONE_Y, 0.38), f"DROPZONE_Y drift: {DROPZONE_Y}"
    assert np.allclose(CART_SURFACE_CENTER, [-0.79373, 0.3584, 0.0573], atol=1e-4), \
        f"CART_SURFACE_CENTER drift: {CART_SURFACE_CENTER}"
    assert np.allclose(_CONVEYOR_POSITION, [0.1327, 0.1531, -0.54184], atol=1e-4), \
        f"_CONVEYOR_POSITION drift: {_CONVEYOR_POSITION}"
    assert np.allclose(CONVEYOR_SURFACE_CENTER, [0.12513, 0.94377, -0.005], atol=1e-4), \
        f"CONVEYOR_SURFACE_CENTER drift: {CONVEYOR_SURFACE_CENTER}"
    assert np.isclose(CONVEYOR_SURFACE_TOP_Z, 0.0, atol=1e-6), \
        f"CONVEYOR_SURFACE_TOP_Z drift: {CONVEYOR_SURFACE_TOP_Z}"
    assert np.allclose(_CART_POSITION, [-0.77754, 0.2714, -0.52312], atol=1e-4), \
        f"_CART_POSITION drift: {_CART_POSITION}"


def is_in_bin_region(x, y, z):
    """Check whether a point (x, y, z) falls within the KLT bin's pick region (with margins).

    Z lower bound is ``BIN_INNER_FLOOR_Z - 0.02`` — the bin's inner floor
    minus a 2 cm safety margin — so items actually resting inside the bin
    (centre Z ≈ floor + item rest_height) are accepted while items that
    have fallen below the bin or through the cart are rejected.
    """
    return (
        z >= BIN_INNER_FLOOR_Z - 0.02
        and (BIN_INNER_REGION.min_y - 0.3 < y < BIN_INNER_REGION.max_y + 0.1)
        and (BIN_INNER_REGION.min_x - 0.05 < x < BIN_INNER_REGION.max_x + 0.2)
    )


cylinder_specs = [
    ([-0.05 + BIN_X_COORD, BIN_Y_COORD + 0.0, BIN_Z_COORD + 0.44], [0.0, 75.8, 0.0]),
    ([-0.01 + BIN_X_COORD, BIN_Y_COORD + 0.065, BIN_Z_COORD + 0.50], [0.0, 75.8, 0.0]),
    ([-0.05 + BIN_X_COORD, BIN_Y_COORD + 0.09, BIN_Z_COORD + 0.48], [0.0, 75.8, 0.0]),
    ([-0.02 + BIN_X_COORD, BIN_Y_COORD + 0.065, BIN_Z_COORD + 0.57], [0.0, 75.8, 39.0]),
]


# Default speed for a moving conveyor (surface velocity in m/s along -Y).
DEFAULT_CONVEYOR_SPEED = -0.015

# Minimum Z for a target to be considered reachable.  Targets that fall below
# this (e.g. after sliding off the conveyor edge) are filtered from the BT's
# target selection.  5 cm below the conveyor surface gives margin for physics
# settling while catching targets that have genuinely fallen off.
TARGET_MIN_REACHABLE_Z = CONVEYOR_SURFACE_TOP_Z - 0.05

# UR10 nominal reach is 1.30 m; use 1.25 m as the working-radius default
# to keep some margin off the singular boundary where RMPFlow loses
# precision.  CheckPickReachable in the cortex BT uses this when no
# per-task TaskSpec.pick_max_reachable_radius_xy override is supplied.
UR10_WORKING_RADIUS = 1.25


def make_z_reachability_check(min_z=None):
    """Create a target reachability predicate checking z > min_z.

    Args:
        min_z: Minimum Z threshold.  Defaults to TARGET_MIN_REACHABLE_Z.

    Returns:
        Callable(target_obj) -> bool
    """
    if min_z is None:
        min_z = TARGET_MIN_REACHABLE_Z

    def _check(target_obj):
        pos, _ = target_obj.get_world_pose()
        return float(pos[2]) > min_z

    return _check

import math
import random
import sys
import logging

from contextlib import contextmanager
from typing import Optional

import numpy as np

from asset_utils import add_asset

# Pure-python geometry constants and helpers live in env_config_values.py so they
# can be imported without Isaac Sim. Re-exported here for back-compat with the
# many task modules that still do `from table_setup import <constant>`.
from env_config_values import (  # noqa: F401
    GROUND_PLANE_Z_OFFSET, ASSERT_ORIGINAL_POSITIONS, DELTA_ROBOT, DELTA_CART, DELTA_DROPZONE,
    UR_COORDS, UR_Z_COORD_0,
    TABLE_THICKNESS, ITEM_SPAWN_REFERENCE_Z_OFFSET, ITEM_SPAWN_REFERENCE_Z, TABLETOP_HEIGHT,
    TABLE_LENGTH, TABLE_WIDTH, TABLE_SIZE, TABLETOP_CENTER_POINT, TABLE_COORDS,
    DROPZONE_CENTER_POINT,
    CAMERA_VIEW1_POS, CAMERA_VIEW1_LOOKAT, CAMERA_VEC, CAMERA_EXTRA_DISTANCE,
    BIN_COORDS, BIN_X_COORD, BIN_Y_COORD, BIN_Z_COORD, _BIN_PLACEMENT_Z_OFFSET,
    _CART_SURFACE_OFFSET, CART_SURFACE_CENTER, CART_SURFACE_SIZE,
    _CART_PRIM_OFFSET, _CART_POSITION,
    DROPZONE_Z, _DROPZONE_HALF_WIDTH, _DROPZONE_HALF_DEPTH, DROPZONE_X, DROPZONE_Y,
    _CONVEYOR_OFFSET, _CONVEYOR_POSITION,
    _CONVEYOR_SURFACE_OFFSET, CONVEYOR_SURFACE_CENTER, CONVEYOR_SURFACE_THICKNESS,
    KLT_BIN_INNER_UNSCALED, BIN_SCALE, BIN_SIZE,
    Region2D, compute_region_2d, compute_klt_bin_inner_size,
    BIN_INNER_REGION, CART_SURFACE_REGION,
    is_in_bin_region, cylinder_specs,
    DEFAULT_CONVEYOR_SPEED,
)

# from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die

from isaacsim.core.api.objects import (
    DynamicCuboid,
    DynamicCylinder,
    FixedCuboid,
    VisualCapsule,
    VisualCuboid,
    VisualSphere,
)
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.scenes.scene import Scene

from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils import (  # noqa E402
    extensions,
    prims,
    rotations,
    stage,
    viewports,
    # transformations
)

# from isaacsim.core.utils.prims import is_prim_path_valid

from isaacsim.core.utils.collisions import ray_cast
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot.manipulators.examples.universal_robots import UR10
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, UsdPhysics, UsdGeom  # noqa E402

logger = logging.getLogger(__name__)


# Ambient conveyor speed published by the task framework around the
# setup_workspace call so that setup_two_tables() automatically inherits
# TaskSpec.conveyor_speed without every task lambda having to forward it
# explicitly. An explicit conveyor_speed= argument to setup_two_tables still
# overrides this.
_AMBIENT_CONVEYOR_SPEED: Optional[float] = None


@contextmanager
def ambient_conveyor_speed(value: Optional[float]):
    """Temporarily publish ``value`` as the ambient conveyor speed.

    Used by ``UR10MultiPickPlaceTask.setup_workspace`` to make
    ``setup_two_tables`` inherit ``TaskSpec.conveyor_speed`` when a task's
    setup_workspace callable does not pass ``conveyor_speed`` explicitly.
    """
    global _AMBIENT_CONVEYOR_SPEED
    prev = _AMBIENT_CONVEYOR_SPEED
    _AMBIENT_CONVEYOR_SPEED = value
    try:
        yield
    finally:
        _AMBIENT_CONVEYOR_SPEED = prev


def random_spawn_transform(spawn_region: Region2D, z_baseline=BIN_Z_COORD):
    _SPAWN_MIN_Z = 0.25  # 1.0
    _SPAWN_MAX_Z = 0.55  # 1.5
    x = random.uniform(spawn_region.min_x + 0.2, spawn_region.max_x - 0.2)
    y = random.uniform(spawn_region.min_y + 0.2, spawn_region.max_y - 0.2)
    z = random.uniform(
        z_baseline + _SPAWN_MIN_Z, z_baseline + _SPAWN_MAX_Z
    )  # high enough to be out of the way
    position = np.array([x, y, z])
    # position = np.array([0.3, 0.3, 0.3]) / get_stage_units()
    # jj = random.random() * 0.02 - 0.01
    w = (random.random() - 0.5) * 1.5
    # norm = np.sqrt(jj**2 + w**2)
    # quat = math_util.Quaternion([w / norm, 0, jj / norm, 0]).vals
    # quat = math_util.Quaternion([1.0, 0, 0, 0]).vals
    quat = rotations.euler_angles_to_quat(np.array([0.0, math.pi / 2 + w, 0.0]))
    if False and random.random() > 0.5:
        print("<flip>")
        # flip the bottle so it's upside down
        quat = quat * math_util.Quaternion([0, 0, 1, 0]).vals
    else:
        print("<no flip>")

    return position, quat


def setup_two_tables(
    scene: Scene,
    assets_root_path=None,
    standard_objs=True,
    add_bin=True,
    bin_scale=None,
    conveyor_speed=None,
) -> None:
    if conveyor_speed is None:
        # Fall back to the ambient speed published by the task framework
        # (TaskSpec.conveyor_speed); 0.0 when neither is set.
        conveyor_speed = _AMBIENT_CONVEYOR_SPEED if _AMBIENT_CONVEYOR_SPEED is not None else 0.0
    if assets_root_path is None:
        assets_root_path = get_assets_root_path()
    # GroundPlane(prim_path="/World/groundPlane", size=3, color=np.array([0.1, 0.15, 0.25]))

    viewports.set_camera_view(
        eye=CAMERA_VIEW1_POS, target=CAMERA_VIEW1_LOOKAT
    )

    cart_sc = 0.012
    cart = prims.create_prim(
        "/World/Cart",
        "Xform",
        position=_CART_POSITION,
        scale=np.array([cart_sc, cart_sc, 0.007]),
        usd_path=assets_root_path
        + "/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Equipment/Carts/WeldedSteelCart_A/WeldedSteelCart_A02_01.usd",
        # usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Carts/WeldedSteelCart_A/WeldedSteelCart_A02_01.usd"
    )

    # invisible rectangle to provide a better (collision) top surface for the cart
    cart_surface = FixedCuboid(
        prim_path="/World/cart_surface",
        name="cart_surface",
        position=CART_SURFACE_CENTER,
        scale=np.array([CART_SURFACE_SIZE[0], CART_SURFACE_SIZE[1], 10 ** (-4)]),
        color=np.array([0.2, 0.3, 0.0]),
        visible=False,
    )

    rotate_90_around_z = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)
        )
    conveyor = prims.create_prim(
        "/World/Conveyor",
        "Xform",
        position=_CONVEYOR_POSITION,
        orientation=rotate_90_around_z,
        scale=np.array([0.8, 0.8, 0.7]),
        usd_path=assets_root_path + "/Isaac/Props/Conveyors/ConveyorBelt_A05.usd",
    )
    # For delete_prim
    # stage_utils.add_reference_to_stage("/test_path/cube.usd", "/World/Cube")
    rubber_bands = prims.get_prim_at_path("/World/Conveyor/Rubberbands", fabric=False)
    prims.set_prim_visibility(rubber_bands, False)

    conveyor_surface = DynamicCuboid(
        prim_path="/World/conveyor_surface",
        name="conveyor_surface",
        position=CONVEYOR_SURFACE_CENTER,
        scale=np.array([0.7, 1.6, CONVEYOR_SURFACE_THICKNESS]),
        color=np.array([0.2, 0.3, 0.0]),
        visible=False,
    )

    # from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema  # noqa E402
    from isaacsim.core.api.materials import PhysicsMaterial
    # create a rigid body physical material
    material = PhysicsMaterial(
        prim_path="/World/Physics_Materials/conveyor_surface",  # path to the material prim to create
        dynamic_friction=0.8,
        static_friction=1.0,
        restitution=0.0,
    )
    conveyor_surface.apply_physics_material(material)

    usd_prim = prims.get_prim_at_path("/World/conveyor_surface")
    rigid_body_api = UsdPhysics.RigidBodyAPI(usd_prim)
    logger.warning("Calling PhysxRigidBodyAPI.Apply(usd_prim for conveyor_surface)")
    physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(usd_prim)
    physx_rigid_body_api.CreateDisableGravityAttr(True)
    assert rigid_body_api is not None
    rigid_body_api.CreateKinematicEnabledAttr(True)
    #    # this does NOT work (drop_zone still falls due to gravity)
    #    rigid_body_api.disable_gravity = True

    #Apply surface velocity API to the object
    if conveyor_speed != 0.0: #-0.015
        PhysxSchema.PhysxSurfaceVelocityAPI.Apply(usd_prim)
        surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI(usd_prim)
        #Set the actual velocity vector
        velocity = Gf.Vec3f(0.0, conveyor_speed, 0.0)  # mm/s in -Y direction
        surface_velocity.CreateSurfaceVelocityAttr(velocity)

    if standard_objs:
        # add some objects on the table
        add_asset(
            scene,
            asset_type="cracker_box",
            obj_name="cracker_box",
            position=TABLETOP_CENTER_POINT
            + [
                -0.19,
                -0.08,
                0.15,
            ],  # np.array([-0.2-UR_X_COORD_0, -0.25-UR_Y_COORD_0, ITEM_SPAWN_REFERENCE_Z+0.15]),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
            ),
            assets_root_path=assets_root_path,
        )
        add_asset(
            scene,
            asset_type="sugar_box",
            obj_name="sugar_box",
            position=TABLETOP_CENTER_POINT
            + [
                -0.06,
                -0.08,
                0.01,
            ],  # np.array([-0.07-UR_X_COORD_0, -0.25-UR_Y_COORD_0, ITEM_SPAWN_REFERENCE_Z+0.1]),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)
            ),
            assets_root_path=assets_root_path,
        )
        add_asset(
            scene,
            asset_type="soup_can",
            obj_name="soup_can",
            position=TABLETOP_CENTER_POINT
            + [
                0.11,
                -0.08,
                0.10,
            ],  # np.array([0.1-UR_X_COORD_0, -0.25-UR_Y_COORD_0, ITEM_SPAWN_REFERENCE_Z+0.10]),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
            ),
            assets_root_path=assets_root_path,
        )
        add_asset(
            scene,
            asset_type="mustard_bottle",
            obj_name="mustard_bottle",
            position=TABLETOP_CENTER_POINT
            + [
                -0.055,
                0.235,
                0.12,
            ],  # np.array([-0.065-UR_X_COORD_0, 0.065-UR_Y_COORD_0, ITEM_SPAWN_REFERENCE_Z+0.12]),
            orientation=rotations.gf_rotation_to_np_array(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
            ),
            assets_root_path=assets_root_path,
        )

    if add_bin:
        _bin_scale = bin_scale if bin_scale is not None else BIN_SCALE
        add_asset(
            scene,
            asset_type="KLT_Bin",
            obj_name="KLT_Bin",
            position=np.array([BIN_X_COORD, BIN_Y_COORD, ITEM_SPAWN_REFERENCE_Z + _BIN_PLACEMENT_Z_OFFSET]),
            orientation=None,  # rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            scale=np.array(
                _bin_scale
            ),  # a shallow bin, to make it easier to pick up the bottles
            assets_root_path=assets_root_path,
        )


def spawn_open_box(scene, name, center, inner_size, wall_height,
                   wall_thickness, base_thickness, color,
                   base_center_z=None):
    """Create an open-top box (base plate + 4 walls) composed of FixedCuboids.

    Args:
        scene: Isaac Sim scene to add prims to.
        name: Name prefix for generated prims (e.g. "cart_box_1").
        center: np.array([x, y, z]) center of the wall region (at mid wall height).
        inner_size: np.array([x, y]) inner dimensions of the box.
        wall_height: Height of the four walls.
        wall_thickness: Thickness of each wall.
        base_thickness: Thickness of the base plate.
        color: np.array([r, g, b]) color for all box parts.
        base_center_z: Z-coordinate of the base plate center. If None, derived
            as center[2] - wall_height / 2 + base_thickness / 2.
    """
    if base_center_z is None:
        base_center_z = center[2] - wall_height / 2 + base_thickness / 2

    # Base plate
    base_pos = np.array([center[0], center[1], base_center_z])
    base_scale = np.array([
        inner_size[0] + 2 * wall_thickness,
        inner_size[1] + 2 * wall_thickness,
        base_thickness,
    ])
    scene.add(FixedCuboid(
        prim_path=f"/World/{name}_base",
        name=f"{name}_base",
        position=base_pos,
        scale=base_scale,
        color=color,
    ))

    # 4 walls
    half_x = inner_size[0] / 2
    half_y = inner_size[1] / 2
    wall_defs = [
        (np.array([half_x + wall_thickness / 2, 0.0, 0.0]),
         np.array([wall_thickness, inner_size[1], wall_height])),
        (np.array([-half_x - wall_thickness / 2, 0.0, 0.0]),
         np.array([wall_thickness, inner_size[1], wall_height])),
        (np.array([0.0, half_y + wall_thickness / 2, 0.0]),
         np.array([inner_size[0] + 2 * wall_thickness, wall_thickness, wall_height])),
        (np.array([0.0, -half_y - wall_thickness / 2, 0.0]),
         np.array([inner_size[0] + 2 * wall_thickness, wall_thickness, wall_height])),
    ]
    for idx, (offset, scale) in enumerate(wall_defs):
        wall_pos = center + offset
        scene.add(FixedCuboid(
            prim_path=f"/World/{name}_wall_{idx}",
            name=f"{name}_wall_{idx}",
            position=wall_pos,
            scale=scale,
            color=color,
        ))



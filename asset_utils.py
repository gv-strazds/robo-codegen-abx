"""IsaacSim-dependent asset utilities.

Scene creation, live-prim AABB computation, and semantic-label helpers live
here. Pure-Python / numpy metadata and geometry math lives in
``asset_data_utils`` and is re-exported from this module for backward
compatibility.
"""

import numpy as np
import os
from typing import Optional
from colors import *

import logging
logger = logging.getLogger(__name__)

# Re-export pure-Python data/geometry symbols for backward compatibility.
from asset_data_utils import (  # noqa: F401
    PRIM_TYPES,
    AssetMetaData,
    AssetSymmetry,
    ITEMS_MAP,
    PrimGeometry,
    get_asset_symmetry,
    scale_aabb,
    lookup_prim_geometry,
    simready_assets,
    _quat_to_rotation_matrix,
    _load_precomputed_geometry,
)

# import omni.log
from isaacsim.core.api.objects import (
    DynamicCuboid,
    FixedCuboid,
    VisualCuboid,
    DynamicCylinder,
    FixedCylinder,
    DynamicSphere,
    DynamicCapsule,
    DynamicCone,
)
# from isaacsim.core.api.objects import VisualCapsule, VisualSphere
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb

from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils import (  # noqa E402
    extensions,
    prims,
    rotations,
    stage,
    viewports,
    # transformations
)
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, UsdGeom  # noqa E402
from pxr import UsdPhysics

# NOTE: isaacsim.core.utils
# bounds utils for bounding boxes
# randomization
# Semantic utils for adding, removing, querying labels
#  - add_labels(prim, labels: list[str], instance_name: str='class', overwrite: bool=True)
#  - count_labels_in_scene() -> dict[str, int] # returns dict of (label:count)
#  - get_labels(prim) -> dict[str, list[str]] :returns map of instance names to lists of labels
# relationships ( set_target() )
# transforms:
#   get_relative_transform( source_prim: pxr.Usd.Prim, target_prim: pxr.Usd.Prim)
#   simlarly for ..._with_normalized_rotation()
#   get_translation


PRIMS_MAP = {
    "cube": DynamicCuboid,
    "disc": DynamicCylinder,
    "ball": DynamicSphere,
    "cylinder": DynamicCylinder,
    "capsule": DynamicCapsule,
    "cone": DynamicCone,
    "rect": FixedCuboid,
    "fixed_disc": FixedCylinder,
    "marker": VisualCuboid,
}


def needs_aabb_scale_correction(prim) -> bool:
    """Check if this prim type suffers from the cuboid double-scale AABB bug."""
    return is_of_type(prim, ["cube", "rect", "marker"])


def _get_prim_local_scale(prim) -> np.ndarray:
    """Retrieve the local scale of a prim using Isaac Sim or USD APIs."""
    # Try Isaac Sim API first
    if hasattr(prim, "get_local_scale"):
        return np.array(prim.get_local_scale())

    # Fallback to USD API
    p = getattr(prim, "prim", prim)
    if p:
        xformable = UsdGeom.Xformable(p)
        # GetLocalTransformation returns a Gf.Matrix4d
        local_xf = xformable.GetLocalTransformation()
        # Decompose matrix to get scale. Matrix4d doesn't have ExtractScale.
        # We can use Factor() or just extract from diagonal if no rotation/shear,
        # but Factor is safer.
        # However, for simple cases in Isaac Sim, we can often just use:
        # scale = [local_xf.GetRow(i).GetLength() for i in range(3)]
        # but that includes rotation.
        # Actually, Gf.Matrix4d has no easy ExtractScale.
        # Let's try to get it from the scale op if it exists.
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                val = op.Get()
                if val:
                    return np.array([val[0], val[1], val[2]])

    return np.array([1.0, 1.0, 1.0])


def compute_prim_geometry(prim, obj_scale=None, asset_type=None, bb_cache=None) -> PrimGeometry:
    """Compute intrinsic geometry from a loaded prim's bounding box.

    Call once after the prim is added to the scene. Uses AABB in its
    initial (world-axis-aligned) pose for reliable measurements.

    If AssetMetaData provides override values (grasp_height, rest_height,
    top_surface_height), those take precedence over computed values.

    Args:
        prim: The loaded prim object (must have prim_path and get_world_pose).
        obj_scale: Optional scale array for cuboid AABB correction. If None,
            the scale is queried from the prim directly.
        asset_type: Optional asset type string for ITEMS_MAP metadata lookup.
        bb_cache: Optional bounding box cache (created if not provided).

    Returns:
        PrimGeometry with cached intrinsic geometry values.
    """
    if bb_cache is None:
        bb_cache = create_bbox_cache()

    prim_path = prim.prim_path
    aabb = compute_aabb(bb_cache, prim_path=prim_path, include_children=True)

    # Detect and apply shape-specific AABB corrections
    needs_correction = needs_aabb_scale_correction(prim)
    if needs_correction:
        # Cuboid double-scale correction
        if obj_scale is None:
            obj_scale = _get_prim_local_scale(prim)

        if obj_scale is not None:
            inv_scale = np.array([1.0 / obj_scale[0], 1.0 / obj_scale[1], 1.0 / obj_scale[2]])
            aabb = scale_aabb(aabb, inv_scale)
    elif is_of_type(prim, "ball"):
        # Sphere rotation correction: USD BBoxCache inflates the AABB when a
        # sphere's extent box is rotated. Recompute from center + scale since
        # a sphere's true AABB is rotation-invariant.
        if obj_scale is None:
            obj_scale = _get_prim_local_scale(prim)
        if obj_scale is not None:
            center = (aabb[:3] + aabb[3:]) / 2
            half_ext = np.abs(np.asarray(obj_scale, dtype=float))
            aabb = np.array([
                center[0] - half_ext[0], center[1] - half_ext[1], center[2] - half_ext[2],
                center[0] + half_ext[0], center[1] + half_ext[1], center[2] + half_ext[2],
            ])

    # Get world pose for origin-relative calculations
    pos, _ = prim.get_world_pose()

    # Compute from AABB
    aabb_height = aabb[5] - aabb[2]
    rest_height = pos[2] - aabb[2]         # origin Z minus bottom Z
    top_surface_height = aabb[5] - pos[2]  # top Z minus origin Z
    grasp_height = aabb_height / 2.0       # default: center of object
    local_half_extents = np.array([
        (aabb[3] - aabb[0]) / 2.0,
        (aabb[4] - aabb[1]) / 2.0,
        (aabb[5] - aabb[2]) / 2.0,
    ])

    # Check ITEMS_MAP for overrides
    default_grasp_offset = np.zeros(3)
    if asset_type is not None and asset_type in ITEMS_MAP:
        meta = ITEMS_MAP[asset_type]
        if meta.grasp_height is not None:
            grasp_height = meta.grasp_height
        if meta.rest_height is not None:
            rest_height = meta.rest_height
        if meta.top_surface_height is not None:
            top_surface_height = meta.top_surface_height
        if meta.default_grasp_offset is not None:
            default_grasp_offset = np.asarray(meta.default_grasp_offset, dtype=float).copy()
            if obj_scale is not None:
                default_grasp_offset = default_grasp_offset * np.asarray(obj_scale, dtype=float)

    return PrimGeometry(
        grasp_height=grasp_height,
        rest_height=rest_height,
        top_surface_height=top_surface_height,
        local_half_extents=local_half_extents,
        needs_aabb_scale_correction=needs_correction,
        symmetry=(
            get_asset_symmetry(asset_type, obj_scale=obj_scale)
            if asset_type is not None else None
        ),
        default_grasp_offset=default_grasp_offset,
    )


def get_or_compute_prim_geometry(
    prim,
    asset_type_default: Optional[str] = None,
    prim_asset_info: Optional[tuple] = None,
    bb_cache=None,
) -> PrimGeometry:
    """Convenience method to first try looking up precomputed geometry, falling back to runtime computation.

    Args:
        prim: The loaded prim object.
        asset_type_default: Default asset type if unable to extract from semantic labels.
        prim_asset_info: Optional tuple of (orig_asset_type, obj_scale, obj_orientation) used for precomputed lookup.
        bb_cache: Optional bounding box cache.

    Returns:
        PrimGeometry with initialized geometry values.
    """
    if prim_asset_info is not None:
        orig_asset_type, obj_scale, obj_orientation = prim_asset_info
        geom = lookup_prim_geometry(orig_asset_type, obj_scale=obj_scale, orientation=obj_orientation)
        if geom is not None:
            return geom
        logger.info(f"No precomputed geometry for '{orig_asset_type}', falling back to runtime computation")

    asset_type_from_label = get_asset_type(prim, asset_type_default=asset_type_default)
    return compute_prim_geometry(prim, obj_scale=None, asset_type=asset_type_from_label, bb_cache=bb_cache)


def _apply_semantic_labels(
    prim,
    type_label: Optional[str] = None,
    obj_name: Optional[str] = None,
    color_labels: Optional[list[str]] = None,
    class_labels: Optional[list[str]] = None,
) -> None:
    """Attach semantic labels to a prim with best-effort logging.
    Semantic labels are used to check whether prim instances belong to particular categories.
    For example, to filter a list of objects by color or by type when matching potential source to target objects.

    - Adds the type label under instance name "type" when provided. The type label corresponds to the asset_type.
    - Adds the object name under instance name "name" when provided.
    - Adds one or more class labels under instance name "class" when provided.
    - Adds one or more color names under instance name "color" when provided.
    """
    try:
        if type_label:
            add_labels(prim, labels=[type_label], instance_name="type")
        if obj_name:
            add_labels(prim, labels=[obj_name], instance_name="name")
        if color_labels is not None:
            for color_name in color_labels:
                add_labels(prim, labels=[color_name], instance_name="color")
        if class_labels:
            for class_label in class_labels:
                add_labels(prim, labels=[class_label], instance_name="class")
    except Exception as e:
        import traceback
        try:
            # import omni.log
            logger.warning(f"apply_semantic_labels: failed to add labels to {prim.name}: {traceback.format_exc()}")
        except Exception:
            pass

def _get_semantic_labels(prim) -> dict[str, list[str]]:
    """
    Retrieves semantic labels that have been assigned to the given prim.

    For LightweightObj instances (which have no USD prim), returns the
    in-memory _semantic_labels dict directly.  For real USD prims, uses
    the UsdSemantics Labels API.
    """
    # LightweightObj stores labels in-memory (no USD prim to query).
    # Only use when populated — empty dict falls through to normal path
    # (e.g. MockPickObj with labels added via the mock store).
    in_memory = getattr(prim, "_semantic_labels", None)
    if in_memory:
        return in_memory

    try:
        from isaacsim.core.utils.semantics import get_labels  # lazy import
    except Exception:
        return None

    try:
        p = getattr(prim, "prim", prim)
        labels_map = get_labels(p)
        return labels_map
    except Exception:
        return None

def get_labels(prim, label_class:str) -> list[str]:
    labels_map = _get_semantic_labels(prim)
    if labels_map is not None:
        return [s.lower() for s in labels_map.get(label_class, [])]
    return []

def get_asset_type(current_obj, asset_type_default=None):
    asset_type_labels = get_labels(current_obj, "type")
    if asset_type_labels:
        asset_type = asset_type_labels[0]
    else:
        logger.warning(f"get_labels returned {asset_type_labels} for current_obj={current_obj.name}, using default asset_type={asset_type_default}")
        asset_type = asset_type_default
        logger.warning(f"get_lables = {_get_semantic_labels(current_obj)}")
    return asset_type

def has_label(prim, label_class:str, label_value) -> bool:
    """
      label_value can be either a string or a list of strings --
      if a list is passed, then returns success if any one matches
    """
    values_list = get_labels(prim, label_class)
    if type(label_value) is not list:
        targets_list = [label_value]
    else:
        targets_list = label_value
    for target_value in targets_list:
        if target_value.lower() in values_list:
            return True
    return False

def is_a(prim, class_label: str) -> bool:
    """Check if the prim belongs to the given class via semantic labels.

    Prefers labels under instance name "class" and falls back to matching
    against the "type" labels when class labels are absent. Case-insensitive.
    """

    labels_map = _get_semantic_labels(prim)
    target = class_label.lower()
    class_vals = [s.lower() for s in labels_map.get("class", [])]
    if target in class_vals:
        return True
    type_vals = [s.lower() for s in labels_map.get("type", [])]
    return target in type_vals

def is_of_type(prim, type_name: str) -> bool:
    """Check if the prim has a "type" label equal to type_name.
    """
    return has_label(prim, "type", type_name)

def has_color(prim, color_name: str) -> bool:
    """Check if the prim has a color label equal to color_name.
    """
    return has_label(prim, "color", color_name)

def add_usd_asset(scene,
                  asset_path,
                  asset_type=None,
                  obj_name=None,
                  position=None,
                  orientation=None,
                  scale=None,
                  assets_root_path=None,
                  prim_path=None,
                  scene_path_root="/",
                  color: Optional[str] = None,
                  visible: bool = True,
                  is_local: bool = False,
                  ):
    # If an asset_type is provided and obj_name is missing, generate a unique default
    if obj_name is None and asset_type is not None:
        obj_name = find_unique_string_name(
            initial_name=asset_type, is_unique_fn=lambda x: not scene.object_exists(x)
        )

    if not prim_path:
        assert obj_name is not None
        prim_scene_path = f"{scene_path_root}{obj_name}"
    elif not prim_path.startswith("/"):
        prim_scene_path = f"{scene_path_root}{prim_path}"
    else:
        prim_scene_path = prim_path
    if not obj_name:
        obj_name = prim_scene_path.split("/")[-1]

    if is_local:
        abs_file_path = asset_path
    elif assets_root_path and not asset_path.startswith(assets_root_path):
        abs_file_path = assets_root_path + asset_path
    else:
        abs_file_path = asset_path
    obj_prim = prims.create_prim(
            prim_scene_path,
            "Xform",
            position=position,
            orientation=orientation,
            scale=scale,
            usd_path=abs_file_path
        )
    if False and asset_type == "madara_bottle" and obj_prim:
        # Get the Xformable interface for the prim
        xformable = UsdGeom.Xformable(obj_prim)

        # Create and set the scale operation
        scale_op = xformable.AddScaleOp(
            opSuffix="unitsResolve",
            precision=UsdGeom.XformOp.PrecisionFloat
        )

        # Set the scale factor (e.g., 0.01 for cm to m)
        scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # Apply semantic labels: class -> asset_type (or derived from file), name -> obj_name
    xform_prim = SingleXFormPrim(prim_scene_path, name=obj_name)
    _asset_type = asset_type if asset_type is not None else os.path.splitext(os.path.basename(abs_file_path))[0]
    _apply_semantic_labels(obj_prim, type_label=_asset_type, obj_name=obj_name)
    if color is not None:
        _apply_semantic_labels(obj_prim, color_labels=[color])
    logger.warning(f"add_usd_asset: {abs_file_path} XFormPrim.name={xform_prim.name} {prim_scene_path}")
    scene.add(xform_prim)
    xform_prim.set_visibility(visible)
    return xform_prim


def add_prim_asset(scene,
                   asset_type="cube",
                   obj_name=None,
                   position=None,
                   orientation=None,
                   scale=None,
                   prim_path=None,
                   scene_path_root="/",
                   color=None,
                   visible=True,
                   ):
    if color is None:
        color = 'blue'
    if obj_name is None:
        obj_name = find_unique_string_name(
            initial_name=asset_type, is_unique_fn=lambda x: not scene.object_exists(x)
        )
    if not prim_path:
           prim_scene_path = f"{scene_path_root}{obj_name}"
    elif not prim_path.startswith("/"):
        prim_scene_path = f"{scene_path_root}{prim_path}"
    else:
        prim_scene_path = prim_path

    obj_prim_path = find_unique_string_name(
        initial_name=prim_scene_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
    )
    # Accept both named colors and explicit RGB triples
    color_name_label = None
    if isinstance(color, (list, tuple, np.ndarray)):
        color_value = np.array(color)
        # Attempt to infer a BasicColor name from the given RGB triple
        try:
            col_tuple = tuple(float(x) for x in np.array(color_value, dtype=float).tolist())
            for bc in BasicColor:
                if tuple(bc.value) == col_tuple:
                    color_name_label = bc.name.lower()
                    break
        except Exception:
            pass
    elif isinstance(color, str) and color.lower() in COLOR_MAP:
        color_value = np.array(COLOR_MAP[color.lower()].value)
        color_name_label = color.lower()
    else:
        color_value = np.array(BasicColor.RED.value)
        color_name_label = "red"
    # logger.warning(f"add_prim_asset: {asset_type} {obj_name} {obj_prim_path}")
    prim = scene.add(
            PRIMS_MAP[asset_type](
                name=obj_name,
                position=position,
                orientation=orientation,
                prim_path=obj_prim_path,
                scale=scale,
                # size=1.0,
                color=color_value
            ),
        )
    prim.set_visibility(visible)
    # Apply semantic labels to the created prim path (include color name when available)
    _apply_semantic_labels(prims.get_prim_at_path(obj_prim_path), type_label=asset_type, obj_name=obj_name)
    if color_name_label:
        _apply_semantic_labels(prims.get_prim_at_path(obj_prim_path), color_labels=[color_name_label])
    return prim


def add_asset(
    scene,
    asset_type,
    obj_name=None,
    position=None,
    orientation=None,
    scale=None,
    prim_path=None,
    scene_path_root="/",
    assets_root_path=None,
    color=None,
    asset_path=None,
    visible=True,
):
    """Create an object by type, dispatching to prim or USD helpers.

    Behavior
    - If `asset_type` is in PRIMS_MAP, create a primitive via `add_prim_asset`.
    - Else if in ITEMS_MAP, create a USD asset via `add_usd_asset` using the mapped metadata.
    - Else if `asset_path` is provided, create a USD asset directly.
    - Else, log an error and return None.
    """
    if asset_type in PRIMS_MAP:
        return add_prim_asset(
            scene,
            asset_type=asset_type,
            obj_name=obj_name,
            position=position,
            orientation=orientation,
            scale=scale,
            prim_path=prim_path,
            scene_path_root=scene_path_root,
            color=color,
            visible=visible,
        )
    elif asset_type in ITEMS_MAP:
        meta = ITEMS_MAP[asset_type]
        return add_usd_asset(
            scene,
            asset_path=meta.usd_path,
            asset_type=asset_type,
            obj_name=obj_name,
            position=position,
            orientation=orientation,
            scale=scale,
            assets_root_path=assets_root_path,
            prim_path=prim_path,
            scene_path_root=scene_path_root,
            color=color if color is not None else meta.color,
            visible=visible,
            is_local=meta.is_local_asset,
        )
    elif asset_path is not None:
        return add_usd_asset(
            scene,
            asset_path=asset_path,
            asset_type=asset_type,
            obj_name=obj_name,
            position=position,
            orientation=orientation,
            scale=scale,
            assets_root_path=assets_root_path,
            prim_path=prim_path,
            scene_path_root=scene_path_root,
            color=color,
            visible=visible,
        )
    else:
        try:
            # import omni.log
            logger.error(f"add_asset: Unknown asset_type '{asset_type}'")
        except Exception:
            pass
        return None

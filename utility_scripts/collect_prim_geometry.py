"""Collect PrimGeometry data for all asset types defined in ITEMS_MAP and PRIMS_MAP.

Launches IsaacSim headless, creates each asset at a known position, computes
its PrimGeometry via compute_prim_geometry(), and writes the results to
asset_prim_geometry.json in the project root.

Usage:
    mamba run -n env_isaacsim51 python utility_scripts/collect_prim_geometry.py
"""

import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

OUTPUT_FILE = os.path.join(project_root, "asset_prim_geometry.json")

# Spacing between assets along X axis and Z height above ground
X_SPACING = 2.0
ASSET_Z = 4.0

HEADLESS = True

def _build_asset_type_list():
    """Return deduplicated list of all asset types from ITEMS_MAP and PRIMS_MAP."""
    from asset_data_utils import ITEMS_MAP
    from asset_utils import PRIMS_MAP

    all_types = list(PRIMS_MAP.keys())
    for prim_type in ITEMS_MAP:
        if prim_type not in PRIMS_MAP:
            all_types.append(prim_type)
    return all_types


def _prim_geometry_to_dict(geom):
    """Convert a PrimGeometry dataclass to a JSON-serializable dict."""
    return {
        "grasp_height": float(geom.grasp_height),
        "rest_height": float(geom.rest_height),
        "top_surface_height": float(geom.top_surface_height),
        "local_half_extents": [float(v) for v in geom.local_half_extents],
        "needs_aabb_scale_correction": bool(geom.needs_aabb_scale_correction),
    }


def main():
    # ---- Initialize SimulationApp FIRST (must happen before any isaacsim.core imports) ----
    logger.info("Initializing SimulationApp in HEADLESS mode...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": HEADLESS})

    from isaacsim.core.api import World
    from isaacsim.core.utils.bounds import create_bbox_cache
    from isaacsim.storage.native import get_assets_root_path

    # Now safe to import asset_utils (which imports isaacsim.core at module level)
    from asset_utils import add_asset, compute_prim_geometry

    all_types = _build_asset_type_list()
    logger.info("Asset types to collect (%d): %s", len(all_types), all_types)

    my_world = World(stage_units_in_meters=1.0)
    scene = my_world.scene
    assets_root_path = get_assets_root_path()
    logger.info("Assets root path: %s", assets_root_path)

    # ---- Add all assets to the scene ----
    prim_refs = {}  # asset_type -> prim object
    errors = {}

    for i, asset_type in enumerate(all_types):
        position = np.array([i * X_SPACING, 0.0, ASSET_Z])
        obj_name = f"geom_collect_{asset_type}_{i}"
        try:
            prim = add_asset(
                scene,
                asset_type=asset_type,
                obj_name=obj_name,
                position=position,
                scale=np.array([1.0, 1.0, 1.0]),
                assets_root_path=assets_root_path,
            )
            if prim is None:
                msg = f"add_asset returned None for '{asset_type}'"
                logger.warning(msg)
                errors[asset_type] = msg
            else:
                prim_refs[asset_type] = prim
                logger.info("Added asset '%s' at position %s", asset_type, position)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to add asset '%s': %s", asset_type, msg)
            errors[asset_type] = msg

    # ---- Initialize physics so AABBs are available ----
    logger.info("Resetting world (initializing physics)...")
    my_world.reset()

    # ---- Compute geometry for each asset ----
    bb_cache = create_bbox_cache()
    results = {}

    for asset_type, prim in prim_refs.items():
        try:
            geom = compute_prim_geometry(
                prim, asset_type=asset_type, bb_cache=bb_cache,
            )
            results[asset_type] = _prim_geometry_to_dict(geom)
            logger.info(
                "  %s: half_extents=%s, grasp=%.4f, rest=%.4f, top=%.4f",
                asset_type,
                [f"{v:.4f}" for v in geom.local_half_extents],
                geom.grasp_height,
                geom.rest_height,
                geom.top_surface_height,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to compute geometry for '%s': %s", asset_type, msg)
            errors[asset_type] = msg

    # ---- Write output JSON ----
    output = dict(results)
    if errors:
        output["_errors"] = errors

    # if not HEADLESS:
    # while simulation_app.is_running():
    #     my_world.step(render=True) 

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote geometry data to %s (%d assets, %d errors)",
                OUTPUT_FILE, len(results), len(errors))

    # ---- Cleanup ----
    my_world.clear()
    simulation_app.close()


if __name__ == "__main__":
    main()

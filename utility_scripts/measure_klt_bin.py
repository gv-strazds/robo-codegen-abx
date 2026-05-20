"""Measure KLT Bin inner/outer dimensions via mesh vertex analysis.

Launches IsaacSim headless, loads the KLT bin at a specified scale,
extracts all mesh vertices, and analyzes the Z-distribution to find
inner cavity dimensions vs outer shell dimensions.

Usage:
    mamba run -n env_isaacsim51 python utility_scripts/measure_klt_bin.py
    mamba run -n env_isaacsim51 python utility_scripts/measure_klt_bin.py --scale 1.0 1.0 1.0
"""

import argparse
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

HEADLESS = True
Z_TOLERANCE = 0.001  # 1mm tolerance for clustering Z levels


def parse_args():
    parser = argparse.ArgumentParser(description="Measure KLT bin dimensions")
    parser.add_argument(
        "--scale", type=float, nargs=3, default=[1.5, 1.5, 0.5],
        help="Scale factors [x, y, z] (default: 1.5 1.5 0.5)",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Optional path to write JSON output",
    )
    return parser.parse_args()


def extract_mesh_vertices_world(stage, root_prim_path):
    """Extract all mesh vertices under root_prim_path in world coordinates.

    Returns:
        all_verts: np.ndarray of shape (N, 3) — all vertices in world frame
        mesh_info: list of dicts with per-mesh details
    """
    from pxr import Usd, UsdGeom

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim not found at {root_prim_path}")

    all_verts = []
    mesh_info = []

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue

        local_verts = np.array(points, dtype=np.float64)

        # Get world transform
        xformable = UsdGeom.Xformable(prim)
        world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tf_matrix = np.array(world_tf, dtype=np.float64)  # 4x4 row-major

        # Transform to world coords: homogeneous multiply
        hom = np.ones((local_verts.shape[0], 4), dtype=np.float64)
        hom[:, :3] = local_verts
        world_verts = (hom @ tf_matrix)[:, :3]

        all_verts.append(world_verts)
        mesh_info.append({
            "prim_path": str(prim.GetPath()),
            "num_vertices": len(world_verts),
            "min": world_verts.min(axis=0).tolist(),
            "max": world_verts.max(axis=0).tolist(),
        })

    if not all_verts:
        raise ValueError(f"No mesh vertices found under {root_prim_path}")

    return np.vstack(all_verts), mesh_info


def find_z_levels(z_values, tolerance):
    """Find distinct Z levels by rounding and counting unique values.

    Returns sorted list of (z_level, count) tuples.
    """
    rounded = np.round(z_values / tolerance) * tolerance
    unique_z, counts = np.unique(rounded, return_counts=True)
    levels = sorted(zip(unique_z.tolist(), counts.tolist()), key=lambda x: x[0])
    return levels


def main():
    args = parse_args()
    scale = np.array(args.scale)

    logger.info("Initializing SimulationApp in HEADLESS mode...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": HEADLESS})

    from isaacsim.core.api import World
    from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Usd, UsdGeom

    from asset_utils import add_asset

    my_world = World(stage_units_in_meters=1.0)
    scene = my_world.scene
    assets_root_path = get_assets_root_path()

    # Add KLT bin at origin
    position = np.array([0.0, 0.0, 0.0])
    prim = add_asset(
        scene,
        asset_type="KLT_Bin",
        obj_name="klt_measure",
        position=position,
        scale=scale,
        assets_root_path=assets_root_path,
    )
    if prim is None:
        logger.error("Failed to add KLT_Bin asset")
        simulation_app.close()
        return

    prim_path = prim.prim_path
    logger.info("KLT bin added at %s with scale %s", prim_path, scale.tolist())

    # Initialize physics so AABBs are available
    my_world.reset()

    # --- Compute outer AABB via Isaac Sim ---
    bb_cache = create_bbox_cache()
    aabb = compute_aabb(bb_cache, prim_path=prim_path, include_children=True)
    aabb_min = aabb[:3]
    aabb_max = aabb[3:]
    aabb_size = aabb_max - aabb_min
    logger.info("AABB min: %s", aabb_min)
    logger.info("AABB max: %s", aabb_max)
    logger.info("AABB size: %s", aabb_size)

    # --- Extract all mesh vertices ---
    usd_stage = my_world.stage
    all_verts, mesh_info = extract_mesh_vertices_world(usd_stage, prim_path)
    logger.info("Total mesh vertices: %d from %d meshes", len(all_verts), len(mesh_info))

    for mi in mesh_info:
        logger.info("  Mesh: %s (%d verts) min=%s max=%s",
                     mi["prim_path"], mi["num_vertices"],
                     [f"{v:.4f}" for v in mi["min"]],
                     [f"{v:.4f}" for v in mi["max"]])

    # --- Analyze Z distribution ---
    z_values = all_verts[:, 2]
    z_levels = find_z_levels(z_values, Z_TOLERANCE)

    logger.info("Z levels found (%d):", len(z_levels))
    for z_val, count in z_levels:
        logger.info("  Z=%.4f  (%d vertices)", z_val, count)

    if len(z_levels) < 3:
        logger.warning("Expected at least 3 Z levels (outer bottom, inner floor, top rim), got %d", len(z_levels))

    z_outer_bottom = z_levels[0][0]
    z_inner_floor = z_levels[1][0] if len(z_levels) > 1 else z_levels[0][0]
    z_top = z_levels[-1][0]

    # --- Find inner X/Y bounds at inner floor level ---
    floor_mask = np.abs(all_verts[:, 2] - z_inner_floor) < Z_TOLERANCE * 2
    floor_verts = all_verts[floor_mask]

    if len(floor_verts) == 0:
        logger.error("No vertices found at inner floor level Z=%.4f", z_inner_floor)
        simulation_app.close()
        return

    inner_min_x = floor_verts[:, 0].min()
    inner_max_x = floor_verts[:, 0].max()
    inner_min_y = floor_verts[:, 1].min()
    inner_max_y = floor_verts[:, 1].max()

    # --- Compute dimensions ---
    outer_x = aabb_size[0]
    outer_y = aabb_size[1]
    outer_z = aabb_size[2]

    inner_x = inner_max_x - inner_min_x
    inner_y = inner_max_y - inner_min_y
    inner_z = z_top - z_inner_floor

    wall_x = (outer_x - inner_x) / 2
    wall_y = (outer_y - inner_y) / 2
    floor_thickness = z_inner_floor - z_outer_bottom

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"KLT Bin Dimensions (scale = {scale.tolist()})")
    print("=" * 60)
    print(f"\nOuter dimensions (from AABB):")
    print(f"  X: {outer_x:.4f} m")
    print(f"  Y: {outer_y:.4f} m")
    print(f"  Z: {outer_z:.4f} m")
    print(f"\nOuter dimensions (from vertices):")
    verts_outer_size = all_verts.max(axis=0) - all_verts.min(axis=0)
    print(f"  X: {verts_outer_size[0]:.4f} m")
    print(f"  Y: {verts_outer_size[1]:.4f} m")
    print(f"  Z: {verts_outer_size[2]:.4f} m")
    print(f"\nInner cavity dimensions (from floor vertices):")
    print(f"  X: {inner_x:.4f} m")
    print(f"  Y: {inner_y:.4f} m")
    print(f"  Z (depth): {inner_z:.4f} m")
    print(f"\nWall thickness:")
    print(f"  X walls: {wall_x:.4f} m")
    print(f"  Y walls: {wall_y:.4f} m")
    print(f"  Floor: {floor_thickness:.4f} m")
    print(f"\nZ levels:")
    print(f"  Outer bottom: {z_outer_bottom:.4f} m")
    print(f"  Inner floor:  {z_inner_floor:.4f} m")
    print(f"  Top rim:      {z_top:.4f} m")
    print(f"\nInner cavity center (world):")
    inner_cx = (inner_min_x + inner_max_x) / 2
    inner_cy = (inner_min_y + inner_max_y) / 2
    print(f"  X: {inner_cx:.4f} m")
    print(f"  Y: {inner_cy:.4f} m")

    print(f"\n--- Suggested BIN_SIZE replacement for table_setup.py ---")
    print(f"BIN_SIZE = [{inner_x:.4f}, {inner_y:.4f}, {inner_z:.4f}]"
          f"  # inner cavity at scale {scale.tolist()}")

    # Compute unscaled inner dimensions for reference
    unscaled_inner = np.array([inner_x, inner_y, inner_z]) / scale
    print(f"# Unscaled inner: [{unscaled_inner[0]:.4f}, {unscaled_inner[1]:.4f}, {unscaled_inner[2]:.4f}]")
    print()

    # --- Optional JSON output ---
    if args.json:
        output = {
            "scale": scale.tolist(),
            "aabb": {"min": aabb_min.tolist(), "max": aabb_max.tolist(), "size": aabb_size.tolist()},
            "outer": {"x": outer_x, "y": outer_y, "z": outer_z},
            "inner": {"x": inner_x, "y": inner_y, "z": inner_z},
            "wall_thickness": {"x": wall_x, "y": wall_y, "floor": floor_thickness},
            "z_levels": {
                "outer_bottom": z_outer_bottom,
                "inner_floor": z_inner_floor,
                "top_rim": z_top,
            },
            "inner_bounds": {
                "min_x": inner_min_x, "max_x": inner_max_x,
                "min_y": inner_min_y, "max_y": inner_max_y,
            },
            "meshes": mesh_info,
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Wrote JSON output to %s", args.json)

    # --- Cleanup ---
    my_world.clear()
    simulation_app.close()


if __name__ == "__main__":
    main()

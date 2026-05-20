"""Mock implementation of isaacsim.core.utils.bounds for testing."""
import numpy as np


def create_bbox_cache(*args, **kwargs):
    """Return a dummy bbox cache."""
    return None


def compute_aabb(cache=None, prim_path=None, include_children=True):
    """Return a default unit AABB [0,0,0,1,1,1].

    Tests should monkeypatch this when specific AABB values are needed.
    """
    return np.array([0, 0, 0, 1, 1, 1], dtype=float)


def compute_obb(cache=None, prim_path=None):
    """Return a dummy OBB (centroid, axes, half_extent)."""
    centroid = np.array([0.5, 0.5, 0.5])
    axes = np.eye(3)
    half_extent = np.array([0.5, 0.5, 0.5])
    return centroid, axes, half_extent

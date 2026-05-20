def get_stage_units() -> float:
    """Mock implementation returning 1.0 (meters)."""
    return 1.0


def add_reference_to_stage(*args, **kwargs):
    """Mock: no-op for adding USD references."""
    return None


def traverse_stage(*args, **kwargs):
    """Mock: no-op for traversing stage."""
    return []

"""Minimal mock of isaacsim.cortex.framework.robot for import compatibility.

Only provides CortexUr10 as a name so that modules importing it at the top
level don't fail in mock/test environments.  The actual CortexUr10 is never
instantiated in mock mode.
"""


class CortexUr10:
    """Stub — never instantiated in mock mode."""
    pass


def add_ur10_to_stage(*args, **kwargs):
    """Stub — never called in mock mode."""
    raise NotImplementedError("add_ur10_to_stage requires real IsaacSim")

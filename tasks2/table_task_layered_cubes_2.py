"""Cortex-tree variant of TableTaskLayeredCubes.

Identical to :class:`TableTaskLayeredCubes` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_layered_cubes.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_layered_cubes import TableTaskLayeredCubes


class TableTaskLayeredCubes2(TableTaskLayeredCubes):
    DEFAULT_TASK_NAME = "table_task_layered_cubes_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Unstack 18 cubes from a 3-layer grid in the bin and place them onto a flat grid on the dropzone (cortex-style BT).",
        )

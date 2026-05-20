"""ContextMonitorBehaviour: queries TaskContext every tick and writes to blackboard.

Runs as the first child of a Parallel(SuccessOnOne) alongside the task
orchestration tree, ensuring fresh simulation state is available in the
/pickplace/ and /task/ blackboard namespaces before task behaviours tick.
"""
import logging

import py_trees

from robot_controllers.pt_pick_place_behaviours import INPUT_KEYS

logger = logging.getLogger(__name__)


class ContextMonitorBehaviour(py_trees.behaviour.Behaviour):
    """Queries TaskContext every tick and writes to /pickplace/ and /task/ blackboard.

    Always returns RUNNING so the Parallel keeps ticking it alongside the
    task orchestration tree.
    """

    def __init__(self, name: str = "ContextMonitor"):
        super().__init__(name=name)
        self._context = None

        # Blackboard client for /pickplace/ namespace (write)
        self.pp_bb = self.attach_blackboard_client(
            name=f"{name}_pp", namespace="/pickplace"
        )
        for key in INPUT_KEYS:
            self.pp_bb.register_key(key=key, access=py_trees.common.Access.WRITE)

        # Blackboard client for /task/ namespace (write)
        self.task_bb = self.attach_blackboard_client(
            name=f"{name}_task", namespace="/task"
        )
        self.task_bb.register_key(key="task_finished", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs) -> None:
        """Receive context via kwargs (passed from BehaviourTree.setup())."""
        if "context" in kwargs:
            self._context = kwargs["context"]

    def initialise(self) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.RUNNING

        # Poll target reachability to detect fallen-off targets early
        self._context.strategy.poll_target_reachability()
        # Poll pick positions so JIT pick strategies can track settled state.
        self._context.strategy.poll_pick_positions()

        pick_name = self._context.get_current_pick_name()
        if pick_name is not None:
            # current robot joint positions
            joint_positions = self._context.get_joint_positions()
            self.pp_bb.current_joint_positions = joint_positions

            # Write pick/place inputs

            picking_pos = self._context.get_picking_position(pick_name)
            ee_offset = self._context.get_end_effector_offset(pick_name)
            ee_orientation = self._context.get_end_effector_orientation(pick_name)

            self.pp_bb.picking_position = picking_pos
            self.pp_bb.end_effector_offset = ee_offset
            self.pp_bb.end_effector_orientation = ee_orientation
            self.pp_bb.ee_height_for_move = self._context.get_ee_height_for_move()

            # Strategy-specific orientations
            target_name = self._context.get_placing_target_name(pick_name)
            drop_orient = self._context.get_end_effector_orientation_for_drop(pick_name, target_name)

            # Geometry-computed placing info
            _, target_pos, target_orient = self._context.get_placing_info(pick_name, drop_orient)
            self.pp_bb.placing_position = target_pos if target_pos is not None else picking_pos

            # Geometry-computed drop offset
            drop_offset = self._context.get_end_effector_offset_for_drop(pick_name, drop_orient)
            self.pp_bb.end_effector_offset_for_drop = drop_offset if drop_offset is not None else ee_offset
            self.pp_bb.end_effector_orientation_for_drop = drop_orient if drop_orient is not None else ee_orientation

        # Write task status
        self.task_bb.task_finished = self._context.task_finished

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        pass

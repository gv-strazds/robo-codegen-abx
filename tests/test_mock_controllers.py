"""Tests for mock robot controller implementations.

Verifies that controllers can be instantiated and used with mock
implementations without requiring the full IsaacSim simulator.
"""
import sys
import os
import pytest
import numpy as np

# Add extsMock to path BEFORE importing robot_controllers
# This allows testing without the real IsaacSim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extsMock'))

# Now we can import from extsMock
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.controllers.base_controller import BaseController

# Import MockRobot from robot_controllers
from robot_controllers.mock_robot import (
    MockGripper,
    MockEndEffectorController,
    MockArticulationController,
    MockRobotArticulation,
)


class TestMockGripper:
    """Tests for MockGripper implementation."""
    
    def test_init(self):
        """Test gripper initialization."""
        gripper = MockGripper(num_dofs=8)
        assert gripper._num_dofs == 8
        assert not gripper.is_closed()
        assert gripper.is_open()
    
    def test_open_close(self):
        """Test opening and closing gripper."""
        gripper = MockGripper()
        
        gripper.close()
        assert gripper.is_closed()
        assert not gripper.is_open()
        
        gripper.open()
        assert gripper.is_open()
        assert not gripper.is_closed()
    
    def test_forward_action(self):
        """Test forward method returns ArticulationAction."""
        gripper = MockGripper(num_dofs=8)
        
        action = gripper.forward("close")
        assert isinstance(action, ArticulationAction)
        assert len(action.joint_positions) == 8
        assert gripper.is_closed()
        
        action = gripper.forward("open")
        assert isinstance(action, ArticulationAction)
        assert gripper.is_open()
    
    def test_set_joint_positions(self):
        """Test ParallelGripper-compatible set_joint_positions."""
        gripper = MockGripper()
        positions = np.array([0.02, 0.02])
        
        gripper.set_joint_positions(positions)
        result = gripper.get_joint_positions()
        np.testing.assert_array_equal(result, positions)


class TestMockEndEffectorController:
    """Tests for MockEndEffectorController implementation."""
    
    def test_init(self):
        """Test controller initialization."""
        controller = MockEndEffectorController(num_joints=6)
        assert controller._num_joints == 6
    
    def test_forward_returns_action(self):
        """Test forward method returns correct ArticulationAction."""
        controller = MockEndEffectorController(num_joints=6)
        target_pos = np.array([0.5, 0.2, 0.3])
        target_orient = np.array([1.0, 0.0, 0.0, 0.0])
        
        action = controller.forward(target_pos, target_orient)
        
        assert isinstance(action, ArticulationAction)
        assert action.joint_positions is not None
        assert len(action.joint_positions) == 6
        np.testing.assert_array_equal(controller._last_target_position, target_pos)
    
    def test_reset(self):
        """Test reset clears state."""
        controller = MockEndEffectorController()
        controller.forward(np.array([0.5, 0.2, 0.3]))
        
        controller.reset()
        
        assert controller._last_target_position is None
        assert controller._last_target_orientation is None


class TestMockArticulationController:
    """Tests for MockArticulationController implementation."""
    
    def test_apply_action_records_history(self):
        """Test that apply_action records actions."""
        controller = MockArticulationController()
        action = ArticulationAction(joint_positions=np.zeros(6))
        
        controller.apply_action(action)
        
        assert controller.last_action is action
        assert len(controller.applied_actions) == 1
        assert controller.applied_actions[0] is action
    
    def test_clear_history(self):
        """Test clearing action history."""
        controller = MockArticulationController()
        controller.apply_action(ArticulationAction(joint_positions=np.zeros(6)))
        
        controller.clear_history()
        
        assert controller.last_action is None
        assert len(controller.applied_actions) == 0


class TestMockRobotArticulation:
    """Tests for MockRobotArticulation implementation."""
    
    def test_init(self):
        """Test robot initialization with defaults."""
        robot = MockRobotArticulation()
        assert robot.name == "mock_robot"
    
    def test_custom_name(self):
        """Test robot with custom name."""
        robot = MockRobotArticulation(name="my_ur10")
        assert robot.name == "my_ur10"
    
    def test_gripper_access(self):
        """Test gripper property returns MockGripper."""
        robot = MockRobotArticulation()
        assert isinstance(robot.gripper, MockGripper)
    
    def test_articulation_controller(self):
        """Test articulation controller access."""
        robot = MockRobotArticulation()
        controller = robot.get_articulation_controller()
        assert isinstance(controller, MockArticulationController)


class TestProtocolCompliance:
    """Test that mock implementations satisfy Protocol requirements."""
    
    def test_gripper_has_required_methods(self):
        """Verify MockGripper has all required IGripper methods."""
        gripper = MockGripper()
        assert callable(getattr(gripper, 'forward', None))
        assert callable(getattr(gripper, 'open', None))
        assert callable(getattr(gripper, 'close', None))
    
    def test_ee_controller_has_required_methods(self):
        """Verify MockEndEffectorController has IEndEffectorController methods."""
        controller = MockEndEffectorController()
        assert callable(getattr(controller, 'forward', None))
        assert callable(getattr(controller, 'reset', None))
    
    def test_articulation_controller_has_apply_action(self):
        """Verify MockArticulationController has apply_action."""
        controller = MockArticulationController()
        assert callable(getattr(controller, 'apply_action', None))
    
    def test_robot_has_required_properties(self):
        """Verify MockRobotArticulation has IRobotArticulation interface."""
        robot = MockRobotArticulation()
        assert hasattr(robot, 'name')
        assert hasattr(robot, 'gripper')
        assert callable(getattr(robot, 'get_articulation_controller', None))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Simple verification script for mock robot implementations.

Runs without pytest to verify basic functionality.
"""
import sys
import os

# Add extsMock to path BEFORE importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extsMock'))

import numpy as np

# Import from extsMock
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.controllers.base_controller import BaseController

# Import MockRobot from robot_controllers
from robot_controllers.mock_robot import (
    MockGripper,
    MockEndEffectorController,
    MockArticulationController,
    MockRobotArticulation,
)


def test_mock_gripper():
    """Test MockGripper functionality."""
    print("Testing MockGripper...")
    gripper = MockGripper(num_dofs=8)
    
    # Test initial state
    assert not gripper.is_closed(), "Gripper should start open"
    assert gripper.is_open(), "Gripper should be open"
    
    # Test close
    gripper.close()
    assert gripper.is_closed(), "Gripper should be closed"
    
    # Test open
    gripper.open()
    assert gripper.is_open(), "Gripper should be open"
    
    # Test forward action
    action = gripper.forward("close")
    assert isinstance(action, ArticulationAction), "forward should return ArticulationAction"
    assert len(action.joint_positions) == 8, "Should have 8 DOFs"
    
    # Test set_joint_positions (ParallelGripper compatibility)
    positions = np.array([0.02, 0.02])
    gripper.set_joint_positions(positions)
    result = gripper.get_joint_positions()
    assert np.allclose(result, positions), "Joint positions should match"
    
    print("  ✓ MockGripper tests passed")


def test_mock_end_effector_controller():
    """Test MockEndEffectorController functionality."""
    print("Testing MockEndEffectorController...")
    controller = MockEndEffectorController(num_joints=6)
    
    target_pos = np.array([0.5, 0.2, 0.3])
    target_orient = np.array([1.0, 0.0, 0.0, 0.0])
    
    action = controller.forward(target_pos, target_orient)
    
    assert isinstance(action, ArticulationAction), "forward should return ArticulationAction"
    assert action.joint_positions is not None, "Should have joint positions"
    assert len(action.joint_positions) == 6, "Should have 6 joints"
    
    # Test reset
    controller.reset()
    assert controller._last_target_position is None, "Reset should clear state"
    
    print("  ✓ MockEndEffectorController tests passed")


def test_mock_articulation_controller():
    """Test MockArticulationController functionality."""
    print("Testing MockArticulationController...")
    controller = MockArticulationController()
    
    action = ArticulationAction(joint_positions=np.zeros(6))
    controller.apply_action(action)
    
    assert controller.last_action is action, "Should record last action"
    assert len(controller.applied_actions) == 1, "Should have 1 action in history"
    
    controller.clear_history()
    assert controller.last_action is None, "Should clear last action"
    assert len(controller.applied_actions) == 0, "Should clear history"
    
    print("  ✓ MockArticulationController tests passed")


def test_mock_robot_articulation():
    """Test MockRobotArticulation functionality."""
    print("Testing MockRobotArticulation...")
    robot = MockRobotArticulation(name="my_ur10", num_joints=6)
    
    assert robot.name == "my_ur10", "Name should match"
    assert isinstance(robot.gripper, MockGripper), "Should have MockGripper"
    assert isinstance(robot.get_articulation_controller(), MockArticulationController), \
        "Should have MockArticulationController"
    
    # Test duck-typing for ParallelGripper reset pattern
    gripper = robot.gripper
    if hasattr(gripper, 'set_joint_positions') and hasattr(gripper, 'joint_opened_positions'):
        gripper.set_joint_positions(gripper.joint_opened_positions)
        print("  ✓ Duck-typing for ParallelGripper reset works")
    
    print("  ✓ MockRobotArticulation tests passed")


def test_base_controller_available():
    """Test BaseController is available in extsMock."""
    print("Testing BaseController availability...")
    
    # Create a concrete implementation 
    class SimpleController(BaseController):
        def forward(self, *args, **kwargs) -> ArticulationAction:
            return ArticulationAction(joint_positions=[None] * 6)
    
    controller = SimpleController(name="test_controller")
    assert controller.name == "test_controller", "Name should match"
    
    action = controller.forward()
    assert isinstance(action, ArticulationAction), "forward should return ArticulationAction"
    
    print("  ✓ BaseController tests passed")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("Mock Robot Implementation Verification")
    print("=" * 60)
    print()
    
    try:
        test_mock_gripper()
        test_mock_end_effector_controller()
        test_mock_articulation_controller()
        test_mock_robot_articulation()
        test_base_controller_available()
        
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

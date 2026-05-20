# RMP Flow Controller — How Motion Planning Works

> **Note (pt-cortex branch):** This document describes the low-level RMPFlow motion
> planning internals. As of the Cortex refactoring, behaviours no longer call
> `cspace_controller.forward()` directly. Instead they call
> `arm_commander.send_ee_target(position, orientation)`, which goes through
> `CortexArmAdapter` → `MotionCommander.send()` → `MotionCommand`. The
> `MotionCommander` internally uses the same RMPFlow policy described here
> and applies the resulting `ArticulationAction` during `CortexWorld.step()`.

## Overview

The RMPFlow motion planning step takes a desired end-effector pose in Cartesian space and computes **joint-space commands** needed to move toward that pose. Under the Cortex framework, this is handled internally by the `MotionCommander` (which wraps `ArticulationMotionPolicy` + `RmpFlow`).

```python
cspace_controller.forward(
    target_end_effector_position=np.array([x, y, z]),       # desired EE position
    target_end_effector_orientation=np.array([w, x, y, z]), # desired EE quaternion (optional)
) -> ArticulationAction
```

## Implementation Chain (Isaac Sim)

### 1. RMPFlowController (UR10-specific)

`exts/.../universal_robots/controllers/rmpflow_controller.py`

Subclass that loads UR10-specific RMP flow config. Does **not** override `forward()` — delegates entirely to parent.

### 2. MotionPolicyController.forward()

`exts/.../motion_generation/motion_policy_controller.py`

```python
def forward(self, target_end_effector_position, target_end_effector_orientation=None):
    self._motion_policy.set_end_effector_target(position, orientation)
    self._motion_policy.update_world()
    action = self._articulation_motion_policy.get_next_articulation_action()
    return action
```

### 3. ArticulationMotionPolicy.get_next_articulation_action()

`exts/.../motion_generation/articulation_motion_policy.py`

Reads current joint positions/velocities from the robot, calls `motion_policy.compute_joint_targets()` with the physics timestep, and wraps the result in an `ArticulationAction(joint_positions=..., joint_velocities=...)`.

### 4. RmpFlow.compute_joint_targets() and Euler Integration

`exts/.../motion_generation/lula/motion_policies.py`

The core algorithm. RMP operates in **acceleration space** with fixed-size internal substeps:

```python
def _euler_integration(self, joint_positions, joint_velocities, frame_duration):
    num_steps = np.ceil(frame_duration / self.maximum_substep_size).astype(int)
    policy_timestep = frame_duration / num_steps

    for i in range(num_steps):
        joint_accel = self._evaluate_acceleration(joint_positions, joint_velocities)
        joint_positions += policy_timestep * joint_velocities
        joint_velocities += policy_timestep * joint_accel

    return joint_positions, joint_velocities
```

`_evaluate_acceleration()` calls into NVIDIA's Lula C++ library (`self._policy.eval_accel()`), which evaluates the RMP tree — a superposition of task-space attractors and repulsors.

## Step Size Behavior

### Substep size is fixed, not adaptive

The `maximum_substep_size` is a config parameter (recommended ~1/200s or 0.005s). If the physics frame is 1/60s and max substep is 1/300s, the algorithm takes `ceil(5) = 5` internal substeps per frame.

### Motion magnitude IS distance-dependent

The **acceleration** produced by the RMP tree depends on the error between current and target EE pose:

- **Far from target** — large attractor acceleration — velocity builds up — large joint position change per frame
- **Near target** — small attractor acceleration — velocity is damped — small corrections (converging smoothly)

So while the timestep is fixed, the *magnitude of motion per frame* varies with distance. The algorithm naturally produces smooth approach behavior without explicit convergence checking.

### One `forward()` call = one physics frame

The `frame_duration` is passed by `ArticulationMotionPolicy`. After integration, RmpFlow returns joint position + velocity targets as an `ArticulationAction`. Isaac Sim then applies these via its internal PD controller: `kp*(target_pos - current_pos) + kd*(target_vel - current_vel)`.

## How Pick-Place Behaviours Use This

The behaviours in `pt_pick_place_behaviours.py` use time-based interpolation (`t += dt`) to smoothly update the Cartesian target each tick (e.g., along a sinusoidal trajectory from move height down to pick height). The RMP algorithm tracks whatever target it's given, taking whatever joint-space step is needed. The behaviours don't check convergence — they rely on the RMP's natural smooth-tracking behavior.

Seven of the nine pick-place phases send end-effector targets via `arm_commander.send_ee_target()`:
- MoveToPickXY, LowerToPick, LiftPicked
- MoveToPlaceXY, LowerToPlace, LiftAfterPlace

The two gripper phases (CloseGripper, OpenGripper) call `gripper_commander.close()` / `gripper_commander.open()` directly.

## Interface Contract

Defined as the `IEndEffectorController` protocol in `robot_controllers/robot_interfaces.py`:

```python
def forward(
    self,
    target_end_effector_position: np.ndarray,
    target_end_effector_orientation: Optional[np.ndarray] = None,
) -> ArticulationAction:
    """Compute joint actions to reach target end-effector pose."""
```

## Mock Implementation

`MockEndEffectorController` in `robot_controllers/mock_robot.py` returns `ArticulationAction(joint_positions=np.zeros(num_joints))` — no actual IK computation. Used for testing the behavior tree logic without Isaac Sim.

## Action Application Path (Cortex)

With the Cortex framework, computation and application are handled by the `MotionCommander`:

1. Behaviour calls `arm_commander.send_ee_target(position, orientation)`
2. `CortexArmAdapter` creates a `MotionCommand` and calls `robot.arm.send(command)`
3. `CortexWorld.step()` calls `robot.pre_step()` → `MotionCommander.step()`
4. `MotionCommander` uses RMPFlow to compute joint targets → `ArticulationAction`
5. `MotionCommander` applies the action via `articulation_controller.apply_action()`
6. Isaac Sim's PD controller applies joint commands to the simulated robot

In teleport mode, `NullArmCommander` swallows all commands silently — the robot stays still while objects are teleported directly.

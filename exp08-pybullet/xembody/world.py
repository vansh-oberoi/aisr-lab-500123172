"""The pick-and-place task and the shared (embodiment-agnostic) interface.

Shared ACTION space (identical for every robot), 4 numbers in [-1, 1]:
    a = [dx, dy, dz, grip]
    dx, dy, dz : tool-tip displacement, scaled by MAX_STEP metres per control step
    grip       : > 0 means "suction on", <= 0 means "suction off"

Shared OBSERVATION (identical meaning for every robot), 10 numbers:
    [tcp_x, tcp_y, tcp_z, holding, cube_x, cube_y, cube_z, goal_x, goal_y, goal_z]
All positions are in the robot's own base frame, so a policy never needs to
know where the robot stands or what it looks like.
"""

import time

import numpy as np
import pybullet as p
import pybullet_data

from .robots import Robot

SIM_DT = 1.0 / 240.0          # physics step (s)
SUBSTEPS = 8                  # physics steps per control step -> 30 Hz control
MAX_STEP = 0.02               # metres of tool motion per control step at |a| = 1
CUBE_HALF = 0.025             # 5 cm cube
GRASP_RADIUS = 0.03           # suction works if the tool tip is this close to the cube top
GOAL_TOL = 0.04               # success if the cube centre lands within 4 cm of the goal
WORKSPACE_LO = np.array([0.15, -0.45, 0.01])
WORKSPACE_HI = np.array([0.80, 0.45, 0.50])
MAX_STEPS = 300               # control steps per episode (10 s)


class Station:
    """One robot + one cube + one goal marker, placed at `origin` in a shared world."""

    def __init__(self, robot_name, client, origin=(0, 0, 0)):
        self.cid = client
        self.origin = np.array(origin, dtype=float)
        self.robot = Robot(robot_name, client, base_pos=origin)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3, physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3,
                                  rgbaColor=[0.85, 0.25, 0.2, 1], physicsClientId=client)
        self.cube = p.createMultiBody(0.1, col, vis, basePosition=(self.origin + [0.5, 0, CUBE_HALF]).tolist(),
                                      physicsClientId=client)
        p.changeDynamics(self.cube, -1, lateralFriction=1.0, physicsClientId=client)
        gvis = p.createVisualShape(p.GEOM_CYLINDER, radius=GOAL_TOL, length=0.002,
                                   rgbaColor=[0.2, 0.75, 0.3, 0.6], physicsClientId=client)
        self.goal_marker = p.createMultiBody(0, -1, gvis, basePosition=(self.origin + [0.5, 0.2, 0.001]).tolist(),
                                             physicsClientId=client)
        self.constraint = None
        self.goal = np.zeros(3)

    # -------------------------------------------------------------- episode
    def reset(self, rng):
        """Random cube on the right (y<0), random goal on the left (y>0)."""
        self.release()
        self.robot.reset()
        cube_xy = [rng.uniform(0.42, 0.60), rng.uniform(-0.25, -0.08)]
        goal_xy = [rng.uniform(0.42, 0.60), rng.uniform(0.08, 0.25)]
        p.resetBasePositionAndOrientation(self.cube, (self.origin + [*cube_xy, CUBE_HALF]).tolist(),
                                          [0, 0, 0, 1], physicsClientId=self.cid)
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        self.goal = np.array([*goal_xy, 0.0])
        p.resetBasePositionAndOrientation(self.goal_marker, (self.origin + [*goal_xy, 0.001]).tolist(),
                                          [0, 0, 0, 1], physicsClientId=self.cid)
        self.steps, self.done, self.success = 0, False, False

    def cube_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.cube, physicsClientId=self.cid)
        return np.array(pos) - self.origin

    def holding(self):
        return self.constraint is not None

    def observe(self):
        return np.concatenate([self.robot.tcp_position(), [float(self.holding())],
                               self.cube_position(), self.goal])

    # -------------------------------------------------------------- gripper
    def try_grasp(self):
        tcp = self.robot.tcp_position()
        top = self.cube_position() + [0, 0, CUBE_HALF]
        if np.linalg.norm(tcp - top) > GRASP_RADIUS:
            return
        ee_pos, ee_orn = self.robot.ee_pose()
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube, physicsClientId=self.cid)
        inv_pos, inv_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, cube_pos, cube_orn)
        self.constraint = p.createConstraint(
            self.robot.uid, self.robot.ee_index, self.cube, -1, p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn, childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.cid)
        p.changeConstraint(self.constraint, maxForce=50, physicsClientId=self.cid)

    def release(self):
        if self.constraint is not None:
            p.removeConstraint(self.constraint, physicsClientId=self.cid)
            self.constraint = None

    # -------------------------------------------------------------- control
    def apply_action(self, action):
        """Shared task-space action -> this robot's joints (via its adapter)."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        tcp = self.robot.tcp_position()
        target = self.robot.target + a[:3] * MAX_STEP
        # keep the commanded target close to the real tool tip (no wind-up) and inside the workspace
        target = tcp + np.clip(target - tcp, -0.05, 0.05)
        target = np.clip(target, WORKSPACE_LO, WORKSPACE_HI)
        self.robot.command_tcp(target)
        self._grip_cmd = a[3]

    def apply_joint_command(self, q_arm, grip):
        """Joint-space command (used only by the naive transfer baseline)."""
        self.robot.command_joints(q_arm)
        self._grip_cmd = grip

    def after_physics(self):
        """Gripper logic + success bookkeeping, once per control step."""
        if self._grip_cmd > 0 and not self.holding():
            self.try_grasp()
        elif self._grip_cmd <= 0 and self.holding():
            self.release()
        self.steps += 1
        cube = self.cube_position()
        on_goal = np.linalg.norm(cube[:2] - self.goal[:2]) < GOAL_TOL
        resting = cube[2] < CUBE_HALF + 0.01 and not self.holding()
        if on_goal and resting:
            self.success = True
        if self.success or self.steps >= MAX_STEPS:
            self.done = True


class World:
    """A PyBullet client holding one or more stations that step together."""

    def __init__(self, robot_names, gui=False, spacing=1.6, realtime=False):
        if isinstance(robot_names, str):
            robot_names = [robot_names]
        self.gui = gui
        self.realtime = realtime and gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(SIM_DT, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        self.stations = []
        for i, name in enumerate(robot_names):
            origin = (0.0, i * spacing - (len(robot_names) - 1) * spacing / 2, 0.0)
            st = Station(name, self.cid, origin)
            self.stations.append(st)
            if gui:
                p.addUserDebugText(st.robot.spec["label"], (st.origin + [0.0, 0.0, 1.25]).tolist(),
                                   textColorRGB=[0.1, 0.1, 0.1], textSize=1.3, physicsClientId=self.cid)
        if gui:
            n = len(robot_names)
            yaw = 90 if n > 1 else 50  # several robots: look at them from the front, in a row
            p.resetDebugVisualizerCamera(1.4 + 0.5 * spacing * (n - 1), yaw, -22, [0.3, 0, 0.32],
                                         physicsClientId=self.cid)

    def reset(self, seed):
        """Every station gets the SAME cube/goal layout for a given seed (fair comparison)."""
        for st in self.stations:
            st.reset(np.random.default_rng(seed))
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.cid)
        return [st.observe() for st in self.stations]

    def step_physics(self):
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)
            if self.realtime:
                time.sleep(SIM_DT)
        for st in self.stations:
            st.after_physics()
        return [st.observe() for st in self.stations]

    def render(self, width=480, height=360, target=None, distance=None, yaw=None, pitch=-25):
        """Offscreen RGB frame (works without a display, uses the TinyRenderer)."""
        n = len(self.stations)
        target = target if target is not None else [0.3, 0.0, 0.32]
        if n > 1:
            spread = np.ptp([st.origin[1] for st in self.stations])
            distance = distance if distance is not None else 0.6 + 0.5 * spread
            yaw = 90 if yaw is None else yaw
        else:
            distance = distance if distance is not None else 1.5
            yaw = 50 if yaw is None else yaw
        view = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, 2, physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(50, width / height, 0.05, 20, physicsClientId=self.cid)
        img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                               physicsClientId=self.cid)
        return np.reshape(np.array(img[2], dtype=np.uint8), (height, width, 4))[:, :, :3]

    def close(self):
        p.disconnect(self.cid)

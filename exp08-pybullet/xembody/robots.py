"""Robot embodiments and the embodiment adapter.

Every robot here is driven through the SAME interface:

    robot.command_tcp(target_xyz)   # Cartesian target for the tool centre point (TCP)
    robot.tcp_position()            # where the TCP actually is

Internally each robot uses its own inverse kinematics (IK), its own joint
limits, its own motor forces and its own tool offset. This adapter is what
makes cross-embodiment transfer possible: the policy speaks "task space",
the adapter translates it into each robot's "joint space".
"""

import math

import numpy as np
import pybullet as p
import pybullet_data

# Gripper pointing straight down (tool z-axis along world -z).
DOWN = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

# Registry of embodiments. Add your own robot here (see MANUAL, Task 6).
#   urdf        : path inside pybullet_data (or an absolute path)
#   ee_link     : name of the link whose pose the IK controls
#   tcp_offset  : distance (m) from ee_link origin to the tool tip, along the tool axis
#   home        : joint angles (rad) for the arm joints, in order
#   scale       : globalScaling applied when loading the URDF
#   fingers     : finger joints that are simply held open (Panda only)
EMBODIMENTS = {
    "kuka_iiwa": dict(
        label="KUKA LBR iiwa (7-DOF)",
        urdf="kuka_iiwa/model.urdf",
        ee_link="lbr_iiwa_link_7",
        tcp_offset=0.045,
        home=[0.0, 0.35, 0.0, -1.6, 0.0, 1.2, 0.0],
        scale=1.0,
        fingers=[],
    ),
    "franka_panda": dict(
        label="Franka Emika Panda (7-DOF)",
        urdf="franka_panda/panda.urdf",
        ee_link="panda_grasptarget",
        tcp_offset=0.0,
        home=[0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785],
        scale=1.0,
        fingers=["panda_finger_joint1", "panda_finger_joint2"],
    ),
    "xarm6": dict(
        label="UFactory xArm6 (6-DOF)",
        urdf="xarm/xarm6_robot.urdf",
        ee_link="link6",
        tcp_offset=0.0,
        home=[0.0, -0.3, -0.6, 0.0, 0.9, 0.0],
        scale=1.0,
        fingers=[],
    ),
    "kuka_mini": dict(
        label="KUKA iiwa scaled to 70% (7-DOF, short reach)",
        urdf="kuka_iiwa/model.urdf",
        ee_link="lbr_iiwa_link_7",
        tcp_offset=0.045,
        home=[0.0, 0.35, 0.0, -1.6, 0.0, 1.2, 0.0],
        scale=0.7,
        fingers=[],
    ),
}

DEFAULT_ROBOTS = ["kuka_iiwa", "franka_panda", "xarm6", "kuka_mini"]
MAX_ARM_DOF = 7


class Robot:
    """One robot arm loaded into a PyBullet client, behind a task-space adapter."""

    def __init__(self, name, client, base_pos=(0, 0, 0)):
        if name not in EMBODIMENTS:
            raise KeyError(f"Unknown robot '{name}'. Choose from {list(EMBODIMENTS)}")
        self.name = name
        self.spec = EMBODIMENTS[name]
        self.cid = client
        self.base_pos = np.array(base_pos, dtype=float)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        self.uid = p.loadURDF(
            self.spec["urdf"],
            basePosition=list(base_pos),
            useFixedBase=True,
            globalScaling=self.spec["scale"],
            physicsClientId=client,
        )
        self._read_joints()
        self.tcp_offset = self.spec["tcp_offset"] * self.spec["scale"]
        self.reset()

    # ------------------------------------------------------------------ setup
    def _read_joints(self):
        """Collect movable joints, limits, forces and the end-effector link index."""
        self.movable = []          # every non-fixed joint, in PyBullet order (IK output order)
        self.lower, self.upper, self.max_force, self.max_vel = [], [], [], []
        self.ee_index = None
        finger_names = set(self.spec["fingers"])
        self.finger_joints = []
        for j in range(p.getNumJoints(self.uid, physicsClientId=self.cid)):
            info = p.getJointInfo(self.uid, j, physicsClientId=self.cid)
            jname, jtype, link_name = info[1].decode(), info[2], info[12].decode()
            if link_name == self.spec["ee_link"]:
                self.ee_index = j
            if jtype == p.JOINT_FIXED:
                continue
            self.movable.append(j)
            lo, hi = info[8], info[9]
            if hi <= lo:  # continuous joint: treat as +/- 2 pi
                lo, hi = -2 * math.pi, 2 * math.pi
            self.lower.append(lo)
            self.upper.append(hi)
            self.max_force.append(info[10] if info[10] > 0 else 200.0)
            self.max_vel.append(info[11] if info[11] > 0 else 2.0)
            if jname in finger_names:
                self.finger_joints.append(j)
        if self.ee_index is None:
            raise RuntimeError(f"{self.name}: end-effector link '{self.spec['ee_link']}' not found")
        self.arm_joints = [j for j in self.movable if j not in self.finger_joints]
        self.dof = len(self.arm_joints)
        self.lower, self.upper = np.array(self.lower), np.array(self.upper)
        self.ranges = self.upper - self.lower
        self.rest = np.zeros(len(self.movable))
        for k, j in enumerate(self.movable):
            if j in self.arm_joints:
                self.rest[k] = self.spec["home"][self.arm_joints.index(j)]
            else:
                self.rest[k] = 0.04  # open fingers

    def reset(self):
        """Teleport to the home configuration."""
        for k, j in enumerate(self.movable):
            p.resetJointState(self.uid, j, self.rest[k], physicsClientId=self.cid)
        self.hold_joints(self.rest)
        self.target = self.tcp_position()

    # ------------------------------------------------------------------ sensing
    def ee_pose(self):
        st = p.getLinkState(self.uid, self.ee_index, computeForwardKinematics=True,
                            physicsClientId=self.cid)
        return np.array(st[4]), np.array(st[5])

    def tcp_position(self):
        """Tool-tip position in the robot's own base frame (base_pos subtracted)."""
        pos, orn = self.ee_pose()
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos + rot[:, 2] * self.tcp_offset - self.base_pos

    def joint_positions(self):
        """Arm joint angles (rad), length = self.dof."""
        return np.array([p.getJointState(self.uid, j, physicsClientId=self.cid)[0]
                         for j in self.arm_joints])

    def padded_joints(self):
        """Arm joint angles zero-padded to MAX_ARM_DOF, for embodiment-specific observations."""
        q = np.zeros(MAX_ARM_DOF)
        q[: self.dof] = self.joint_positions()[:MAX_ARM_DOF]
        return q

    # ------------------------------------------------------------------ acting
    def hold_joints(self, q_movable):
        """Send position targets for all movable joints (in self.movable order)."""
        for k, j in enumerate(self.movable):
            p.setJointMotorControl2(
                self.uid, j, p.POSITION_CONTROL, targetPosition=float(q_movable[k]),
                force=self.max_force[k], maxVelocity=self.max_vel[k],
                positionGain=0.1, velocityGain=1.0, physicsClientId=self.cid)

    def command_joints(self, q_arm):
        """JOINT-SPACE command: arm joint targets (rad). Clipped to this robot's limits."""
        q = self.rest.copy()
        for i, j in enumerate(self.arm_joints):
            k = self.movable.index(j)
            if i < len(q_arm):
                q[k] = np.clip(q_arm[i], self.lower[k], self.upper[k])
        self.hold_joints(q)
        return q

    def command_tcp(self, target):
        """TASK-SPACE command: move the tool tip to `target` (base frame, metres)."""
        self.target = np.asarray(target, dtype=float)
        link_target = self.target + self.base_pos + np.array([0.0, 0.0, self.tcp_offset])
        q = p.calculateInverseKinematics(
            self.uid, self.ee_index, link_target.tolist(), DOWN,
            lowerLimits=self.lower.tolist(), upperLimits=self.upper.tolist(),
            jointRanges=self.ranges.tolist(), restPoses=self.rest.tolist(),
            maxNumIterations=100, residualThreshold=1e-4, physicsClientId=self.cid)
        q = np.clip(np.array(q), self.lower, self.upper)
        for k, j in enumerate(self.movable):
            if j in self.finger_joints:
                q[k] = 0.04
        self.hold_joints(q)
        return q

    def describe(self):
        return dict(name=self.name, label=self.spec["label"], dof=self.dof,
                    n_links=p.getNumJoints(self.uid, physicsClientId=self.cid),
                    ee_link=self.spec["ee_link"], scale=self.spec["scale"])

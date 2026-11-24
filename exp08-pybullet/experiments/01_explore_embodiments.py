"""Experiment 1: how different are the bodies?

For every robot we measure
  * the number of arm joints (DOF) and their limits,
  * the reachable workspace (random joint samples -> tool-tip positions),
  * task coverage: the share of task-region points the tool can reach while
    pointing straight down, within 1 cm (checked with the robot's own IK).

    python experiments/01_explore_embodiments.py
    python experiments/01_explore_embodiments.py --robots kuka_iiwa xarm6 --samples 6000
"""

import argparse

import _common
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from xembody import DEFAULT_ROBOTS, World
from xembody.robots import DOWN
from xembody.rollout import print_table, results_path, write_csv
from xembody.world import CUBE_HALF


def sample_workspace(robot, n, rng):
    pts = []
    for _ in range(n):
        q = rng.uniform(robot.lower, robot.upper)
        for k, j in enumerate(robot.movable):
            p.resetJointState(robot.uid, j, q[k], physicsClientId=robot.cid)
        pts.append(robot.tcp_position())
    robot.reset()
    return np.array(pts)


def ik_reachable(robot, target, tol=0.01):
    """Solve IK by repeated projection (teleporting joints), then check the real error."""
    robot.reset()
    link_target = np.asarray(target) + robot.base_pos + [0, 0, robot.tcp_offset]
    for _ in range(8):
        q = p.calculateInverseKinematics(
            robot.uid, robot.ee_index, link_target.tolist(), DOWN,
            lowerLimits=robot.lower.tolist(), upperLimits=robot.upper.tolist(),
            jointRanges=robot.ranges.tolist(), restPoses=robot.rest.tolist(),
            maxNumIterations=200, residualThreshold=1e-5, physicsClientId=robot.cid)
        q = np.clip(q, robot.lower, robot.upper)
        for k, j in enumerate(robot.movable):
            p.resetJointState(robot.uid, j, q[k], physicsClientId=robot.cid)
    err = np.linalg.norm(robot.tcp_position() - target)
    robot.reset()
    return err < tol


def task_grid():
    xs, ys = np.linspace(0.42, 0.60, 5), np.linspace(-0.25, 0.25, 9)
    zs = [2 * CUBE_HALF + 0.005, 0.15]  # grasp height and travel height
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=DEFAULT_ROBOTS)
    ap.add_argument("--samples", type=int, default=3000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    grid = task_grid()

    rows, clouds = [], {}
    for name in args.robots:
        world = World(name)
        rb = world.stations[0].robot
        cloud = sample_workspace(rb, args.samples, rng)
        clouds[name] = cloud
        ok = np.array([ik_reachable(rb, t) for t in grid])
        reach = np.linalg.norm(cloud[:, :2], axis=1).max()
        rows.append(dict(robot=name, label=rb.spec["label"], dof=rb.dof,
                         joint_range_deg=float(np.degrees(rb.ranges[:rb.dof]).mean()),
                         max_reach_m=float(reach), task_coverage=float(ok.mean())))
        world.close()

    cols = ["robot", "dof", "joint_range_deg", "max_reach_m", "task_coverage"]
    print_table(rows, cols, "Embodiment summary (task_coverage = share of task points reachable, gripper down)")
    write_csv(results_path("01_embodiments.csv"), rows, ["robot", "label", *cols[1:]])

    n = len(clouds)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7), squeeze=False)
    for col, (name, cloud) in enumerate(clouds.items()):
        c = _common.ROBOT_COLORS.get(name, None)
        top, side = axes[0, col], axes[1, col]
        top.scatter(cloud[:, 0], cloud[:, 1], s=2, alpha=0.3, color=c)
        side.scatter(np.linalg.norm(cloud[:, :2], axis=1), cloud[:, 2], s=2, alpha=0.3, color=c)
        top.add_patch(plt.Rectangle((0.42, -0.25), 0.18, 0.5, fill=False, lw=2, ec="green"))
        side.add_patch(plt.Rectangle((0.42, 0.0), 0.24, 0.15, fill=False, lw=2, ec="green"))
        cov = next(r["task_coverage"] for r in rows if r["robot"] == name)
        top.set_title(f"{name}\ntask coverage {cov:.0%}", fontsize=10)
        top.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="x (m)")
        side.set(xlim=(0, 1), ylim=(-0.4, 1.35), xlabel="horizontal distance r (m)")
        top.set_aspect("equal")
        for ax in (top, side):
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("top view: y (m)")
    axes[1, 0].set_ylabel("side view: z (m)")
    fig.suptitle("Reachable tool positions from random joint angles (green box = task area)")
    fig.tight_layout()
    fig.savefig(results_path("01_workspace.png"), dpi=130)

    # one picture of each robot, same camera
    fig, axes = plt.subplots(1, len(args.robots), figsize=(4 * len(args.robots), 3.6))
    for ax, name in zip(np.atleast_1d(axes), args.robots):
        world = World(name)
        world.reset(seed=0)
        ax.imshow(world.render(400, 340, target=[0.25, 0, 0.35], distance=1.7, yaw=45, pitch=-20))
        ax.set_title(name)
        ax.axis("off")
        world.close()
    fig.tight_layout()
    fig.savefig(results_path("01_robots.png"), dpi=130)
    print("\nSaved results/01_embodiments.csv, results/01_workspace.png, results/01_robots.png")


if __name__ == "__main__":
    main()

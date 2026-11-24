"""Experiment 2: naive transfer in JOINT space (the baseline that fails).

1. The expert solves the task on the SOURCE robot. We record the joint angles
   at every control step (plus the suction command).
2. We replay exactly those joint angles on every TARGET robot.
   Extra joints keep their home angle, missing joints are dropped, and angles
   outside a robot's limits are clipped (a real controller would do the same).

    python experiments/02_joint_space_transfer.py
    python experiments/02_joint_space_transfer.py --source franka_panda --episodes 5
    python experiments/02_joint_space_transfer.py --gui        # watch the replay
"""

import argparse

import _common
import matplotlib.pyplot as plt
import numpy as np

from xembody import DEFAULT_ROBOTS, World, expert_action
from xembody.rollout import print_table, results_path, write_csv


def record_source(source, seed):
    world = World(source)
    st = world.stations[0]
    obs = world.reset(seed)[0]
    q_log, grip_log, path = [], [], [obs[:3]]
    while not st.done:
        a = expert_action(obs)
        st.apply_action(a)
        obs = world.step_physics()[0]
        q_log.append(st.robot.joint_positions())
        grip_log.append(a[3])
        path.append(obs[:3])
    ok = st.success
    world.close()
    return np.array(q_log), np.array(grip_log), np.array(path), ok


def replay(target, seed, q_log, grip_log, gui=False):
    world = World(target, gui=gui, realtime=gui)
    st = world.stations[0]
    obs = world.reset(seed)[0]
    path = [obs[:3]]
    for q, g in zip(q_log, grip_log):
        st.apply_joint_command(q, g)
        obs = world.step_physics()[0]
        path.append(obs[:3])
        if st.success:
            break
    for _ in range(30):  # let the cube settle after the last command
        st.apply_joint_command(q_log[-1], -1.0)
        world.step_physics()
    ok, err = st.success, float(np.linalg.norm(st.cube_position()[:2] - st.goal[:2]))
    world.close()
    return np.array(path), ok, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="kuka_iiwa")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_ROBOTS)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--gui", action="store_true", help="show the replay of the first episode")
    args = ap.parse_args()

    per_robot = {t: [] for t in args.targets}
    example = None
    for ep in range(args.episodes):
        seed = 2000 + ep
        q_log, g_log, src_path, src_ok = record_source(args.source, seed)
        if example is None:
            example = dict(seed=seed, src=src_path, paths={})
        for t in args.targets:
            path, ok, err = replay(t, seed, q_log, g_log, gui=args.gui and ep == 0)
            n = min(len(path), len(src_path))
            dev = np.linalg.norm(path[:n] - src_path[:n], axis=1)
            per_robot[t].append(dict(success=int(ok), dev_mean=dev.mean(), dev_max=dev.max(), err=err))
            if ep == 0:
                example["paths"][t] = path

    rows = []
    for t, eps in per_robot.items():
        rows.append(dict(robot=t, source=args.source, episodes=len(eps),
                         success_rate=float(np.mean([e["success"] for e in eps])),
                         path_dev_mean_cm=100 * float(np.mean([e["dev_mean"] for e in eps])),
                         path_dev_max_cm=100 * float(np.mean([e["dev_max"] for e in eps])),
                         cube_goal_err_cm=100 * float(np.median([e["err"] for e in eps]))))
    cols = ["robot", "success_rate", "path_dev_mean_cm", "path_dev_max_cm", "cube_goal_err_cm"]
    print_table(rows, cols, f"JOINT-SPACE replay of {args.source} trajectories ({args.episodes} episodes)")
    write_csv(results_path("02_joint_space_transfer.csv"), rows, ["robot", "source", "episodes", *cols[1:]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(example["src"][:, 0], example["src"][:, 1], "k--", lw=2.5, label=f"{args.source} (source)")
    axes[1].plot(example["src"][:, 1], example["src"][:, 2], "k--", lw=2.5)
    for t, path in example["paths"].items():
        c = _common.ROBOT_COLORS.get(t)
        axes[0].plot(path[:, 0], path[:, 1], color=c, lw=1.5, label=f"{t} (replay)")
        axes[1].plot(path[:, 1], path[:, 2], color=c, lw=1.5)
    axes[0].set(title="Tool-tip path, top view", xlabel="x (m)", ylabel="y (m)")
    axes[1].set(title="Tool-tip path, front view", xlabel="y (m)", ylabel="z (m)")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Same joint angles, different bodies -> different tool paths")
    fig.tight_layout()
    fig.savefig(results_path("02_joint_space_paths.png"), dpi=130)
    print("\nSaved results/02_joint_space_transfer.csv and results/02_joint_space_paths.png")


if __name__ == "__main__":
    main()

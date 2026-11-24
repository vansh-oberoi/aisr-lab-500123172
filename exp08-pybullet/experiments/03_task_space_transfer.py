"""Experiment 3: transfer through a shared TASK-SPACE interface.

The same expert (which only sees tool, cube and goal positions and only
outputs tool displacements + suction) drives every robot. Each robot's own
IK adapter turns the displacement into joint targets.

    python experiments/03_task_space_transfer.py
    python experiments/03_task_space_transfer.py --episodes 50
    python experiments/03_task_space_transfer.py --gui --robots xarm6   # watch one robot
"""

import argparse

import _common
import matplotlib.pyplot as plt
import numpy as np

from xembody import DEFAULT_ROBOTS, World, expert_action
from xembody.rollout import evaluate, print_table, results_path, run_episode, summarise, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=DEFAULT_ROBOTS)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    policy = lambda obs, joints: expert_action(obs)  # noqa: E731  (joints are ignored)
    seeds = [3000 + i for i in range(args.episodes)]
    rows = []
    for name in args.robots:
        s = summarise(evaluate(name, policy, seeds, gui=args.gui))
        rows.append(dict(robot=name, **s))
    cols = ["robot", "success_rate", "mean_steps", "median_error_cm"]
    print_table(rows, cols, f"TASK-SPACE transfer of the same expert ({args.episodes} episodes per robot)")
    write_csv(results_path("03_task_space_transfer.csv"), rows, ["robot", "n", *cols[1:]])

    # all robots in one world, same seed, for the path figure
    world = World(args.robots)
    eps = run_episode(world, policy, seeds[0])
    world.close()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for e in eps:
        c, path = _common.ROBOT_COLORS.get(e["robot"]), e["path"]
        tag = "ok" if e["success"] else "fail"
        axes[0].plot(path[:, 0], path[:, 1], color=c, lw=1.6, label=f"{e['robot']} ({tag})")
        axes[1].plot(path[:, 1], path[:, 2], color=c, lw=1.6)
    axes[0].set(title="Tool-tip path, top view", xlabel="x (m)", ylabel="y (m)")
    axes[1].set(title="Tool-tip path, front view", xlabel="y (m)", ylabel="z (m)")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Same task-space commands, different bodies -> (almost) the same tool paths")
    fig.tight_layout()
    fig.savefig(results_path("03_task_space_paths.png"), dpi=130)
    print("\nSaved results/03_task_space_transfer.csv and results/03_task_space_paths.png")


if __name__ == "__main__":
    main()

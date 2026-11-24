"""Experiment 4: learn on ONE robot, deploy on MANY (zero-shot).

1. Collect expert demonstrations on the source robot only.
2. Train two behaviour-cloning (BC) policies on the same data:
      BC-agnostic : sees only task-space features
      BC-proprio  : also sees the source robot's joint angles
3. Deploy both, unchanged, on every robot.

    python experiments/04_learn_once_deploy_many.py
    python experiments/04_learn_once_deploy_many.py --source xarm6 --demos 100
"""

import argparse
import time

import _common
import matplotlib.pyplot as plt
import numpy as np

from xembody import DEFAULT_ROBOTS, BCPolicy, collect_demos, expert_action
from xembody.rollout import evaluate, print_table, results_path, summarise, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="kuka_iiwa")
    ap.add_argument("--robots", nargs="+", default=DEFAULT_ROBOTS)
    ap.add_argument("--demos", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    print(f"Collecting {args.demos} noisy-expert demonstrations on {args.source} ...")
    t0 = time.time()
    data = collect_demos([args.source], args.demos)
    print(f"  {len(data['agnostic'][0])} (observation, action) pairs in {time.time() - t0:.0f} s")

    policies = {"expert": lambda o, q: expert_action(o)}
    for mode in ("agnostic", "proprio"):
        X, Y = data[mode]
        t0 = time.time()
        pol = BCPolicy(mode).fit(X, Y)
        pol.save(results_path("models", f"bc_{mode}_{args.source}.pkl"))
        print(f"  trained BC-{mode:8s} on {X.shape[1]} features in {time.time() - t0:.0f} s")
        policies[f"BC-{mode}"] = pol.act

    seeds = [4000 + i for i in range(args.episodes)]
    rows = []
    for name in args.robots:
        for pname, pol in policies.items():
            s = summarise(evaluate(name, pol, seeds))
            rows.append(dict(robot=name, policy=pname, trained_on=args.source, **s))
            print(f"  {name:13s} {pname:12s} success {s['success_rate']:.2f}")
    cols = ["robot", "policy", "success_rate", "mean_steps", "median_error_cm"]
    print_table(rows, cols, f"Learn on {args.source}, deploy everywhere ({args.episodes} episodes each)")
    write_csv(results_path(f"04_learn_once_deploy_many_{args.source}.csv"), rows,
              ["robot", "policy", "trained_on", "n", *cols[2:]])

    pnames = list(policies)
    x = np.arange(len(args.robots))
    w = 0.8 / len(pnames)
    shades = ["#bbbbbb", "#2b7bba", "#d95f02"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, pn in enumerate(pnames):
        vals = [next(r["success_rate"] for r in rows if r["robot"] == rb and r["policy"] == pn)
                for rb in args.robots]
        bars = ax.bar(x + (i - (len(pnames) - 1) / 2) * w, vals, w, label=pn, color=shades[i % 3])
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, padding=2)
    ax.set_xticks(x, [f"{r}\n(source)" if r == args.source else r for r in args.robots])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("success rate")
    ax.set_title(f"Policies trained only on {args.source}, evaluated zero-shot on every robot")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_path(f"04_learn_once_deploy_many_{args.source}.png"), dpi=130)
    print(f"\nSaved results/04_learn_once_deploy_many_{args.source}.csv/.png and results/models/")


if __name__ == "__main__":
    main()

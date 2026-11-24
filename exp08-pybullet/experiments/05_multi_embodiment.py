"""Experiment 5: does training on SEVERAL bodies help a NEW body?

Leave-one-embodiment-out: for each held-out robot, train BC on the other
robots (same total number of demonstrations as a single-source policy) and
test on the held-out robot it has never seen.

    python experiments/05_multi_embodiment.py
    python experiments/05_multi_embodiment.py --pool kuka_iiwa franka_panda xarm6 --demos 60
"""

import argparse

import _common
import matplotlib.pyplot as plt
import numpy as np

from xembody import BCPolicy, collect_demos
from xembody.rollout import evaluate, print_table, results_path, summarise, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", default=["kuka_iiwa", "franka_panda", "xarm6"])
    ap.add_argument("--demos", type=int, default=60, help="total demonstrations per policy")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    seeds = [5000 + i for i in range(args.episodes)]

    rows, trained = [], {}  # the same training set appears for several held-out robots: train it once
    for held_out in args.pool:
        others = [r for r in args.pool if r != held_out]
        setups = {f"single ({others[0]})": [others[0]], f"multi ({' + '.join(others)})": others}
        for setup, train_on in setups.items():
            key = tuple(train_on)
            if key not in trained:
                data = collect_demos(train_on, args.demos // len(train_on), log=lambda *_: None)
                trained[key] = {m: BCPolicy(m).fit(*data[m]) for m in ("agnostic", "proprio")}
            for mode in ("agnostic", "proprio"):
                pol = trained[key][mode]
                s = summarise(evaluate(held_out, pol.act, seeds))
                rows.append(dict(held_out=held_out, training=setup.split(" ")[0], trained_on="+".join(train_on),
                                 obs=mode, **s))
                print(f"  test {held_out:13s} | {setup:40s} | {mode:8s} success {s['success_rate']:.2f}")

    cols = ["held_out", "training", "trained_on", "obs", "success_rate", "median_error_cm"]
    print_table(rows, cols, "Leave-one-embodiment-out")
    write_csv(results_path("05_multi_embodiment.csv"), rows, [*cols, "n", "mean_steps"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, mode in zip(axes, ("agnostic", "proprio")):
        x = np.arange(len(args.pool))
        for i, (tr, col) in enumerate([("single", "#bbbbbb"), ("multi", "#2b7bba")]):
            vals = [next(r["success_rate"] for r in rows
                         if r["held_out"] == h and r["training"] == tr and r["obs"] == mode) for h in args.pool]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=col, label=f"{tr}-robot training")
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, padding=2)
        ax.set_xticks(x, [f"unseen:\n{h}" for h in args.pool])
        ax.set_title(f"BC-{mode}")
        ax.set_ylim(0, 1.25)
        if all(r["success_rate"] == 0 for r in rows if r["obs"] == mode):
            ax.text(1, 0.5, "0% on every unseen robot", ha="center", fontsize=11, color="#555555")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("success on the unseen robot")
    axes[0].legend(fontsize=8, loc="upper center", ncol=2)
    fig.suptitle(f"Same budget ({args.demos} demos): one source robot vs several")
    fig.tight_layout()
    fig.savefig(results_path("05_multi_embodiment.png"), dpi=130)
    print("\nSaved results/05_multi_embodiment.csv and results/05_multi_embodiment.png")


if __name__ == "__main__":
    main()

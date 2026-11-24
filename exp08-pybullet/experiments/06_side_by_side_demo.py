"""Demo: one policy, four bodies, one world (good for the projector).

All robots get the same cube/goal layout and the same policy at the same time.

    python experiments/06_side_by_side_demo.py --gui                       # expert policy, live window
    python experiments/06_side_by_side_demo.py --gui --policy results/models/bc_agnostic_kuka_iiwa.pkl
    python experiments/06_side_by_side_demo.py --gif                       # save results/06_side_by_side.gif
"""

import argparse

import _common  # noqa: F401
import numpy as np

from xembody import DEFAULT_ROBOTS, BCPolicy, World, expert_action
from xembody.rollout import results_path, run_episode, save_gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=DEFAULT_ROBOTS)
    ap.add_argument("--policy", default="expert", help="'expert' or a path to a saved BC .pkl")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    if args.policy == "expert":
        policy = lambda o, q: expert_action(o)  # noqa: E731
    else:
        policy = BCPolicy.load(args.policy).act
    world = World(args.robots, gui=args.gui, realtime=args.gui, spacing=1.3)
    frames = [] if args.gif else None
    for ep in range(args.episodes):
        res = run_episode(world, policy, seed=6000 + ep, record_every=2 if args.gif else 0, frames=frames)
        print(f"episode {ep}: " + ", ".join(f"{r['robot']}={'OK' if r['success'] else 'fail'}" for r in res))
    if args.gif:
        out = results_path("06_side_by_side.gif")
        save_gif([np.ascontiguousarray(f) for f in frames], out, fps=12)
        print(f"Saved {out}")
    world.close()


if __name__ == "__main__":
    main()

"""Running episodes, scoring them and saving results."""

import csv
import os

import numpy as np

from .world import GOAL_TOL, World

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def results_path(*parts):
    path = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def run_episode(world, policy, seed, record_every=0, frames=None):
    """Run one episode on every station of `world` with the same policy.

    policy(obs, joints) -> action (4,). Returns one metrics dict per station,
    each holding the full tool-tip path for plotting.
    """
    obs = world.reset(seed)
    paths = [[o[:3].copy()] for o in obs]
    k = 0
    while not all(st.done for st in world.stations):
        for st, o in zip(world.stations, obs):
            if not st.done:
                st.apply_action(policy(o, st.robot.padded_joints()))
        obs = world.step_physics()
        for path, o in zip(paths, obs):
            path.append(o[:3].copy())
        if record_every and frames is not None and k % record_every == 0:
            frames.append(world.render(720, 320))
        k += 1
    out = []
    for st, path in zip(world.stations, paths):
        cube = st.cube_position()
        out.append(dict(robot=st.robot.name, seed=seed, success=int(st.success), steps=st.steps,
                        final_error=float(np.linalg.norm(cube[:2] - st.goal[:2])),
                        path=np.array(path)))
    return out


def evaluate(robot_name, policy, seeds, gui=False):
    world = World(robot_name, gui=gui, realtime=gui)
    rows = [run_episode(world, policy, s)[0] for s in seeds]
    world.close()
    return rows


def summarise(rows):
    succ = np.array([r["success"] for r in rows])
    steps = np.array([r["steps"] for r in rows if r["success"]])
    err = np.array([r["final_error"] for r in rows])
    return dict(success_rate=succ.mean(), n=len(rows),
                mean_steps=float(steps.mean()) if len(steps) else float("nan"),
                median_error_cm=100 * float(np.median(err)))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(rows, cols, title=None):
    if title:
        print("\n" + title)
    widths = [max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(_fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def save_gif(frames, path, fps=15):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


__all__ = ["GOAL_TOL", "results_path", "run_episode", "evaluate", "summarise", "write_csv",
           "print_table", "save_gif"]

"""Behaviour cloning (BC): learn a policy from expert demonstrations.

Two observation designs are compared:

  "agnostic" : only task-space quantities (tool tip, cube, goal, holding flag)
               -> the same input means the same thing on every robot
  "proprio"  : the agnostic features PLUS the robot's own joint angles
               (zero-padded to 7). Joint angles mean different things on
               different robots, so this input carries an embodiment gap.
"""

import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .expert import expert_action
from .world import World

OBS_MODES = ("agnostic", "proprio")


def features(obs, joints, mode):
    tcp, holding, cube, goal = obs[0:3], obs[3:4], obs[4:7], obs[7:10]
    f = np.concatenate([cube - tcp, goal - cube, goal - tcp, tcp, cube[2:3], holding])
    if mode == "proprio":
        f = np.concatenate([f, joints])
    return f


class BCPolicy:
    def __init__(self, mode="agnostic", hidden=(256, 256), seed=0):
        assert mode in OBS_MODES
        self.mode = mode
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden, activation="relu", learning_rate_init=1e-3,
                         max_iter=400, early_stopping=True, n_iter_no_change=10, random_state=seed))

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def act(self, obs, joints):
        x = features(obs, joints, self.mode)[None, :]
        return np.clip(self.model.predict(x)[0], -1, 1)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def collect_demos(robot_names, n_episodes, seed0=1000, noise=0.3, log=print):
    """Run the expert on each robot. Actions are executed with noise (DART-style) so the
    data covers recovery from small mistakes, but the LABEL is always the clean expert action."""
    data = {m: ([], []) for m in OBS_MODES}
    rng = np.random.default_rng(seed0)
    for name in robot_names:
        world = World(name)
        st = world.stations[0]
        ok = 0
        for ep in range(n_episodes):
            obs = world.reset(seed0 + ep)[0]
            while not st.done:
                a_star = expert_action(obs)
                q = st.robot.padded_joints()
                for m in OBS_MODES:
                    data[m][0].append(features(obs, q, m))
                    data[m][1].append(a_star)
                a = a_star.copy()
                a[:3] += rng.normal(0, noise, 3)
                st.apply_action(a)
                obs = world.step_physics()[0]
            ok += st.success
        log(f"  demos on {name:13s}: {n_episodes} episodes, expert success {ok}/{n_episodes}")
        world.close()
    return {m: (np.array(X), np.array(Y)) for m, (X, Y) in data.items()}

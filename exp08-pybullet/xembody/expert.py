"""A scripted expert that works ONLY with the shared observation.

It never looks at joint angles, link lengths or the robot model, so the same
expert can drive any robot that has a task-space adapter. It is also the
"teacher" whose demonstrations the learned policy imitates.
"""

import numpy as np

from .world import CUBE_HALF, MAX_STEP

HOVER = 0.15        # travel height (m)
GRASP_Z = 2 * CUBE_HALF + 0.005   # tool-tip height when touching the cube top


def _toward(tcp, target):
    """Proportional controller in task space, saturated to the action range."""
    return np.clip((np.asarray(target) - tcp) / MAX_STEP, -1.0, 1.0)


def expert_action(obs):
    tcp, holding, cube, goal = obs[0:3], obs[3] > 0.5, obs[4:7], obs[7:10]
    if not holding:
        cube_on_goal = np.linalg.norm(cube[:2] - goal[:2]) < 0.03 and cube[2] < CUBE_HALF + 0.01
        if cube_on_goal:                       # task finished: back off upwards
            return np.r_[_toward(tcp, [tcp[0], tcp[1], HOVER]), -1.0]
        xy_err = np.linalg.norm(tcp[:2] - cube[:2])
        if xy_err > 0.05 and tcp[2] < HOVER - 0.04:   # too low to travel: rise first
            return np.r_[_toward(tcp, [tcp[0], tcp[1], HOVER]), -1.0]
        if xy_err > 0.012:                     # travel to above the cube
            return np.r_[_toward(tcp, [cube[0], cube[1], HOVER]), -1.0]
        top = cube[2] + CUBE_HALF + 0.005      # descend onto the cube, switch suction on at contact
        grip = 1.0 if tcp[2] - top < 0.015 else -1.0
        return np.r_[_toward(tcp, [cube[0], cube[1], top]), grip]
    xy_err = np.linalg.norm(cube[:2] - goal[:2])
    if xy_err > 0.05 and tcp[2] < HOVER - 0.04:       # lift before carrying
        return np.r_[_toward(tcp, [tcp[0], tcp[1], HOVER]), 1.0]
    if xy_err > 0.012:                         # carry above the goal
        return np.r_[_toward(tcp, [goal[0], goal[1], HOVER]), 1.0]
    grip = -1.0 if tcp[2] < GRASP_Z + 0.012 else 1.0   # lower, then release
    return np.r_[_toward(tcp, [goal[0], goal[1], GRASP_Z]), grip]

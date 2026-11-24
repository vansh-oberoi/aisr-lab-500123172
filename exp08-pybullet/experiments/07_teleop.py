"""Bonus: YOU are the policy. Drive any robot with the keyboard (GUI only).

Click inside the PyBullet window first so it receives the keys.

    Arrow keys : move the tool in x / y
    z / x      : tool up / down
    space      : suction on / off
    n          : new episode (new cube and goal)

    python experiments/07_teleop.py --robot xarm6
    python experiments/07_teleop.py --robot franka_panda kuka_iiwa   # drive two bodies at once
"""

import argparse
import time

import _common  # noqa: F401
import numpy as np
import pybullet as p

from xembody import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", nargs="+", default=["kuka_iiwa"])
    args = ap.parse_args()

    world = World(args.robot, gui=True, realtime=True, spacing=1.3)
    seed = 0
    world.reset(seed)
    grip = -1.0
    print(__doc__)
    try:
        while p.isConnected(world.cid):
            keys = p.getKeyboardEvents(physicsClientId=world.cid)
            a = np.zeros(4)
            held = lambda k: k in keys and keys[k] & p.KEY_IS_DOWN  # noqa: E731
            hit = lambda k: k in keys and keys[k] & p.KEY_WAS_TRIGGERED  # noqa: E731
            if held(p.B3G_UP_ARROW):
                a[0] = 1
            if held(p.B3G_DOWN_ARROW):
                a[0] = -1
            if held(p.B3G_LEFT_ARROW):
                a[1] = 1
            if held(p.B3G_RIGHT_ARROW):
                a[1] = -1
            if held(ord("z")):
                a[2] = 1
            if held(ord("x")):
                a[2] = -1
            if hit(ord(" ")):
                grip = -grip
                print("suction", "ON" if grip > 0 else "OFF")
            if hit(ord("n")):
                seed += 1
                world.reset(seed)
                grip = -1.0
            a[3] = grip
            for st in world.stations:
                st.apply_action(a * 0.5)
                st.done = False  # teleop never times out
            world.step_physics()
            for st in world.stations:
                if st.success:
                    print(f"{st.robot.name}: cube placed on the goal!")
                    st.success = False
            time.sleep(0.005)
    except p.error:
        pass  # window closed


if __name__ == "__main__":
    main()

"""Step 0: check that the installation works.

    python experiments/00_check_setup.py          # headless check
    python experiments/00_check_setup.py --gui    # also opens the PyBullet window for 5 s
"""

import argparse
import platform
import time

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import sklearn

from xembody import DEFAULT_ROBOTS, World
from xembody.rollout import results_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    print(f"Python      {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"PyBullet    API {p.getAPIVersion()}   data: {pybullet_data.getDataPath()}")
    print(f"NumPy       {np.__version__}")
    print(f"scikit-learn {sklearn.__version__}")

    world = World(DEFAULT_ROBOTS, gui=args.gui)
    world.reset(seed=0)
    print("\nRobots loaded:")
    for st in world.stations:
        d = st.robot.describe()
        print(f"  {d['name']:13s} {d['dof']} arm joints, end-effector link '{d['ee_link']}'")
    img = world.render(960, 420)
    path = results_path("00_setup_snapshot.png")
    plt.imsave(path, img)
    print(f"\nSnapshot saved to {path}")
    if args.gui:
        print("GUI open for 5 seconds ...")
        time.sleep(5)
    world.close()
    print("\nSETUP OK")


if __name__ == "__main__":
    main()

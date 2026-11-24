"""Shared helpers for the experiment scripts (makes `xembody` importable from anywhere)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # figures are saved to results/, never shown in a window

ROBOT_COLORS = {"kuka_iiwa": "#e07b00", "franka_panda": "#4a4a4a", "xarm6": "#1f6fb5", "kuka_mini": "#b04fa0"}

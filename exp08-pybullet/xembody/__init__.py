"""xembody: a small PyBullet testbed for cross-embodiment transfer (AISR lab)."""

from .robots import DEFAULT_ROBOTS, EMBODIMENTS, Robot
from .world import MAX_STEPS, Station, World
from .expert import expert_action
from .policy import BCPolicy, collect_demos, features

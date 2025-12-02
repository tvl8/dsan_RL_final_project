# frog_env_registry.py

import os, sys

# Ensure parent directory (which contains frog_fly_env.py) is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ray.tune.registry import register_env
from frog_fly_env import FrogFly3DEnv
from obs_history_wrapper import ObsHistoryWrapper

def make_frog_hist_env(env_config):
    return ObsHistoryWrapper(FrogFly3DEnv(env_config), K=24)

register_env("frog_hist_env", make_frog_hist_env)

print("frog_hist_env successfully registered")

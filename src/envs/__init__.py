from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
import marlenv
from lle import LLE

# from .grf import Academy_3_vs_1_with_Keeper, Academy_Pass_and_Shoot_with_Keeper, Academy_Run_Pass_and_Shoot_with_Keeper, Academy_Corner
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


def lle_fn(**kwargs):
    level = int(kwargs.get("level", 6))
    obs_type = kwargs.get("obs_type", "layered")
    state_type = kwargs.get("state_type", "flattened")
    # env = LLE.level(level).obs_type(obs_type).state_type(state_type).single_objective()
    env = LLE.from_file("map.toml").obs_type(obs_type).state_type(state_type).single_objective()
    env = marlenv.adapters.PymarlAdapter(env, 78)
    return env


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "lle": lle_fn,
    # "academy_3_vs_1_with_keeper": partial(env_fn, env=Academy_3_vs_1_with_Keeper),
    # "academy_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper),
    # "academy_run_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Run_Pass_and_Shoot_with_Keeper),
    # "academy_corner": partial(env_fn, env=Academy_Corner),
}

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

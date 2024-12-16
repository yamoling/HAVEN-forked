from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
import marlenv
from lle import LLE

# from .grf import Academy_3_vs_1_with_Keeper, Academy_Pass_and_Shoot_with_Keeper, Academy_Run_Pass_and_Shoot_with_Keeper, Academy_Corner


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


def lle_fn(**kwargs):
    map_file: str = kwargs["map"]
    if map_file.isnumeric():
        env = LLE.level(int(map_file))
    else:
        env = LLE.from_file(map_file)
    obs_type = kwargs.get("obs_type", "layered")
    state_type = kwargs.get("state_type", "flattened")
    env = env.obs_type(obs_type).state_type(state_type).single_objective()
    seed = kwargs.get("seed", None)
    if seed is not None:
        seed = int(seed)
        env.seed(seed)
    time_limit = env.width * env.height // 2
    return marlenv.adapters.PymarlAdapter(env, time_limit)


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

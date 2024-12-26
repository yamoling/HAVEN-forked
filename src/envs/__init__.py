from functools import partial
from typing import Literal
from .multiagentenv import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
import marlenv
from lle import LLE
from lle.tiles import Direction
from .shaping import LLEPotentialShaping

# from .grf import Academy_3_vs_1_with_Keeper, Academy_Pass_and_Shoot_with_Keeper, Academy_Run_Pass_and_Shoot_with_Keeper, Academy_Corner


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


def lle_fn(**kwargs):
    match kwargs["map"]:
        case str(map_file):
            env = LLE.from_file(map_file)
        case int(level):
            env = LLE.level(level)
        case other:
            raise ValueError(f"Invalid map: {other}")

    obs_type = kwargs.get("obs_type", "layered")
    state_type = kwargs.get("state_type", "flattened")
    env = env.obs_type(obs_type).state_type(state_type).single_objective()
    seed = kwargs.get("seed", None)
    if seed is not None:
        seed = int(seed)
        env.seed(seed)
    time_limit = env.width * env.height // 2
    return marlenv.adapters.PymarlAdapter(env, time_limit)


def shaped_lle(
    *,
    gamma: float,
    map: str | int,
    obs_type: Literal["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"] = "layered",
    state_type: Literal["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"] = "flattened",
    seed: int | None = None,
):
    match map:
        case str(map_file):
            env = LLE.from_file(map_file)
        case int(level):
            env = LLE.level(level)
        case other:
            raise ValueError(f"Invalid map: {other}")

    env = env.obs_type(obs_type).state_type(state_type).single_objective()
    if seed is not None:
        seed = int(seed)
        env.seed(seed)
    time_limit = env.width * env.height // 2
    l1 = env.world.laser_sources[4, 0]
    l2 = env.world.laser_sources[6, 12]
    env = LLEPotentialShaping(env, {l1: Direction.SOUTH, l2: Direction.SOUTH}, gamma)
    return marlenv.adapters.PymarlAdapter(env, time_limit)


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "lle": lle_fn,
    "shaped_lle": shaped_lle,
    # "academy_3_vs_1_with_keeper": partial(env_fn, env=Academy_3_vs_1_with_Keeper),
    # "academy_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper),
    # "academy_run_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Run_Pass_and_Shoot_with_Keeper),
    # "academy_corner": partial(env_fn, env=Academy_Corner),
}

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

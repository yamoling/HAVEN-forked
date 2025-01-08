import random

from lle import World
from marlenv.wrappers import RLEnvWrapper, MARLEnv


class RandomizedLLE(RLEnvWrapper):
    def __init__(self, env: MARLEnv, world: World):
        super().__init__(env)
        self.sources = list(world.laser_sources.values())

    def reset(self):
        for laser_source in self.sources:
            new_colour = random.randint(0, self.n_agents)
            laser_source.set_agent_id(new_colour)
        return super().reset()

    def seed(self, seed_value: int):
        random.seed(seed_value)
        return super().seed(seed_value)

from abc import abstractmethod
from typing import Optional
import numpy as np

from lle import World, Position, LLE
from lle.tiles import Direction, LaserSource
from marlenv.wrappers import RLEnvWrapper, MARLEnv
from marlenv import Observation

HORIZONTAL = [Direction.EAST, Direction.WEST]
VERTICAL = [Direction.NORTH, Direction.SOUTH]


class PotentialShaping(RLEnvWrapper):
    """
    Potential shaping for the Laser Learning Environment (LLE).

    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    """

    gamma: float

    def __init__(
        self,
        env: MARLEnv,
        gamma: float,
        extra_shape: Optional[tuple[int]] = None,
    ):
        super().__init__(env, extra_shape=extra_shape)
        self.gamma = gamma
        self.current_potential = self.compute_potential()

    def add_extras(self, obs: Observation) -> Observation:
        """Add the extras related to potential shaping. Does nothing by default."""
        return obs

    def reset(self):
        obs, state = super().reset()
        return self.add_extras(obs), state

    def step(self, actions):
        phi_t = self.current_potential
        step = super().step(actions)

        self.current_potential = self.compute_potential()
        shaped_reward = self.gamma * phi_t - self.current_potential
        step.obs = self.add_extras(step.obs)
        step.reward += shaped_reward
        return step

    @abstractmethod
    def compute_potential(self) -> float:
        """Compute the potential of the current state of the environment."""


class LLEPotentialShaping(PotentialShaping):
    def __init__(
        self,
        env: LLE,
        lasers_to_reward: dict[LaserSource, Direction],
        gamma: float,
        reward_value: float = 1.0,
        enable_extras: bool = True,
    ):
        """
        Parameters:
         - env: The environment to wrap.
         - world: The world of the LLE.
         - lasers_to_reward: A dictionary mapping each laser source that has to be rewarded to the direction in which the agents have to move.
         - discount_factor: The discount factor `gamma`
        """
        self.gamma = gamma
        self.reward_value = reward_value
        self.world = env.world
        self.enable_extras = enable_extras
        self.pos_to_reward = self._compute_positions_to_reward(lasers_to_reward, env.world)
        self.agents_pos_reached = np.full((env.n_agents, len(self.pos_to_reward)), False, dtype=np.bool)
        if enable_extras:
            assert len(env.extra_shape) == 1
            # *2 because we reward in the laser and after the laser
            n_extras = len(lasers_to_reward) * 2
            assert self.agents_pos_reached.shape[1] == n_extras
        super().__init__(env, gamma)  # type: ignore

    @staticmethod
    def _compute_positions_to_reward(lasers_to_reward: dict[LaserSource, Direction], world: World):
        pos_to_reward = list[set[Position]]()
        for source, direction in lasers_to_reward.items():
            # Make sure that the source and direction are compatible
            source_is_vertical = source.direction in VERTICAL
            goal_direction_is_horizontal = direction in HORIZONTAL
            assert (
                source_is_vertical == goal_direction_is_horizontal
            ), "The source and direction are incompatible. For horizontal lasers, the direction must be vertical and vice versa."
            in_laser_rewards = set[Position]()
            after_laser_rewards = set[Position]()
            for (i, j), laser in world.lasers:
                if laser.laser_id == source.laser_id:
                    in_laser_rewards.add((i, j))
                    di, dj = direction.delta()
                    ri, rj = (i + di, j + dj)
                    if ri >= 0 and ri < world.width and rj >= 0 and rj < world.height and (ri, rj) not in world.wall_pos:
                        after_laser_rewards.add((i + di, j + dj))
            pos_to_reward.append(in_laser_rewards)
            pos_to_reward.append(after_laser_rewards)
        return pos_to_reward

    def compute_potential(self) -> float:
        for agent_num, agent_pos in enumerate(self.world.agents_positions):
            for j, rewarded_positions in enumerate(self.pos_to_reward):
                if agent_pos in rewarded_positions:
                    self.agents_pos_reached[agent_num, j] = True
        return float(self.agents_pos_reached.size - self.agents_pos_reached.sum()) * self.reward_value

    def get_laser_shaping(self):
        return self.agents_pos_reached.astype(np.float32)

    def reset(self):
        self.agents_pos_reached.fill(False)
        return super().reset()

import warnings
from functools import partial

import numpy as np
import torch
from marlenv.adapters import PymarlAdapter

from components.episode_buffer import EpisodeBatch
from controllers import BasicMAC, MacroMAC, ValueMAC
from envs import REGISTRY as env_REGISTRY
from envs import LLEPotentialShaping, MultiAgentEnv

warnings.filterwarnings("ignore")


class EpisodeRunner:
    env: MultiAgentEnv
    mac: BasicMAC

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args, gamma=args.gamma)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(
        self,
        scheme,
        macro_scheme,
        groups,
        preprocess,
        macro_preprocess,
        mac,
        macro_mac: MacroMAC | None,
        value_mac: ValueMAC | None,
        learner,
    ):
        self.new_batch = partial(
            EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1, preprocess=preprocess, device=self.args.device
        )
        if hasattr(self.args, "k"):
            self.new_macro_batch = partial(
                EpisodeBatch,
                macro_scheme,
                groups,
                self.batch_size,
                (self.episode_limit // self.args.k) + 1 + (self.episode_limit % self.args.k != 0),
                preprocess=macro_preprocess,
                device=self.args.device,
            )
        else:
            self.new_macro_batch = lambda: None
        self.mac = mac
        self.macro_mac = macro_mac
        self.value_mac = value_mac
        self.learner = learner

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.macro_batch = self.new_macro_batch()
        self.env.reset()
        self.t = 0

    def get_laser_shaping(self):
        assert isinstance(self.env, PymarlAdapter)
        shaping = self.env.env.wrapped
        assert isinstance(shaping, LLEPotentialShaping)
        return shaping.get_laser_shaping()

    def run(self, test_mode=False):
        self.reset()
        macro_actions = None
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.macro_mac is not None:
            self.macro_mac.init_hidden(batch_size=self.batch_size)
        if self.value_mac is not None:
            self.value_mac.init_hidden(batch_size=self.batch_size)
        env_info = {"alive_allies_list": [1 for _ in range(self.args.n_agents)]}

        macro_reward = 0
        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            if "laser_shaping" in self.batch.scheme:
                pre_transition_data["laser_shaping"] = [self.get_laser_shaping()]
            self.batch.update(pre_transition_data, ts=self.t)

            if self.macro_batch is not None and self.t % self.args.k == 0:
                pre_macro_transition_data = {
                    "state": [self.env.get_state()],
                    "obs": [self.env.get_obs()],
                }
                self.macro_batch.update(pre_macro_transition_data, ts=self.t // self.args.k)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.macro_batch is not None and self.t % self.args.k == 0 and self.t != 0:
                post_macro_transition_data = {"macro_reward": [(macro_reward,)], "terminated": [(False,)]}
                if macro_actions is not None:
                    post_macro_transition_data["macro_actions"] = macro_actions

                macro_reward = 0
                self.macro_batch.update(post_macro_transition_data, ts=self.t // self.args.k - 1)
            if self.macro_batch is not None and self.t % self.args.k == 0:
                if self.macro_mac is not None:
                    macro_actions = self.macro_mac.select_actions(
                        self.macro_batch, t_ep=self.t // self.args.k, t_env=self.t_env, test_mode=test_mode
                    )
                elif "laser_shaping" in self.batch.scheme:
                    macro_actions = pre_transition_data["laser_shaping"]
            if self.macro_batch is not None and macro_actions is not None:
                pre_transition_data = {
                    "subgoals": macro_actions,
                }
            else:
                pre_transition_data = {}
            self.batch.update(pre_transition_data, ts=self.t)
            with torch.no_grad():
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            macro_reward += reward

            post_transition_data = {
                "reward": [(reward,)],
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        if self.macro_batch is not None:
            post_macro_transition_data = {
                "macro_reward": [(macro_reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if macro_actions is not None:
                post_macro_transition_data["macro_actions"] = (macro_actions,)
            # macro_index = (self.t - 1) // self.args.k if ((self.t - 1) % self.args.k) == 0 else (self.t - 1) // self.args.k + 1
            macro_index = (self.t - 1) // self.args.k
            self.macro_batch.update(post_macro_transition_data, ts=macro_index)
            last_macro_data = {
                "state": [self.env.get_state()],
                "obs": [self.env.get_obs()],
            }
            self.macro_batch.update(last_macro_data, ts=macro_index + 1)
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.macro_batch is not None:
            pre_transition_data = {}
            if self.macro_mac is not None:
                macro_actions = self.macro_mac.select_actions(self.macro_batch, t_ep=macro_index + 1, t_env=self.t_env, test_mode=test_mode)
            elif "laser_shaping" in self.batch.scheme:
                macro_actions = self.get_laser_shaping()
            if macro_actions is not None:
                pre_transition_data["subgoals"] = macro_actions
            if macro_actions is not None:
                self.macro_batch.update({"macro_actions": macro_actions}, ts=macro_index + 1)
        else:
            pre_transition_data = {}
        self.batch.update(pre_transition_data, ts=self.t)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, self.macro_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

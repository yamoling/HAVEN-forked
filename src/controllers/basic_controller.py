from typing import Any
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import math
from .controller import Controller
from components.episode_buffer import EpisodeBatch


# This multi-agent controller shares parameters between agents
class BasicMAC(Controller):
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        super().__init__(self.agent)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector["type"]](args.action_selector)
        self.hidden_states = None
        self.is_recurrent = self.agent.is_recurrent

    def select_actions(self, ep_batch: EpisodeBatch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch: EpisodeBatch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = (1 - self.action_selector.epsilon) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch: EpisodeBatch, t):
        bs = batch.batch_size
        extras = []
        if "subgoals_onehot" in batch.scheme:
            extras.append(batch["subgoals_onehot"][:, t])
        if "laser_shaping" in batch.scheme:
            extras.append(batch["laser_shaping"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                extras.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                extras.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            extras.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        match batch.scheme["obs"]["vshape"]:
            case int():
                inputs = [batch["obs"][:, t], *extras]
                inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
                return inputs
            case tuple():
                inputs = th.tensor(batch["obs"][:, t])
                extras = th.cat(extras, dim=-1)
                return inputs, extras
            case _:
                raise NotImplementedError()

    def _get_input_shape(self, scheme: dict[str, Any]):
        if "subgoals_onehot" in scheme:
            extras_shape = math.prod(scheme["subgoals_onehot"]["vshape"])
        else:
            extras_shape = 0
        if "laser_shaping" in scheme:
            extras_shape += scheme["laser_shaping"]["vshape"][0]
        if self.args.obs_last_action:
            extras_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            extras_shape += self.n_agents
        match scheme["obs"]["vshape"]:
            case int(obs_shape):
                return obs_shape + extras_shape
            case tuple(obs_shape):
                return obs_shape, extras_shape
            case _:
                raise NotImplementedError()

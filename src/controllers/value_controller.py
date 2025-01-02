from typing import Any
from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
from .controller import Controller
from components.episode_buffer import EpisodeBatch


# This multi-agent controller shares parameters between agents
class ValueMAC(Controller):
    is_recurrent: bool

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        super().__init__(self.agent)
        self.hidden_states = None
        self.is_recurrent = self.agent.is_recurrent

    def forward(self, ep_batch, t: int, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        return agent_outs  # .view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/value_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/value_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.macro_value_network](input_shape, self.args)

    def _build_inputs(self, batch: EpisodeBatch, t: int):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        extras = []
        if self.args.obs_last_action:
            if t == 0:
                extras.append(th.zeros_like(batch["macro_actions_onehot"][:, t]))
            else:
                extras.append(batch["macro_actions_onehot"][:, t - 1])
        if "laser_shaping" in batch.scheme:
            extras.append(batch["laser_shaping"][:, t])
        if self.args.obs_agent_id:
            extras.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        obs = batch["obs"][:, t]
        match batch.scheme["obs"]["vshape"]:
            case int():
                inputs = [obs, *extras]
                inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
                return inputs
            case tuple():
                extras = th.cat(extras, dim=-1)
                return obs, extras
            case _:
                raise NotImplementedError()

    def _get_input_shape(self, scheme) -> int | tuple[tuple, int]:
        extras_shape = 0
        if self.args.obs_last_action:
            extras_shape += scheme["macro_actions_onehot"]["vshape"][0]
        # There is no laser_shaping for the ValueMAC, although it may be encoded in the last_action
        # if "laser_shaping" in scheme:
        #     extras_shape += scheme["laser_shaping"]["vshape"][0]
        if self.args.obs_agent_id:
            extras_shape += self.n_agents

        match scheme["obs"]["vshape"]:
            case int(input_shape):
                return input_shape + extras_shape
            case tuple(input_shape):
                return input_shape, extras_shape
            case _:
                raise NotImplementedError("Unsupported input shape")

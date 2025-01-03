from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .controller import Controller


# This multi-agent controller shares parameters between agents
class MacroMAC(Controller):
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        super().__init__(self.agent)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.macro_action_selector["type"]](args.macro_action_selector)

        self.hidden_states = None
        self.is_recurrent = self.agent.is_recurrent

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        # avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], None, t_env, test_mode=test_mode)

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        # avail_actions = ep_batch["avail_actions"][:, t]
        agent_outputs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outputs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def to(self, device):
        self.agent.to(device)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/macro_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/macro_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.macro_network](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        extras = []
        if self.args.obs_last_action:
            if t == 0:
                last_action = th.zeros_like(batch["macro_actions_onehot"][:, t])
            else:
                last_action = batch["macro_actions_onehot"][:, t - 1]
            extras.append(last_action)
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

    def _get_input_shape(self, scheme):
        extras_shape = 0
        if self.args.obs_last_action and self.args.enable_haven_subgoals:
            extras_shape += scheme["macro_actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            extras_shape += self.n_agents
        match scheme["obs"]["vshape"]:
            case int(input_shape):
                return input_shape + extras_shape
            case tuple(input_shape):
                return input_shape, extras_shape
            case _:
                raise NotImplementedError()

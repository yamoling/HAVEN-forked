import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from controllers import ValueMAC, MacroMAC, BasicMAC


class HAVENLearner:
    def __init__(self, mac: BasicMAC, macro_mac: MacroMAC, value_mac: ValueMAC | None, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.macro_mac = macro_mac
        self.value_mac = value_mac
        self.logger = logger
        self.device = th.device("cpu")
        if self.args.intrinsic_switch != 0:
            assert value_mac is not None

        self.params = list(mac.parameters())
        self.macro_params = list(macro_mac.parameters())
        self.value_params = list[th.nn.Parameter]()
        if value_mac is not None:
            self.value_params = list(value_mac.parameters())

        self.last_target_update_episode = 0
        self.last_target_macro_update_episode = 0
        self.last_target_value_update_episode = 0

        match args.mixer:
            case "vdn":
                self.mixer = VDNMixer()
            case "qmix":
                self.mixer = QMixer(args)
            case other:
                raise ValueError(f"Mixer {other} not recognised.")
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        match args.value_mixer:
            case "vdn":
                self.value_mixer = VDNMixer()
            case "qmix":
                self.value_mixer = QMixer(args)
            case other:
                raise ValueError(f"Mixer {other} not recognised.")
        self.value_params += list(self.value_mixer.parameters())
        self.target_value_mixer = copy.deepcopy(self.value_mixer)

        match args.macro_mixer:
            case "vdn":
                self.macro_mixer = VDNMixer()
            case "qmix":
                self.macro_mixer = QMixer(args)
            case other:
                raise ValueError(f"Mixer {other} not recognised.")
        self.macro_params += list(self.macro_mixer.parameters())
        self.target_macro_mixer = copy.deepcopy(self.macro_mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.macro_optimiser = RMSprop(params=self.macro_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        if self.value_mac is not None:
            self.value_optimiser = RMSprop(params=self.value_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        else:
            self.value_optimiser = None
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_value_mac = copy.deepcopy(value_mac)
        self.target_macro_mac = copy.deepcopy(macro_mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def value_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if self.value_mac is None:
            return
        assert self.value_optimiser is not None
        rewards = batch["macro_reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.value_mac.is_recurrent:
            value_out, mac_out = [], []
            self.value_mac.init_hidden(batch.batch_size)
            self.macro_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                value = self.value_mac.forward(batch, t=t)
                agent_outs = self.macro_mac.forward(batch, t=t)
                value_out.append(value)
                mac_out.append(agent_outs)
            value_out = th.stack(value_out, dim=1)
            mac_out = th.stack(mac_out[1:], dim=1)
        else:
            v_obs, v_extras, m_obs, m_extras = [], [], [], []
            for t in range(batch.max_seq_length):
                o, e = self.value_mac._build_inputs(batch, t)
                v_obs.append(o)
                v_extras.append(e)
                o, e = self.macro_mac._build_inputs(batch, t)
                m_obs.append(o)
                m_extras.append(e)
            v_obs = th.stack(v_obs, dim=1)
            v_extras = th.stack(v_extras, dim=1)
            m_obs = th.stack(m_obs[1:], dim=1)
            m_extras = th.stack(m_extras[1:], dim=1)
            value_out, _ = self.value_mac.agent.forward((v_obs, v_extras))
            mac_out, _ = self.macro_mac.agent.forward((m_obs, m_extras))

        max_qvals = mac_out.max(dim=3)[0]
        max_qvals = self.macro_mixer(max_qvals, batch["state"][:, 1:])

        values = self.value_mixer(value_out[:, :-1], batch["state"][:, :-1])
        # target_values = self.target_value_mixer(target_value_out[:, 1:], batch["state"][:, 1:])

        # target_values = rewards + self.args.gamma * max_qvals * (1 - terminated)
        target_values = rewards + self.args.gamma * max_qvals * (1 - terminated)
        td_loss = values - target_values.detach()

        mask = mask.expand_as(td_loss)
        masked_loss = td_loss * mask
        masked_loss = (masked_loss**2).sum() / mask.sum()

        self.value_optimiser.zero_grad()
        masked_loss.backward()
        _grad_norm = th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
        self.value_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_value_targets()
            self.last_target_value_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("value_loss", masked_loss.item(), t_env)

    def train(self, batch: EpisodeBatch, macro_batch: EpisodeBatch, t_env: int, episode_num: int):
        actions = batch["actions"][:, :-1]
        intrinsic_reward = self.calc_intrinsic_reward(batch, macro_batch)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values and next Q-values
        if self.mac.is_recurrent:
            qvalues = []
            next_qvalues = []
            self.mac.init_hidden(batch.batch_size)
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                qvalues.append(agent_outs)
                target_agent_outs = self.target_mac.forward(batch, t=t)
                next_qvalues.append(target_agent_outs)
            qvalues = th.stack(qvalues, dim=1)  # Concat over time
            # We don't need the first timesteps Q-Value estimate for calculating targets
            next_qvalues = th.stack(next_qvalues[1:], dim=1)  # Concat across time
        else:
            obs, extras, target_obs, target_extras = [], [], [], []
            for t in range(batch.max_seq_length):
                o, e = self.mac._build_inputs(batch, t)
                obs.append(o)
                extras.append(e)
                o, e = self.target_mac._build_inputs(batch, t)
                target_obs.append(o)
                target_extras.append(e)
            obs = th.stack(obs, dim=1)
            extras = th.stack(extras, dim=1)
            qvalues, _ = self.mac.agent.forward((obs, extras))
            target_obs = th.stack(target_obs[1:], dim=1)
            target_extras = th.stack(target_extras[1:], dim=1)
            next_qvalues, _ = self.target_mac.agent.forward((target_obs, target_extras))

        # Mask out unavailable actions
        next_qvalues[avail_actions[:, 1:] == 0] = -9999999

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(qvalues[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = qvalues.detach().clone()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = th.max(mac_out_detach[:, 1:], dim=3, keepdim=True).indices
            target_max_qvals = th.gather(next_qvalues, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = next_qvalues.max(dim=3)[0]

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = intrinsic_reward + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.detach().item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def macro_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["macro_reward"][:, :-1]
        actions = batch["macro_actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate estimated Q-Values
        if self.macro_mac.is_recurrent:
            mac_out = []
            self.macro_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.macro_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time
        else:
            obs, extras = [], []
            for t in range(batch.max_seq_length):
                o, e = self.macro_mac._build_inputs(batch, t)
                obs.append(o)
                extras.append(e)
            obs = th.stack(obs, dim=1)
            extras = th.stack(extras, dim=1)
            mac_out, _ = self.macro_mac.agent.forward((obs, extras))

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        if self.target_macro_mac.is_recurrent:
            target_mac_out = []
            self.target_macro_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_macro_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        else:
            obs, extras = [], []
            for t in range(1, batch.max_seq_length):
                o, e = self.target_macro_mac._build_inputs(batch, t)
                obs.append(o)
                extras.append(e)
            obs = th.stack(obs, dim=1)
            extras = th.stack(extras, dim=1)
            target_mac_out, _ = self.target_macro_mac.agent.forward((obs, extras))

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.macro_mixer is not None:
            chosen_action_qvals = self.macro_mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_macro_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.macro_optimiser.zero_grad()
        loss.backward()
        _grad_norm = th.nn.utils.clip_grad_norm_(self.macro_params, self.args.grad_norm_clip)
        self.macro_optimiser.step()
        if (episode_num - self.last_target_macro_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_macro_targets()
            self.last_target_macro_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("macro_loss", loss.item(), t_env)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_macro_targets(self):
        self.target_macro_mac.load_state(self.macro_mac)
        if self.macro_mixer is not None:
            self.target_macro_mixer.load_state_dict(self.macro_mixer.state_dict())
        self.logger.console_logger.info("Updated target macro network")

    def _update_value_targets(self):
        if self.value_mac is not None and self.target_value_mac is not None:
            self.target_value_mac.load_state(self.value_mac)
        if self.value_mixer is not None:
            self.target_value_mixer.load_state_dict(self.value_mixer.state_dict())
        self.logger.console_logger.info("Updated target value network")

    def to(self, device: th.device):
        self.device = device
        self.mac.to(device)
        self.target_mac.to(device)
        self.macro_mac.to(device)
        self.target_macro_mac.to(device)
        if self.value_mac is not None:
            self.value_mac.to(device)
        if self.target_value_mac is not None:
            self.target_value_mac.to(device)
        if self.mixer is not None:
            self.mixer.to(device)
            self.target_mixer.to(device)
        if self.macro_mixer is not None:
            self.macro_mixer.to(device)
            self.target_macro_mixer.to(device)
        if self.value_mixer is not None:
            self.value_mixer.to(device)
            self.target_value_mixer.to(device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.value_mac is not None:
            self.value_mac.save_models(path)
        if self.value_mixer is not None and self.value_optimiser is not None:
            th.save(self.value_mixer.state_dict(), "{}/value_mixer.th".format(path))
            th.save(self.value_optimiser.state_dict(), "{}/value_opt.th".format(path))
        self.macro_mac.save_models(path)
        if self.macro_mixer is not None:
            th.save(self.macro_mixer.state_dict(), "{}/macro_mixer.th".format(path))
        th.save(self.macro_optimiser.state_dict(), "{}/macro_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.value_mac is not None:
            self.value_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        if self.target_value_mac is not None:
            self.target_value_mac.load_models(path)
        if self.value_mixer is not None and self.value_optimiser is not None:
            self.value_mixer.load_state_dict(th.load("{}/value_mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.value_optimiser.load_state_dict(th.load("{}/value_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.macro_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_macro_mac.load_models(path)
        if self.macro_mixer is not None:
            self.macro_mixer.load_state_dict(th.load("{}/macro_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.macro_optimiser.load_state_dict(th.load("{}/macro_opt.th".format(path), map_location=lambda storage, loc: storage))

    def calc_intrinsic_reward(self, batch: EpisodeBatch, macro_batch: EpisodeBatch):
        if self.args.intrinsic_switch != 0 and self.value_mac is not None:
            origin_reward = batch["reward"][:, :-1]
            if self.args.mean_weight:
                origin_reward = th.ones_like(origin_reward)
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

            if self.value_mac.is_recurrent:
                assert self.macro_mac.is_recurrent
                value_out = []
                macro_mac_out = []
                self.value_mac.init_hidden(macro_batch.batch_size)
                self.macro_mac.init_hidden(macro_batch.batch_size)
                for t in range(macro_batch.max_seq_length):
                    value_out.append(self.value_mac.forward(macro_batch, t=t))
                    macro_mac_out.append(self.macro_mac.forward(macro_batch, t=t))
                value_out = th.stack(value_out, dim=1)
                macro_mac_out = th.stack(macro_mac_out, dim=1)  # Concat over time
            else:
                v_obs, v_extras, m_obs, m_extras = [], [], [], []
                for t in range(macro_batch.max_seq_length):
                    o, e = self.value_mac._build_inputs(macro_batch, t)
                    v_obs.append(o)
                    v_extras.append(e)
                    o, e = self.macro_mac._build_inputs(macro_batch, t)
                    m_obs.append(o)
                    m_extras.append(e)
                v_obs = th.stack(v_obs, dim=1)
                v_extras = th.stack(v_extras, dim=1)
                value_out, _ = self.value_mac.agent.forward((v_obs, v_extras))
                m_obs = th.stack(m_obs, dim=1)
                m_extras = th.stack(m_extras, dim=1)
                macro_mac_out, _ = self.macro_mac.agent.forward((m_obs, m_extras))

            values = self.value_mixer(value_out, macro_batch["state"])
            macro_mac_out = th.gather(macro_mac_out, dim=3, index=macro_batch["macro_actions"]).squeeze(3)
            macro_mac_out = self.macro_mixer(macro_mac_out, macro_batch["state"])
            # values = value_out.squeeze(-1)

            # intrinsic_reward = (macro_mac_out[:, :-1] - values[:, :-1])
            macro_reward = macro_batch["macro_reward"][:, :-1]
            intrinsic_reward = macro_reward + self.args.gamma * values[:, 1:] - values[:, :-1]
            intrinsic_reward = intrinsic_reward.unsqueeze(-2)
            gap = intrinsic_reward.size(1) * self.args.k - origin_reward.size(1)
            if gap != 0:
                origin_reward = th.cat(
                    [
                        origin_reward,
                        th.zeros([intrinsic_reward.size(0), intrinsic_reward.size(1) * self.args.k - origin_reward.size(1), 1]).to(
                            self.device
                        ),
                    ],
                    dim=1,
                )
            origin_reward = origin_reward.view(origin_reward.size(0), -1, self.args.k, 1)
            if not self.args.mean_weight:
                origin_reward[origin_reward == 0] = -9999999
                origin_reward = intrinsic_reward.sign() * origin_reward
            if gap != 0:
                origin_reward[:, :, -gap:] = -9999999
            origin_reward_weight = th.softmax(origin_reward, dim=-2)
            intrinsic_reward = intrinsic_reward * origin_reward_weight
            intrinsic_reward = intrinsic_reward.view(intrinsic_reward.size(0), -1, 1 if self.args.mixer is not None else self.args.n_agents)
            intrinsic_reward = (intrinsic_reward[:, : batch.max_seq_length - 1] * mask).detach()
        else:
            intrinsic_reward = 0.0
        intrinsic_reward = (
            self.args.intrinsic_switch * intrinsic_reward + self.args.reward_switch * batch["reward"][:, : batch.max_seq_length - 1]
        )
        return intrinsic_reward

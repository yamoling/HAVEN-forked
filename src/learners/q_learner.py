import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from .learner import Learner


class QLearner(Learner):
    def __init__(self, mac, macro_mac, value_mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.min_eps = args.action_selector["epsilon_finish"]
        self.last_target_update_episode = 0

        if args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = QMixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        if hasattr(args, "intrinsic_type"):
            self.intrinsic_reward = True
            match args.intrinsic_type:
                case "potential":
                    pass
                case other:
                    raise ValueError(f"Invalid intrinsic type: {other}")
        else:
            self.intrinsic_reward = False

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def compute_potential_reward(
        self,
        qvalues: th.Tensor,
        next_qvalues: th.Tensor,
        terminated: th.Tensor,
        available_actions: th.Tensor,
        batch,
    ) -> th.Tensor:
        with th.no_grad():
            next_values = th.max(next_qvalues, dim=-1)[0] * (1 - terminated)
            phi_t_prime = self.target_mixer.forward(next_values, batch["state"][:, 1:])
            available_actions = available_actions[:, :-1]
            qvalues = qvalues.clone()
            qvalues[available_actions == 0] = -9999999
            # We take the weighted average of the Q-Values across the agents
            # All available actions that are not the maximal action have a weight of self.min_eps / n_actions
            # The max action has a weight of (1 - self.min_eps) + self.min_eps / n_actions
            n_available_actions = th.sum(available_actions, dim=-1, keepdim=True) + 1e-8
            weights = th.full_like(qvalues, self.min_eps) / n_available_actions * available_actions
            max_values = th.max(qvalues, dim=-1, keepdim=True).values
            mask = qvalues == max_values
            # Add (1-min_eps) to the indices that match the ones of the maximal action
            weights[mask] += 1 - self.min_eps
            weighted_qvalues = th.sum(qvalues * weights, dim=-1)
            phi_t = self.mixer.forward(weighted_qvalues, batch["state"][:, :-1])

        return self.args.gamma * phi_t_prime - phi_t

    def train(self, batch: EpisodeBatch, _marco_batch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated: th.Tensor = batch["terminated"][:, :-1].float()  # type: ignore
        mask: th.Tensor = batch["filled"][:, :-1].float()  # type: ignore
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions: th.Tensor = batch["avail_actions"]  # type: ignore

        # Calculate estimated Q-Values
        qvalues = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            qvalues.append(agent_outs)
        all_qvalues = th.stack(qvalues, dim=1)  # Concat over time
        qvalues = all_qvalues.clone()[:, :-1]  # Don't need the last one

        # Pick the Q-Values for the actions taken by each agent
        chosen_qvalues = th.gather(qvalues, dim=3, index=actions).squeeze(3)  # type: ignore

        # Calculate the Q-Values necessary for the target
        next_qvalues = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            next_qvalues.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        next_qvalues = th.stack(next_qvalues[1:], dim=1)  # Concat across time
        # Mask out unavailable actions
        next_qvalues[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = all_qvalues.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(next_qvalues, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = next_qvalues.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_qvalues = self.mixer(chosen_qvalues, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        if self.intrinsic_reward:
            shaped_reward = self.compute_potential_reward(qvalues, next_qvalues, terminated, avail_actions, batch).detach()
            shaped_reward = shaped_reward * mask
            self.logger.log_stat("shaped_reward", float(shaped_reward.mean().item()), t_env)
            rewards = rewards + shaped_reward

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = chosen_qvalues - targets.detach()

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
            self.logger.log_stat("loss", float(loss.item()), t_env)
            self.logger.log_stat("grad_norm", float(grad_norm.item()), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", float((masked_td_error.abs().sum().item() / mask_elems)), t_env)
            self.logger.log_stat("q_taken_mean", float((chosen_qvalues * mask).sum().item() / (mask_elems * self.args.n_agents)), t_env)
            self.logger.log_stat("target_mean", float((targets * mask).sum().item() / (mask_elems * self.args.n_agents)), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def to(self, device):
        self.mac.to(device)
        self.target_mac.to(device)
        if self.mixer is not None:
            self.mixer.to(device)
            self.target_mixer.to(device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

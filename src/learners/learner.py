from components.episode_buffer import EpisodeBatch
from controllers import ValueMAC, MacroMAC, BasicMAC
import torch


class Learner:
    def __init__(self, mac: BasicMAC, macro_mac: MacroMAC | None, value_mac: ValueMAC | None, scheme, logger, args):
        self.mac = mac
        self.macro_mac = macro_mac
        self.value_mac = value_mac
        self.scheme = scheme
        self.logger = logger
        self.args = args

    def train(self, batch: EpisodeBatch, macro_batch: EpisodeBatch, t_env: int, episode_num: int):
        pass

    def to(self, device: torch.device):
        pass

    def save_models(self, path: str):
        pass

    def load_models(self, path: str):
        pass

import torch


class Controller:
    n_agents: int
    agent: torch.nn.Module
    add_last_action: bool
    add_agent_id: bool

    def __init__(self, agent: torch.nn.Module):
        self.agent = agent

    def to(self, device: torch.device):
        self.agent.to(device)

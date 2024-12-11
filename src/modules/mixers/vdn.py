import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        bs = agent_qs.size(0)
        return th.sum(agent_qs, dim=2, keepdim=True).view(bs, -1, 1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

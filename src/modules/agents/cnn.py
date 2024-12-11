from torch import nn
import torch
import math


class CNNAgent(nn.Module):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers (512, 256, output_shape[0]).
    """

    def __init__(self, inputs: tuple[tuple[int, int, int], int], n_outputs: int):
        super().__init__()
        input_shape, extras_size = inputs
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        self.linear = MLP(n_features, extras_size, (64, 64), n_outputs)
        self.output_shape = (n_outputs,)
        self.extras_shape = (extras_size,)

    def forward(self, obs_extras: tuple[torch.Tensor, torch.Tensor], hidden_states):
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        obs, extras = obs_extras
        *dims, channels, height, width = obs.shape
        bs = math.prod(dims)
        # obs = obs.view(bs, channels, height, width)
        obs = obs.reshape(bs, channels, height, width)
        features = self.cnn.forward(obs)
        # extras = extras.view(bs, *self.extras_shape)
        extras = extras.reshape(bs, *self.extras_shape)
        res = self.linear.forward(features, extras)
        return res.view(*dims, *self.output_shape), hidden_states

    @staticmethod
    def agent(inputs: tuple[tuple[int, int, int], int], args):
        return CNNAgent(inputs, args.n_actions)

    @staticmethod
    def macro_agent(inputs: tuple[tuple[int, int, int], int], args):
        return CNNAgent(inputs, args.n_subgoals)

    @staticmethod
    def value(inputs: tuple[tuple[int, int, int], int], args):
        return CNNAgent(inputs, 1)

    def init_hidden(self):
        return


def make_cnn(input_shape, filters: list[int], kernel_sizes: list[int], strides: list[int], min_output_size=1024):
    """Create a CNN with flattened output based on the given filters, kernel sizes and strides."""
    channels, height, width = input_shape
    paddings = [0 for _ in filters]
    n_padded = 0
    output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
    output_size = filters[-1] * output_w * output_h
    while output_w <= 1 or output_h <= 1 or output_size < min_output_size:
        # Add paddings if the output size is negative
        paddings[n_padded % len(paddings)] += 1
        n_padded += 1
        output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
        output_size = filters[-1] * output_w * output_h
    assert output_h > 0 and output_w > 0, f"Input size = {input_shape}, output witdh = {output_w}, output height = {output_h}"
    modules = []
    for f, k, s, p in zip(filters, kernel_sizes, strides, paddings):
        modules.append(torch.nn.Conv2d(in_channels=channels, out_channels=f, kernel_size=k, stride=s, padding=p))
        modules.append(torch.nn.ReLU())
        channels = f
    modules.append(torch.nn.Flatten())
    return torch.nn.Sequential(*modules), output_size


def conv2d_size_out(input_width: int, input_height: int, kernel_sizes: list[int], strides: list[int], paddings: list[int]):
    """
    Compute the output width and height of a sequence of 2D convolutions.
    See shape section on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    width = input_width
    height = input_height
    for kernel_size, stride, pad in zip(kernel_sizes, strides, paddings):
        width = (width + 2 * pad - (kernel_size - 1) - 1) // stride + 1
        height = (height + 2 * pad - (kernel_size - 1) - 1) // stride + 1
    return width, height


class MLP(torch.nn.Module):
    """
    Multi layer perceptron
    """

    layer_sizes: tuple[int, ...]

    def __init__(
        self,
        input_size: int,
        extras_size: int,
        hidden_sizes: tuple[int, ...],
        n_outputs: int,
    ):
        super().__init__()
        self.output_shape = (n_outputs,)
        self.layer_sizes = (input_size + extras_size, *hidden_sizes, n_outputs)
        layers = [torch.nn.Linear(input_size + extras_size, hidden_sizes[0]), torch.nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], n_outputs))
        self.nn = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, _obs_size = obs.shape
        obs = torch.concat((obs, extras), dim=-1)
        x = self.nn(obs)
        return x.view(*dims, *self.output_shape)

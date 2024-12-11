from .rnn_agent import RNNAgent
from .cnn import CNNAgent
from .macro_agent import MacroAgent
from .value_agent import VALUEAgent

REGISTRY = {
    "rnn": RNNAgent,
    "macro": MacroAgent,
    "value": VALUEAgent,
    "cnn": CNNAgent.agent,
    "macro-cnn": CNNAgent.macro_agent,
    "value-cnn": CNNAgent.value,
}

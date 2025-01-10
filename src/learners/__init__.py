from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .haven_learner import HAVENLearner
from .maser_q_learner import MASERQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["haven_learner"] = HAVENLearner
REGISTRY["maser_q_learner"] = MASERQLearner

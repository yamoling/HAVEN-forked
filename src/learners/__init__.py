from typing import Type

from .learner import Learner
from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .haven_learner import HAVENLearner
from .maser_q_learner import MASERQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

REGISTRY: dict[str, Type[Learner]] = {
    "q_learner": QLearner,
    "coma_learner": COMALearner,
    "qtran_learner": QTranLearner,
    "haven_learner": HAVENLearner,
    "maser_q_learner": MASERQLearner,
    "dmaq_qatten_learner": DMAQ_qattenLearner,
}

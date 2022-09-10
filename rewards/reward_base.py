from abc import ABC, abstractmethod
from numbers import Number


class RewardBase(ABC):
    def __init__(self):
        super(RewardBase, self).__init__()

    @abstractmethod
    def get_reward(self) -> Number:
        pass

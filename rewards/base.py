from abc import ABC, abstractmethod


class RewardBase(ABC):
    def __init__(self) -> None:
        super(RewardBase, self).__init__()

    @abstractmethod
    def get_reward(self) -> float:
        pass

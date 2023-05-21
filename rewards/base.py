from abc import ABC, abstractmethod
import numpy as np
from utility_functions import UtilityFunctionBase


class RewardBase(ABC):
    def __init__(self) -> None:
        super(RewardBase, self).__init__()

    @abstractmethod
    def get_reward(self) -> float:
        raise NotImplementedError()

    def get_expected_utility(
            self,
            utility_function: UtilityFunctionBase,
            n_samples: int,
    ) -> float:
        utility_rewards = []
        for sample in range(n_samples):
            reward = self.get_reward()
            utility_reward = utility_function.apply(reward)
            utility_rewards.append(utility_reward)
        return np.mean(utility_rewards)

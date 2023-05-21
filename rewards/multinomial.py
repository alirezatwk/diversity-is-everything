from typing import List

import numpy as np

from rewards.base import RewardBase
from utility_functions import UtilityFunctionBase


# TODO: Choose a better name
class MultinomialReward(RewardBase):
    def __init__(self, values: List[float], probabilities: List[float]) -> None:
        super(MultinomialReward, self).__init__()
        self.values = values
        self.probabilities = probabilities

    def get_reward(self) -> float:
        return np.random.choice(self.values, p=self.probabilities)

    def get_expected_utility(self, utility_function: UtilityFunctionBase, n_samples: int,) -> float:
        expected_utility = 0
        for value, probability in zip(self.values, self.probabilities):
            utility_reward = utility_function.apply(value)
            expected_utility += utility_reward * probability
        return expected_utility

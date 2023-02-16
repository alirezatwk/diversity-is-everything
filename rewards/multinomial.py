from typing import List

import numpy as np

from rewards.base import RewardBase


# TODO: Better name
class MultinomialReward(RewardBase):
    def __init__(self, values: List[float], probabilities: List[float]) -> None:
        super(MultinomialReward, self).__init__()
        self.values = values
        self.probabilities = probabilities

    def get_reward(self) -> float:
        return np.random.choice(self.values, p=self.probabilities)

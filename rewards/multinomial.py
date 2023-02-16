import numpy as np

from rewards.base import RewardBase


class MultinomialReward(RewardBase):
    def __init__(self, values, probabilities):
        super(MultinomialReward, self).__init__()
        self.values = values
        self.probabilities = probabilities

    def get_reward(self) -> float:
        return np.random.choice(self.values, p=self.probabilities)

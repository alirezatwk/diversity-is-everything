from numbers import Number

import numpy as np

from rewards.reward_base import RewardBase


class MultinomialReward(RewardBase):
    def __init__(self, values, probabilities):
        super(MultinomialReward, self).__init__()
        self.values = values
        self.probabilities = probabilities

    def get_reward(self) -> Number:
        ind = np.random.randint(len(self.values))
        return self.values[ind]

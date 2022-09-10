from numbers import Number

import numpy as np

from rewards.reward_base import RewardBase


class GaussianReward(RewardBase):
    def __init__(self, mean, std):
        super(GaussianReward, self).__init__()
        self.mean = mean
        self.std = std

    def get_reward(self) -> Number:
        return np.random.normal(loc=self.mean, scale=self.std)

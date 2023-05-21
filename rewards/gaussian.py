import numpy as np

from rewards import RewardBase


class GaussianReward(RewardBase):
    def __init__(self, mean: float, std: float) -> None:
        super(GaussianReward, self).__init__()
        self.mean = mean
        self.std = std

    def get_reward(self) -> float:
        return np.random.normal(loc=self.mean, scale=self.std)

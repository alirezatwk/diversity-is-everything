from typing import List

from rewards import RewardBase


class RewardPackage:
    def __init__(self, rewards: List[RewardBase]):
        self.rewards = rewards

from rewards.reward_base import RewardBase


class DeterministicReward(RewardBase):
    def __init__(self, value):
        super(DeterministicReward, self).__init__()
        self.value = value

    def get_reward(self) -> float:
        return self.value

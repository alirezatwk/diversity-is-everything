from rewards import RewardBase


class DeterministicReward(RewardBase):
    def __init__(self, value: float) -> None:
        super(DeterministicReward, self).__init__()
        self.value = value

    def get_reward(self) -> float:
        return self.value

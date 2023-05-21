from rewards import RewardBase
from utility_functions import UtilityFunctionBase


class DeterministicReward(RewardBase):
    def __init__(self, value: float) -> None:
        super(DeterministicReward, self).__init__()
        self.value = value

    def get_reward(self) -> float:
        return self.value

    def get_expected_utility(self, utility_function: UtilityFunctionBase, n_samples: int,) -> float:
        return utility_function.apply(self.value)

import numpy as np
from configs import INFINITY
from agents import AgentBase
from utility_functions import UtilityFunctionBase
from environments import EnvironmentBase


class ThompsonSamplingAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            environment: EnvironmentBase,
            part_of_agent: bool = False,
            infinity: int = INFINITY,
    ):
        super(ThompsonSamplingAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent,
        )
        self.utility_function = utility_function
        self.stds = np.full((self.n_actions,), infinity)
        self.means = np.zeros((self.n_actions,))

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)

    def _get
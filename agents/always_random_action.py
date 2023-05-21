import numpy as np

from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class AlwaysRandomAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            EUs: np.array = None,
            environment: EnvironmentBase = None,
            part_of_agent: bool = False,
    ):
        super(AlwaysRandomAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)
    def select_action(self) -> int:
        return np.random.choice(self.n_actions)

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        pass
    
    def set_environment_info_after_submission(self):
        pass

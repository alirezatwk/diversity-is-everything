import numpy as np

from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class AlwaysSecondBestAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            EUs: np.array = None,
            environment: EnvironmentBase = None,
            part_of_agent: bool = False,
    ):
        super(AlwaysSecondBestAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function
        if environment is not None:
            self.EUs = EUs
            self.best_action = np.random.choice(np.flatnonzero(self.EUs == self.EUs.max()))
            SecondEUs = np.array([x for x in self.EUs if x < max(self.EUs)])
            self.second_best_action = np.random.choice(np.flatnonzero(self.EUs == SecondEUs.max()))

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)

    def set_environment(self, environment, exp_us):
        super().set_environment(environment, exp_us)
        self.EUs = exp_us
        self.best_action = np.random.choice(np.flatnonzero(self.EUs == self.EUs.max()))
        SecondEUs = np.array([x for x in self.EUs if x < max(self.EUs)])
        self.second_best_action = np.random.choice(np.flatnonzero(self.EUs == SecondEUs.max()))

    def select_action(self) -> int:
        return self.second_best_action

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        pass
    
    def set_environment_info_after_submission(self):
        pass

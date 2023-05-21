import numpy as np

from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class AlwaysWorstAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            EUs: np.array = None,
            environment: EnvironmentBase = None,
            part_of_agent: bool = False,
    ):
        super(AlwaysWorstAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function
        if environment is not None: 
            self.EUs = EUs
            self.worst_action = np.random.choice(np.flatnonzero(self.EUs == self.EUs.min()))

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)
    
    def set_environment(self, environment, exp_us):
        super().set_environment(environment, exp_us) 
        self.EUs = exp_us
        self.worst_action = np.random.choice(np.flatnonzero(self.EUs == self.EUs.min()))
        
    def select_action(self) -> int:
        return self.worst_action
    
    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        pass

    def set_environment_info_after_submission(self):
        pass

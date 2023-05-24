import numpy as np

from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class PercentBestAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            p: float,
            difference:float = 0.0,
            limit:float = 1,
            zero_at: bool = False,
            EUs: np.array = None,
            environment: EnvironmentBase = None,
            part_of_agent: bool = False,
    ):
        super(PercentBestAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function
        self.p = p
        self.differnce = difference
        self.zero_at = zero_at
        self.limit = limit
        if environment is not None: 
            self.EUs = EUs
            self.best_action = np.random.choice(np.flatnonzero(self.EUs == self.EUs.max()))

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)
    
    def set_environment(self, environment, exp_us):
        super().set_environment(environment, exp_us) 
        self.EUs = exp_us
        self.best_action = np.random.choice(np.flatnonzero(self.EUs == self.EUs.max()))
        
    def select_action(self) -> int:
        if np.random.rand() < self.p:
            action = self.best_action
        else:
            action = np.random.choice(self.n_actions)
        return action
    
    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        self.p = self.p + self.differnce
        if self.p > self.limit:
            self.p = self.limit
            self.differnce = 0
            
        if self.zero_at:
            if self.p > 1:
                self.p = 0
                self.differnce = 0

    def set_environment_info_after_submission(self):
        pass

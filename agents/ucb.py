import numpy as np

from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class UCBAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            c: float,
            environment: EnvironmentBase,
            part_of_agent: bool = False,
    ):
        super(UCBAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function
        self.c = c
        self.trial = 0
        self.steps = np.zeros(self.n_actions)
        self.q_values = np.zeros(self.n_actions)

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)

    def select_action(self) -> int:
        if self.trial < self.n_actions:
            return np.random.choice(np.flatnonzero(self.steps == 0))
        ucb = self.q_values + np.sqrt(self.c * np.log(self.trial + 1) / self.steps)
        return np.random.choice(np.flatnonzero(ucb == ucb.max()))

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int):
        self.trial += 1
        self.steps[action] += 1
        self.q_values[action] += (personalized_reward - self.q_values[action]) / self.steps[action]

    def set_environment_info_after_submission(self):
        pass

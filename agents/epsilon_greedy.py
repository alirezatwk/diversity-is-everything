import numpy as np

from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class EpsilonGreedyAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            epsilon: float,
            epsilon_decay: float,
            environment: EnvironmentBase,
            part_of_agent: bool = False,
    ):
        super(EpsilonGreedyAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.steps = np.zeros(self.n_actions)
        self.q_values = np.zeros(self.n_actions)

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)

    def _get_best_action(self) -> int:
        max_indices = np.argwhere(self.q_values == np.amax(self.q_values)).flatten()
        return np.random.choice(max_indices)

    def select_action(self) -> int:
        p = np.random.rand()
        if p < self.epsilon:
            return np.random.randint(self.n_actions)
        return self._get_best_action()

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int):
        self.steps[action] += 1
        self.q_values[action] += (personalized_reward - self.q_values[action]) / self.steps[action]
        self.epsilon *= self.epsilon_decay

    def set_environment_info_after_submission(self):
        pass

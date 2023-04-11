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

            environment: EnvironmentBase = None,
    ):
        super(EpsilonGreedyAgent, self).__init__(id=id, utility_function=utility_function, environment=environment)
        self.epsilon = epsilon
        if environment is not None:
            self.steps = np.zeros(self.n_actions)
            self.q_values = np.zeros(self.n_actions)

    def set_environment(self, environment: EnvironmentBase):
        self.environment = environment
        self.n_actions = environment.get_n_actions()
        self.environment.add_agent(agent_id=self.id, agent=self)
        self.steps = np.zeros(self.n_actions)
        self.q_values = np.zeros(self.n_actions)

    def _get_best_action(self) -> int:
        max_indices = np.argwhere(self.q_values == np.amax(self.q_values)).flatten()
        return np.random.choice(max_indices)

    def select_action(self) -> int:
        p = np.random.rand()
        if p < self.epsilon:
            return np.random.randint(self.n_actions)
        return self._get_best_action()

    def update(self, observation, inner_reward, done, info, action):
        self.steps[action] += 1
        if self.learning_rate is not None:
            self.q_values[action] += (inner_reward - self.q_values[action]) * self.learning_rate
        else:
            self.q_values[action] += (inner_reward - self.q_values[action]) / self.steps[action]

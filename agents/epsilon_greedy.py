import numpy as np
from agents.base import AgentBase


class EpsilonGreedyAgent(AgentBase):
    def __init__(self, id, environment, utility_function, epsilon):
        super(EpsilonGreedyAgent, self).__init__(id=id, utility_function=utility_function, environment=environment)
        self.n_actions = environment.get_n_actions()
        self.epsilon = epsilon
        self.steps = np.zeros(self.n_actions)
        self.q_values = np.zeros(self.n_actions)

    def select_action(self) -> int:
        p = np.random.rand()
        if p < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_values)

    def update(self, observation, inner_reward, done, info, action):
        self.n_actions[action] += 1
        self.q_values[action] += (inner_reward - self.q_values[action]) / self.n_actions[action]

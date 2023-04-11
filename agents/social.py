import numpy as np

from agents import AgentBase
from configs import EPSILON
from environments import EnvironmentBase


class SocialAgent(AgentBase):
    def __init__(
            self,
            id: str,
            individual_agent: AgentBase,
            learning_rate: float = None,
            environment: EnvironmentBase = None,
            epsilon: float = EPSILON,
    ):
        super(SocialAgent, self).__init__(
            id=id,
            utility_function=individual_agent.utility_function,
            learning_rate=learning_rate,
            environment=environment
        )
        self.individual = individual_agent
        self.epsilon = epsilon

        self.history = []
        # Total mean reward
        self.mean_reward = 0
        self.n = 0

        if self.environment is not None:
            self.set_environment(environment=environment)

    def set_environment(self, environment: EnvironmentBase):
        self.individual.set_environment(environment=environment)
        self.environment = environment
        self.n_actions = environment.get_n_actions()
        self.environment.add_agent(agent_id=self.id, agent=self)
        self.n_agents = self.environment.get_n_agents()
        self.preference = np.zeros(self.n_agents) + self.epsilon
        self.agents_id = self.environment.get_agents_id()


    def update_social_preference(self, social_agent_ind, reward, action, alpha=0.1):
        same_action = [i for i in range(len(self.environment.agents))
                       if self.environment.agents_last_choice[i] == action]
        if self.id not in same_action:
            if self.individual.select_action() == action:
                same_action.append(self.id)
        self.preference = np.round(self.preference, 2)
        d = np.sum(np.exp(self.preference))
        mask = np.ones(len(self.preference)) * alpha
        self.preference = self.preference - mask * (reward - self.mean_reward) * np.exp(self.preference) / d
        for index in same_action:
            self.preference[index] = self.preference[index] + alpha * (reward - self.mean_reward)
        return

    def select_agent(self):
        scores = np.exp(self.preference)
        probabilities = scores / np.sum(scores)
        agent_ind = np.random.choice(len(self.preference), p=probabilities)
        return agent_ind

    def take_action(self) -> (object, float, bool, object, int):
        social_agent_id = self.select_agent()
        if self.agents_id[social_agent_id] == self.id or len(self.environment.agents_actions[social_agent_id]) == 0:
            observation, reward, done, info, action = self.individual.take_action()
            self.history.append(self.id)
        else:
            action = self.environment.get_action(step=-1, agent_id=self.agents_id[social_agent_id])
            observation, reward, done, info = self.environment.step(action, self.id)
            inner_reward = self.individual.utility_function.apply(reward)
            self.individual.update(observation, inner_reward, done, info, action)
            self.history.append(self.agents_id[social_agent_id])

        self.n += 1
        inner_reward = self.utility_function.apply(reward)
        self.mean_reward = self.mean_reward + (inner_reward - self.mean_reward) / self.n
        self.update_social_preference(social_agent_ind, inner_reward, action)
        return observation, reward, done, info, action
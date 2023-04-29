from typing import TYPE_CHECKING

import numpy as np

from agents import AgentBase
from configs import EPSILON, SOCIAL_ALPHA

if TYPE_CHECKING:
    from environments import EnvironmentBase


class PreferenceBasedSocialAgent(AgentBase):
    def __init__(
            self,
            id: str,
            individual_agent: AgentBase,
            environment: 'EnvironmentBase',
            epsilon: float = EPSILON,
            alpha: float = SOCIAL_ALPHA,
    ):
        super(PreferenceBasedSocialAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=False,
        )
        self.individual = individual_agent
        self.epsilon = epsilon
        self.alpha = alpha

        self.history = []
        self.mean_reward = 0
        self.step = 0

    def _personalize_reward(self, reward: float) -> float:
        return self.individual.utility_function.apply(reward)

    def _select_agent(self) -> int:
        scores = np.exp(self.preference)
        probabilities = scores / np.sum(scores)
        agent_ind = np.random.choice(len(self.preference), p=probabilities)
        return agent_ind

    def _update_social_preference(self, reward: float, action: int) -> None:
        same_action = [i for i in range(self.n_agents)
                       if self.environment.get_action(step=self.step-2, agent_id=self.agents_id[i]) == action]
        self.preference = np.round(self.preference, 2)
        sum_preferences = np.sum(np.exp(self.preference))
        mask = np.ones(len(self.preference)) * self.alpha
        self.preference = self.preference - mask * (reward - self.mean_reward) * np.exp(self.preference) / sum_preferences
        for index in same_action:
            self.preference[index] = self.preference[index] + self.alpha * (reward - self.mean_reward)

    def select_action(self) -> int:
        social_agent_id = self._select_agent()
        if self.agents_id[social_agent_id] == self.id or self.step == 0:
            action = self.individual.select_action()
            self.history.append(self.id)
        else:
            action = self.environment.get_action(step=self.step-1, agent_id=self.agents_id[social_agent_id])
            self.history.append(self.agents_id[social_agent_id])
        self.step += 1
        return action

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        self.individual.update(
            observation=observation,
            personalized_reward=personalized_reward,
            done=done,
            info=info,
            action=action
        )
        self.mean_reward += (personalized_reward - self.mean_reward) / self.step
        if self.step > 1:
            self._update_social_preference(reward=personalized_reward, action=action)

    def set_environment_info_after_submission(self):
        self.n_agents = self.environment.get_n_agents()
        self.preference = np.zeros(self.n_agents) + self.epsilon
        self.agents_id = self.environment.get_agents_id()

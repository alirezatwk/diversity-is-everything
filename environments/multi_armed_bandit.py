from typing import List

from environments import EnvironmentBase
from rewards import RewardBase


class MultiArmedBanditEnvironment(EnvironmentBase):
    def __init__(self, rewards: List[RewardBase]):
        super(MultiArmedBanditEnvironment, self).__init__()
        self.arms_rewards = rewards
        self.agents = {}
        self.agents_actions = {}
        self.is_submitted = False

    def add_agent(self, agent_id: str, agent):  # TODO: Add type hint
        assert self.is_submitted
        self.agents[agent_id] = agent
        self.agents_actions[agent_id] = []

    def get_n_agents(self) -> int:
        return len(self.agents)

    def get_agents_id(self) -> List[str]:
        return list(self.agents.keys())

    def submit(self):
        self.is_submitted = True

    def _calculate_reward(self, action: int) -> float:
        return self.arms_rewards[action].get_reward()

    def _update_state(self, action: int, agent_id: str):
        self.agents_actions[agent_id].append(action)

    def step(self, action: int, agent_id: str):
        assert not self.is_submitted
        reward = self._calculate_reward(action=action)
        self._update_state(action=action, agent_id=agent_id)
        observation = {}
        info = {}
        done = False
        return observation, reward, done, info

    def get_n_actions(self) -> int:
        return len(self.arms_rewards)

    def get_action(self, step: int, agent_id: str) -> int:
        return self.agents_actions[agent_id][step]

    def reset(self):
        self.agents = {}
        self.agents_actions = {}

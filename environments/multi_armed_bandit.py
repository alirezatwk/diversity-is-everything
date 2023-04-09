from environments import EnvironmentBase
from rewards import RewardBase
from typing import List
from collections import defaultdict


class MultiArmedBanditEnvironment(EnvironmentBase):
    def __init__(self, rewards: List[RewardBase]):
        super(MultiArmedBanditEnvironment, self).__init__()
        self.arms_rewards = rewards
        self.agents_actions = defaultdict(lambda: [])

    def calculate_reward(self, action: int) -> float:
        return self.arms_rewards[action].get_reward()

    def update_state(self, action: int, agent_id: str):
        self.agents_actions[agent_id].append(action)

    def step(self, action: int, agent_id: str):
        reward = self.calculate_reward(action=action)
        self.update_state(action=action, agent_id=agent_id)
        

    def reset(self):
        self.agents_actions = defaultdict(lambda: [])
        
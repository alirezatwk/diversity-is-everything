from abc import ABC
from typing import List


class EnvironmentBase(ABC):

    def __init__(self):
        pass

    def add_agent(self, agent_id: str, agent):  # TODO: Use Type hint
        pass

    def set_agent(self, agent_id: str, agent):
        pass

    def get_n_agents(self) -> int:
        pass

    def step(self, action: int, agent_id: str):
        pass

    def get_n_actions(self) -> int:
        pass

    def get_agents_id(self) -> List[str]:
        pass

    def get_action(self, step: int, agent_id: str):
        pass

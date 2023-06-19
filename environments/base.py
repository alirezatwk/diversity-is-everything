from abc import ABC
from typing import List


# TODO: Clean the methods of this class.
class EnvironmentBase(ABC):

    def __init__(self):
        pass

    def add_agent(self, agent_id: str, agent):
        raise NotImplementedError()

    def set_agent(self, agent_id: str, agent):
        raise NotImplementedError()

    def get_n_agents(self) -> int:
        raise NotImplementedError()

    def step(self, action: int, agent_id: str):
        raise NotImplementedError()

    def get_n_actions(self) -> int:
        raise NotImplementedError()

    def get_agents_id(self) -> List[str]:
        raise NotImplementedError()

    def get_action(self, step: int, agent_id: str):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

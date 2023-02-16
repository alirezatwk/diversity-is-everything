from abc import ABC


class EnvironmentBase(ABC):

    def __init__(self):
        pass

    def add_agent(self):
        pass

    def step(self, action: int, id: int):
        pass
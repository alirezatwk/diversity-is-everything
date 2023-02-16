from abc import ABC, abstractmethod

from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class AgentBase(ABC):
    def __init__(
            self,
            id: int,
            utility_function: UtilityFunctionBase,
            environment: EnvironmentBase,
    ) -> None:
        super(AgentBase, self).__init__()
        self.id = id
        self.utility_function = utility_function
        self.environment = environment
        self.n_actions = environment.actions_count()
        self.add_agent()

    def add_agent(self) -> None:
        if self.id != -1:
            self.environment.add_agent()

    @abstractmethod
    def select_action(self) -> int:
        pass

    @abstractmethod
    def update(self, observation: object, inner_reward: float, done: bool, info: object, action: int) -> None:
        pass

    def take_action(self) -> (object, float, bool, object, int):
        action = self.select_action()
        observation, reward, done, info = self.environment.step(action, self.id)
        inner_reward = self.utility_function.apply(reward)
        self.update(observation, inner_reward, done, info, action)
        return observation, inner_reward, done, info, action

from abc import ABC, abstractmethod

from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class AgentBase(ABC):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            learning_rate: float = None,
            environment: EnvironmentBase = None,
    ) -> None:
        super(AgentBase, self).__init__()
        self.id = id
        self.utility_function = utility_function
        self.learning_rate = learning_rate
        if environment is not None:
            self.set_environment(environment=environment)

    def set_environment(self, environment: EnvironmentBase):
        self.environment = environment
        self.n_actions = environment.get_n_actions()
        self.environment.add_agent(agent_id=self.id, agent=self)

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

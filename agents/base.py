from abc import ABC, abstractmethod
from typing import Tuple

from environments import EnvironmentBase

class AgentBase(ABC):
    def __init__(self, id: str, environment: EnvironmentBase = None, part_of_agent: bool = False) -> None:
        super(AgentBase, self).__init__()
        self.id = id
        self.environment = environment
        self.part_of_agent = part_of_agent
        if environment is not None:
            self.n_actions = environment.get_n_actions()
            if not part_of_agent:
                environment.add_agent(agent_id=self.id, agent=self)

    @abstractmethod
    def _personalize_reward(self, reward: float) -> float:
        raise NotImplementedError()

    @abstractmethod
    def set_environment_info_after_submission(self):
        raise NotImplementedError()

    @abstractmethod
    def select_action(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        raise NotImplementedError()

    def set_environment(self, environment, exp_us):
        self.environment = environment
        self.n_actions = environment.get_n_actions()
        if not self.part_of_agent:
            environment.add_agent(agent_id=self.id, agent=self)

    def take_action(self) -> Tuple[object, float, bool, object, int]:
        if self.environment is None:
            raise("Environment is not defined")
        action = self.select_action()
        observation, reward, done, info = self.environment.step(action, self.id)
        personalized_reward = self._personalize_reward(reward=reward)
        self.update(
            observation=observation,
            personalized_reward=personalized_reward,
            done=done,
            info=info,
            action=action
        )
        return observation, personalized_reward, done, info, action

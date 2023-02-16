from abc import ABC, abstractmethod


class AgentBase(ABC):
    def __init__(self, id, utility_function, environment):
        super(AgentBase, self).__init__()
        self.id = id
        self.utility_function = utility_function
        self.environment = environment
        self.add_agent()

    def add_agent(self):
        if self.id != -1:
            self.environment.add_agent()

    @abstractmethod
    def select_action(self) -> int:
        pass

    @abstractmethod
    def update(self, observation, inner_reward, done, info, action):
        pass

    def take_action(self) -> (object, float, bool, object, int):
        action = self.select_action()
        observation, reward, done, info = self.environment.step(action, self.id)
        inner_reward = self.utility_function.apply(reward)
        self.update(observation, inner_reward, done, info, action)
        return observation, inner_reward, done, info, action

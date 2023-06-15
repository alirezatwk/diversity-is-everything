import numpy as np

from abc import ABC, abstractmethod
from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase
from configs import INITIAL_STD


class ConjugatePriorBase(ABC):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    @abstractmethod
    def get_samples(self) -> np.array:
        NotImplementedError()

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        NotImplementedError()


class BernoulliPrior(ConjugatePriorBase):
    def __init__(self, n_actions: int):
        super(BernoulliPrior, self).__init__(n_actions=n_actions)
        self.alphas = np.ones(self.n_actions)
        self.betas = np.ones(self.n_actions)

    def get_samples(self) -> np.array:
        return np.random.beta(self.alphas, self.betas)

    def update(self, action: int, reward: float) -> None:
        self.alphas[action] += reward > 0
        self.betas[action] += reward <= 0


class GaussianPrior(ConjugatePriorBase):
    def __init__(self, n_actions: int, init_std: float = INITIAL_STD):
        super(GaussianPrior, self).__init__(n_actions=n_actions)
        self.means = np.zeros(self.n_actions)
        self.stds = init_std * np.ones(self.n_actions)

    def get_samples(self) -> np.array:
        return np.random.normal(self.means, self.stds)

    def update(self, action: int, reward: float) -> None:
        term = (1 / self.stds[action] ** 2) + 1
        self.means[action] = (reward + self.means[action] / self.stds[action] ** 2) / term
        self.stds[action] = np.sqrt(1 / term)  # TODO: Check variance to be in range of 0.01, 1000


class ThompsonSamplingAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            conjugate_prior: ConjugatePriorBase,
            environment: EnvironmentBase,
            part_of_agent: bool = False,
    ):
        super(ThompsonSamplingAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function
        self.conjugate_prior = conjugate_prior

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)

    def select_action(self) -> int:
        samples = self.conjugate_prior.get_samples()
        max_indices = np.argwhere(samples == samples.max()).flatten()
        return np.random.choice(max_indices)

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int):
        self.conjugate_prior.update(action=action, reward=personalized_reward)

    def set_environment_info_after_submission(self):
        pass

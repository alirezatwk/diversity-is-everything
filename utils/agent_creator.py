import numpy as np

from configs import SEED, ALPHA_MEAN, ALPHA_STD, BETA_MEAN, BETA_STD, GAMMA_MEAN, GAMMA_STD
from utility_functions import ProspectUtilityFunction


class AgentCreator:
    def __init__(
            self,
            seed: int = SEED,
            alpha_mean: float = ALPHA_MEAN,
            alpha_std: float = ALPHA_STD,
            beta_mean: float = BETA_MEAN,
            beta_std: float = BETA_STD,
            gamma_mean: float = GAMMA_MEAN,
            gamma_std: float = GAMMA_STD,
    ):
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.beta_mean = beta_mean
        self.beta_std = beta_std
        self.gamma_mean = gamma_mean
        self.gamma_std = gamma_std
        self.set_seed(seed=seed)

    @staticmethod
    def set_seed(seed: int = SEED):
        np.random.seed(seed)

    def get_agent(self, count: int = 1):
        agents = []
        for agent_id in range(count):
            alpha = np.random.normal(loc=self.alpha_mean, scale=self.alpha_std)
            beta = np.random.normal(loc=self.beta_mean, scale=self.beta_std)
            gamma = np.random.normal(loc=self.gamma_mean, scale=self.gamma_std)
            prospect_agent = ProspectUtilityFunction(alpha=alpha, beta=beta, gamma=gamma)
            agents.append(prospect_agent)
        return agents

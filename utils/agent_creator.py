import numpy as np
from utility_functions import ProspectUtilityFunction


class AgentCreator:
    def __init__(
            self,
            seed: int,
            alpha_mean: float,
            alpha_std: float,
            beta_mean: float,
            beta_std: float,
            gamma_mean: float,
            gamma_std: float,
    ):
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.beta_mean = beta_mean
        self.beta_std = beta_std
        self.gamma_mean = gamma_mean
        self.gamma_std = gamma_std
        self.set_seed(seed=seed)

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)

    def create_agent(self):
        alpha = np.random.normal(loc=self.alpha_mean, scale=self.alpha_std)
        beta = np.random.normal(loc=self.beta_mean, scale=self.beta_std)
        gamma = np.random.normal(loc=self.gamma_mean, scale=self.gamma_std)
        prospect_agent = ProspectUtilityFunction(alpha=alpha, beta=beta, gamma=gamma)
        return prospect_agent

    def create_agents(self, count: int):
        agents = []
        for agent_id in range(count):
            prospect_agent = self.create_agent()
            agents.append(prospect_agent)
        return agents

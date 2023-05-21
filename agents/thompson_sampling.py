import numpy as np
from math import sqrt

from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class ThompsonSamplingAgent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            environment: EnvironmentBase = None,
            part_of_agent: bool = False,
            conjugate_prior: str = "Bernoulli"
    ):
        super(ThompsonSamplingAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=part_of_agent
        )
        self.utility_function = utility_function
        self.conjugate_prior = conjugate_prior 
        if environment is not None:
            if self.conjugate_prior == "Bernoulli":
                self.hp1 = list(np.ones((self.n_actions,1)))            #alphas of estimated beta distributions
                self.hp2 = list(np.ones((self.n_actions,1)))            #betas of estimated beta distributions

            elif self.conjugate_prior == "Gaussian with known var=1":
                self.hp1 = list( np.zeros((self.n_actions,1)))            #means of estimated gaussian distributions
                self.hp2 = list(1000 * np.ones((self.n_actions,1)))        #stds of estimated gaussian distributions

            elif self.conjugate_prior == "Model Free Interval Estimation":
                self.hp1 = list(np.zeros((self.n_actions,1)))            #means of estimated gaussian distributions
                self.hp2 = list(1000 * np.ones((self.n_actions,1)))        #stds of estimated gaussian distributions
                self.actions_count = list(np.zeros((self.n_actions,1)))
                self.S = list(np.zeros((self.n_actions,1))) 

    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)
    
    def set_environment(self, environment, exp_us):
        super().set_environment(environment, exp_us)
        if self.conjugate_prior == "Bernoulli":
            self.hp1 = list(np.ones((self.n_actions,1)))            #alphas of estimated beta distributions
            self.hp2 = list(np.ones((self.n_actions,1)))            #betas of estimated beta distributions

        elif self.conjugate_prior == "Gaussian with known var=1":
            self.hp1 = list( np.zeros((self.n_actions,1)))            #means of estimated gaussian distributions
            self.hp2 = list(1000 * np.ones((self.n_actions,1)))        #stds of estimated gaussian distributions

        elif self.conjugate_prior == "Model Free Interval Estimation":
            self.hp1 = list(np.zeros((self.n_actions,1)))            #means of estimated gaussian distributions
            self.hp2 = list(1000 * np.ones((self.n_actions,1)))        #stds of estimated gaussian distributions
            self.actions_count = list(np.zeros((self.n_actions,1)))
            self.S = list(np.zeros((self.n_actions,1)))         

    def _get_samples(self,hp1:list[float],hp2:list[float]):
        if self.conjugate_prior == "Bernoulli":
            samples = [np.random.beta(hp1[i], hp2[i])[0] for i in range(len(hp1))]
        elif self.conjugate_prior == "Gaussian with known var=1":
            samples = [np.random.normal(hp1[i],hp2[i])[0] for i in range(len(hp1))]
        elif self.conjugate_prior == "Model Free Interval Estimation":
            samples = [np.random.normal(hp1[i],hp2[i])[0] for i in range(len(hp1))]
        return samples
    
    def _get_best_action(self, samples:np.array) -> int:
        max_indices = np.argwhere(samples == np.amax(samples)).flatten()
        return np.random.choice(max_indices)
        # return np.argmax(samples)
    
    def select_action(self) -> int:
        samples = np.array(self._get_samples(self.hp1,self.hp2))
        return self._get_best_action(samples)

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int):
        if self.conjugate_prior == "Bernoulli":
            r = int(personalized_reward > 0)
            self.hp1[action] += r 
            self.hp2[action] += (1-r)

        elif self.conjugate_prior == "Gaussian with known var=1":
            new_std = sqrt( 1 / ((1/self.hp2[action]**2) + 1) )            #TODO: Check variance to be in range of 1000, 0.01
            new_mean = (personalized_reward + (self.hp1[action] / self.hp2[action] ** 2)) / ((1/self.hp2[action]**2) + 1) 
            self.hp1[action] = new_mean
            self.hp2[action] = new_std
    
        elif self.conjugate_prior == "Model Free Interval Estimation":
            self.actions_count[action] += 1
            new_mean = self.hp1[action] + (personalized_reward - self.hp1[action])/self.actions_count[action]
            self.S[action] = self.S[action] + (personalized_reward - self.hp1[action])*(personalized_reward - new_mean)
            new_std = sqrt(self.S[action]/self.actions_count[action])
            self.hp1[action] = new_mean
            self.hp2[action] = 2 * new_std
    
    def set_environment_info_after_submission(self):
        pass

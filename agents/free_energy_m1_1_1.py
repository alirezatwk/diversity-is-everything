import numpy as np
from math import sqrt, erf
import scipy.integrate as integrate
from scipy.stats import norm, beta

from typing import Tuple
from agents import AgentBase
from environments import EnvironmentBase


class FreeEnergySocialAgent_M1_1_1(AgentBase):
    def __init__(
            self,
            id: str,
            individual_agent: AgentBase,
            lamda: float, 
            c: float,
            n0: float,
            conjugate_prior: str,
            epsilon: float, 
            environment: EnvironmentBase = None,
    ):
        super(FreeEnergySocialAgent_M1_1_1, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=False,
        )
        self.individual = individual_agent
        self.lamda = lamda 
        self.c = c
        self.n0 = n0
        self.conjugate_prior = conjugate_prior 
        self.epsilon = epsilon 
        
        self.history = []
        self.step = 0
        if environment is not None:
            if self.conjugate_prior == "Bernoulli":
                self.hp1 = list(np.ones((self.n_actions)))            #alphas of estimated beta distributions
                self.hp2 = list(np.ones((self.n_actions)))            #betas of estimated beta distributions
                self.lower_bound = list(np.zeros((self.n_actions)))
                self.upper_bound = list(np.ones((self.n_actions)))

            elif self.conjugate_prior == "Gaussian with known var=1":
                hp1 = np.zeros((self.n_actions))            #means of estimated gaussian distributions
                hp2 = 1000 * np.ones((self.n_actions))       #stds of estimated gaussian distributions
                self.lower_bound = list(hp1 - 4 * hp2)
                self.upper_bound = list(hp1 + 4 * hp2)
                self.hp1 = list(hp1)
                self.hp2 = list(hp2)

    def _personalize_reward(self, reward: float) -> float:
        return self.individual.utility_function.apply(reward)

    def set_environment(self, environment, exp_us):
        self.individual.set_environment(environment, exp_us)
        super().set_environment(environment, exp_us)
        self.history = []
        self.step = 0
        if self.conjugate_prior == "Bernoulli":
            self.hp1 = list(np.ones((self.n_actions)))            #alphas of estimated beta distributions
            self.hp2 = list(np.ones((self.n_actions)))            #betas of estimated beta distributions
            self.lower_bound = list(np.zeros((self.n_actions)))
            self.upper_bound = list(np.ones((self.n_actions)))

        elif self.conjugate_prior == "Gaussian with known var=1":
            hp1 = np.zeros((self.n_actions))            #means of estimated gaussian distributions
            hp2 = 1000 * np.ones((self.n_actions))       #stds of estimated gaussian distributions
            self.lower_bound = list(hp1 - 4 * hp2)
            self.upper_bound = list(hp1 + 4 * hp2)
            self.hp1 = list(hp1)
            self.hp2 = list(hp2)
        
    def _get_samples(self,hp1:list[float],hp2:list[float]):
        if self.conjugate_prior == "Bernoulli":
            # samples = [np.random.beta(hp1[i], hp2[i])[0] for i in range(len(hp1))]
            samples = list(np.random.beta(hp1, hp2)[0])
        elif self.conjugate_prior == "Gaussian with known var=1":
            # samples = [np.random.normal(hp1[i],hp2[i])[0] for i in range(len(hp1))]
            samples = list(np.random.normal(hp1, hp2)[0])
        return samples

    def _get_best_action(self, samples:np.array) -> int:
        max_indices = np.argwhere(samples == np.amax(samples)).flatten()
        return np.random.choice(max_indices)
    
    def _f(self, x: float, action_ind: int) -> float:
        # A = list(np.arange(self.n_actions))
        # A.remove(action_ind)
        if self.conjugate_prior == "Bernoulli":
        #     f = beta.pdf(x, a = self.hp1[action_ind], b = self.hp2[action_ind])
        #     for a_ind in A:
        #         f = f * beta.cdf(x, a = self.hp1[a_ind], b = self.hp2[a_ind])      
            f = np.prod(beta.cdf(x, a = self.hp1, b= self.hp2)) 
            f = f * beta.pdf(x, a = self.hp1[action_ind], b = self.hp2[action_ind])
            f = f / (beta.cdf(x, a = self.hp1[action_ind], b = self.hp2[action_ind]) + self.epsilon)
            # print(self.hp1[action_ind], self.hp2[action_ind],f)

        elif self.conjugate_prior == "Gaussian with known var=1":
            # f = norm.pdf(x, loc = self.hp1[action_ind], scale = self.hp2[action_ind])
            # for a_ind in A:
            #     f = f * norm.cdf(x, loc = self.hp1[a_ind], scale = self.hp2[a_ind])
            f = np.prod(norm.cdf(x, loc = self.hp1, scale= self.hp2)) 
            f = f * norm.pdf(x, loc = self.hp1[action_ind], scale = self.hp2[action_ind])
            f = f / norm.cdf(x, loc = self.hp1[action_ind], scale = self.hp2[action_ind])
        return f

    def _calculate_TS_policy(self, n_samples = 1000) -> np.array:
        # TS_policy = np.zeros(self.n_actions)
        # for i in range(n_samples):
        #     samples = np.array(self._get_samples(self.means,self.stds))
        #     act = self._get_best_action(samples)
        #     TS_policy[act] += 1     
        TS_policy = [integrate.quad(lambda x: self._f(x,a), self.lower_bound[a], self.upper_bound[a], epsabs = 1e-3)[0] for a in range(self.n_actions)]
        TS_policy = np.array(TS_policy)
        TS_policy = TS_policy / np.sum(TS_policy)
        return TS_policy #np.clip(TS_policy, 0, 1)
    
    def _FE_Calculator(self) -> np.array:
        U = np.log(self.social_information/np.sum(self.social_information, axis = 1, keepdims = True) + self.epsilon) 
        # U = np.round(U, 8)
        Pi_star = self.pi_TS * np.exp((1/self.c) * U)
        # Pi_star = np.round(Pi_star, 8)
        self.Pi_star = Pi_star / np.sum(Pi_star, axis = 1, keepdims = True)
        FE = np.sum(self.Pi_star * (self.c*np.log((self.Pi_star + self.epsilon)/(self.pi_TS + self.epsilon)) - U), axis = 1)
        return FE
    
    def _update_FE(self, reward: float, action: int) -> None:
        if self.conjugate_prior == "Bernoulli":
            r = int(reward > 0)
            self.hp1[action] += r 
            self.hp2[action] += (1-r)

        elif self.conjugate_prior == "Gaussian with known var=1":
            new_std = sqrt( 1 / ((1/self.hp2[action]**2) + 1) )            #TODO: Check variance to be in range of 1000, 0.01
            new_mean = (reward + (self.hp1[action] / self.hp2[action] ** 2)) / ((1/self.hp2[action]**2) + 1) 
            self.hp1[action] = new_mean
            self.hp2[action] = new_std
            self.lower_bound[action] = new_mean - 4 * new_std
            self.upper_bound[action] = new_mean + 4 * new_std

        self.pi_TS = self._calculate_TS_policy()
        self.FE = self._FE_Calculator()
        for i in range(self.n_agents):
            act = self.environment.get_action(step=self.step-2, agent_id=self.agents_id[i])
            self.social_information[i] =  (1 - self.lamda) * self.social_information[i] + self.lamda * np.eye(1, self.n_actions, k = act)

    def _select_agent(self) -> int:
        # select the agent based on selecting policy
        min_indices = np.argwhere(self.FE == np.amin(self.FE)).flatten()
        #print(self.step, min_indices, self.FE, self.pi_TS)
        agent_ind = np.random.choice(min_indices)
        return agent_ind
    
    def select_action(self) -> int:
        social_agent_id = self._select_agent()
        if self.agents_id[social_agent_id] == self.id or self.step == 0:
            # action = np.random.choice(self.n_actions, p = self.Pi_star[self.agents_id.index(self.id)])
            action = self.individual.select_action()
            self.history.append(self.id)
        else:
            action = np.random.choice(self.n_actions, p = self.Pi_star[social_agent_id])
            self.history.append(self.agents_id[social_agent_id])
        self.step += 1
        return action

    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        self.individual.update(
            observation=observation,
            personalized_reward=personalized_reward,
            done=done,
            info=info,
            action=action
        )
        if self.step > 1:
            self._update_FE(reward=personalized_reward, action=action)
        
    def take_action(self) -> Tuple[object, float, bool, object, int]:
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
        info['FE'] = self.FE
        return observation, personalized_reward, done, info, action
    
    def set_environment_info_after_submission(self):
        self.n_agents = self.environment.get_n_agents()
        self.k = self.environment.get_n_actions()
        self.social_information = self.n0 * np.ones((self.n_agents, self.k))
        self.pi_TS = self._calculate_TS_policy()
        self.FE = self._FE_Calculator()
        self.agents_id = self.environment.get_agents_id()
    
    def get_TS_policy(self):
        return self.pi_TS
    
    def get_Pi_Star(self):
        return self.Pi_star
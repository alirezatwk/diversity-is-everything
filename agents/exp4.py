import numpy as np

from typing import Tuple
from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class EXP4Agent(AgentBase):
    def __init__(
            self,
            id: str,
            utility_function: UtilityFunctionBase,
            epsilon:float,
            gamma: float = 0,
            environment: EnvironmentBase = None,
    ):
        super(EXP4Agent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=False,
        )
        self.utility_function = utility_function
        self.gamma = gamma
        self.epsilon = epsilon 
        self.trial = 0                                          #number of total trials 
            
    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)

    def set_environment(self, environment, exp_us):
        super().set_environment(environment, exp_us)
        self.trial = 0                                          #number of total trials 
        self.max_u = exp_us.max()
        self.min_u = exp_us.min()
    
    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:
        self.trial += 1
        self.E_star += self.E.max(axis=0).sum() 
        self.etta = np.sqrt(np.log(self.n_agents) / self.E_star) 
        
        self.P = self.Q.T @ self.E

        rewards_hat = np.ones((self.n_actions,1))
        z =  (personalized_reward - self.min_u) / (self.max_u - self.min_u)
        rewards_hat[action] = 1 - (1 - z)/self.P[0][action]

        rewards_tilda = self.E @ rewards_hat 
        self.Q = np.exp(self.etta * rewards_tilda)
        self.Q = self.Q /np.sum(self.Q)

        if self.trial > 1: 
            for agent_id in self.agents_id:
                action = self.environment.get_action(step=self.trial-2, agent_id=agent_id)
                self.social_information[self.agents_id.index(agent_id)][action] += 1

        self.E = self.social_information/np.sum(self.social_information, axis = 1, keepdims = True) # Experts' advice(M*K)

    def _select_agent(self) -> int:
        # select the agent based on selecting policy
        agent_ind = np.random.choice(self.n_agents, p = self.Q.squeeze())
        return agent_ind
    
    def select_action(self):
        selected_agent_ind = self._select_agent()
        action = np.random.choice(self.n_actions, p = self.E[selected_agent_ind])
        return action 
    
    
    def set_environment_info_after_submission(self):
        self.n_agents = self.environment.get_n_agents() 
        self.k = self.environment.get_n_actions()
        self.social_information = np.ones((self.n_agents, self.k))
        self.agents_id = self.environment.get_agents_id()
        
        self.Q = np.ones((self.n_agents,1)) #Expert Weights(M*1)
        self.Q = self.Q /np.sum(self.Q)
        self.E = self.social_information/np.sum(self.social_information, axis = 1, keepdims = True) # Experts' advice(M*K)
        self.E_star = 0

        # self.N = np.zeros((self.n_actions,1))                #number of doing each arm by agent
        # self.N_T = np.zeros((self.n_actions,1))              #number of doing each arm by target
        # self.UCB = np.zeros((self.n_actions,1))              #Upper confidence bound for each arm
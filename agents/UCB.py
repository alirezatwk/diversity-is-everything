import numpy as np

from typing import Tuple
from agents import AgentBase
from environments import EnvironmentBase
from utility_functions import UtilityFunctionBase


class UCBAgent(AgentBase):
    def __init__(
            self,
            c_ucb:float,
            id: str,
            utility_function: UtilityFunctionBase,
            lr: float = None,
            environment: EnvironmentBase = None,
    ):
        super(UCBAgent, self).__init__(
            id=id,
            environment=environment,
            part_of_agent=False,
        )
        self.utility_function = utility_function
        self.lr = lr
        self.c_ucb = c_ucb
        self.trial = 0                                          #number of total trials 
        self.actions = []
            
    def _personalize_reward(self, reward: float) -> float:
        return self.utility_function.apply(reward)

    def set_environment(self, environment, exp_us):
        super().set_environment(environment, exp_us)
        self.trial = 0                                          #number of total trials 
        self.actions = []    
        self.Q = np.zeros((self.n_actions,1))                #action value function(expected reward) for each arm
        self.N = np.zeros((self.n_actions,1))                #number of doing each arm by the agent
        self.UCB = np.zeros((self.n_actions,1))              #Upper confidence bound for each arm
    
    def update(self, observation: object, personalized_reward: float, done: bool, info: object, action: int) -> None:

        # update action values
        self.trial += 1

        self.N[action] = self.N[action] + 1
        if self.lr is not None:
            self.Q[action] += self.lr*(personalized_reward-self.Q[action])
            self.lr = self.lr * self.lr_decay
        else:
            self.Q[action] += (personalized_reward-self.Q[action]) / self.N[action]


    def select_action(self):
        if self.trial < self.n_actions:
            action = np.random.choice(np.flatnonzero(self.N == 0))
        else:        
            self.UCB = self.Q + np.sqrt(self.c_ucb * np.log(self.trial+1)/(self.N))
            action = np.random.choice(np.flatnonzero(self.UCB == self.UCB.max()))

        return action 
    
    def set_environment_info_after_submission(self):
        pass
        
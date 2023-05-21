import numpy as np
from typing import List

class FreeEnergyResult: 
    def __init__(self, repetition: int, trials: int, n_agents: int, agents_id: List[str]) -> None:
        self.repetition = repetition
        self.trials = trials
        self.n_agents = n_agents
        self.fe = np.zeros((n_agents, repetition, trials))
        self.agents_id = agents_id 

    def __init(self, repetition: int, trials: int, fe: np.array) -> None:
        self.__init__(repetition=repetition, trials=trials)
        self.fe = fe 
    
    def set_fe(self, fe: np.array) -> None:
        self.fe = fe
    
    def get_repetition(self) -> int:
        return self.repetition

    def get_trials(self) -> int:
        return self.trials

    def get_n_agents(self) -> int:
        return self.n_agents
    
    def average_free_energy(self) -> np.array:
        return np.mean(self.fe, axis=1), 2 * np.std(self.fe, axis=1)/np.sqrt(self.fe.shape[1])

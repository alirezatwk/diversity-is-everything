from typing import List

from agents import AgentBase
from tqdm import tqdm
from environments import EnvironmentBase
from rewards import RewardBase
import numpy as np


class Simulator:
    def __init__(
            self,
            environment: EnvironmentBase,
            agents: List[AgentBase],
            repetition: int,
            trials: int,
    ) -> None:
        self.environment = environment
        self.agents = agents
        self.repetition = repetition
        self.trials = trials


    def simulate(self):
        taken_actions = np.zeros((self.repetition, self.trials))
        for r in tqdm(range(self.repetition)):
            for trial in range(self.trials):
                for agent_idx, agent in enumerate(self.agents):
                    _, reward, _, info, action = agent.take_action()
                    if agent_idx == 0:
                        taken_actions[r][trial] = action

            for agent in self.agents:
                agent.reset() # TODO!

            self.environment.reset()


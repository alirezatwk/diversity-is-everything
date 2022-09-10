from agents.agent_base import AgentBase

class GreedyAgent(AgentBase):
    def __init__(self, id, environment=None, utility_function=None):
        super(GreedyAgent, self).__init__(id=id, utility_function=utility_function, environment=environment)
        
        self.
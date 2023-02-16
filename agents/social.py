from agents import AgentBase


# TODO: Make this agent work correctly
class SocialAgent(AgentBase):
    def __init__(self, id, utility_function, environment, individual_learner, social_learner):
        super(SocialAgent, self).__init__(id=id, utility_function=utility_function, environment=environment)
        self.individual_learner = individual_learner
        self.social_learner = social_learner

    def select_action(self) -> int:
        ind = self.social_learner.select_action()
        
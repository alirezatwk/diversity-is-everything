import numpy as np


class ActionsResult:
    def __init__(self, repetition: int, trials: int) -> None:
        self.repetition = repetition
        self.trials = trials
        self.actions = np.zeros((repetition, trials))

    def __init(self, repetition: int, trials: int, actions: np.array):
        self.__init__(repetition=repetition, trials=trials)
        self.actions = actions

    def set_actions(self, actions: np.array):
        self.actions = actions

    def get_repetition(self) -> int:
        return self.repetition

    def get_trials(self) -> int:
        return self.trials

    def probability_of_choosing_action(self, desired_action: int) -> np.array:
        return np.mean(self.actions == desired_action, axis=0)


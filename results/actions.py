import numpy as np


class ActionsResult:
    def __init__(self, repetition: int, trails: int) -> None:
        self.repetition = repetition
        self.trails = trails
        self.actions = np.zeros((repetition, trails))

    def get_repetition(self) -> int:
        return self.repetition

    def get_trails(self) -> int:
        return self.trails

    def probability_of_choosing_action(self, desired_action: int) -> np.array:
        return np.mean(self.actions == desired_action, axis=0)


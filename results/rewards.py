import numpy as np

class RewardsResult: 
    def __init__(self, repetition: int, trials: int) -> None:
        self.repetition = repetition
        self.trials = trials
        self.rewards = np.zeros((repetition, trials))
    
    def __init(self, repetition: int, trials: int, rewards: np.array) -> None:
        self.__init__(repetition=repetition, trials=trials)
        self.rewards = rewards 
    
    def set_rewards(self, rewards: np.array) -> None:
        self.rewards = rewards
    
    def get_repetition(self) -> int:
        return self.repetition

    def get_trials(self) -> int:
        return self.trials
    
    def exp_random_regret(self, exp_best: float) -> np.array:
        diff_rewards =  exp_best - self.rewards 
        random_regrets = np.cumsum(diff_rewards, axis=1)
        return np.mean(random_regrets, axis=0)

    def exp_pseudo_regret(self, exp_best: float) -> np.array:
        diff_rewards =  exp_best - self.rewards 
        pseudo_regrets = np.cumsum(diff_rewards, axis=1)
        return np.mean(pseudo_regrets, axis=0),2 * np.std(pseudo_regrets, axis=0)/np.sqrt(pseudo_regrets.shape[0])

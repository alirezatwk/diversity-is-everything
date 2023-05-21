from utility_functions import UtilityFunctionBase


class ProspectUtilityFunction(UtilityFunctionBase):
    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(ProspectUtilityFunction, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def apply(self, input: float) -> float:
        if input >= 0:
            return input ** self.alpha
        return -self.gamma * ((-input) ** self.beta)

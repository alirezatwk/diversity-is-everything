from utility_functions import UtilityFunctionBase


class IdenticalUtilityFunction(UtilityFunctionBase):
    def __init__(self) -> None:
        super(IdenticalUtilityFunction, self).__init__()

    def apply(self, input: float) -> float:
        return input

from utility_functions.base import UtilityFunctionBase


class IdenticalUtilityFunction(UtilityFunctionBase):
    def __init__(self):
        super(IdenticalUtilityFunction, self).__init__()

    def apply(self, input: float) -> float:
        return input

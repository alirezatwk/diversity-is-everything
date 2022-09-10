from abc import ABC, abstractmethod


class UtilityFunctionBase(ABC):
    def __init__(self):
        super(UtilityFunctionBase, self).__init__()

    @abstractmethod
    def apply(self, input: float) -> float:
        pass

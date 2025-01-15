from enum import Enum
from typing import Self

from casadi import MX

from .muscle_hill_model_abstract import ComputeForceDamping

"""
Implementations of the ComputeForceDamping protocol
"""


class ComputeForceDampingConstant:
    def __init__(self, factor: float = 0.0):
        self._factor = factor

    @property
    def copy(self) -> Self:
        return ComputeForceDampingConstant(factor=self.factor)

    def serialize(self) -> dict:
        return {
            "method": "ComputeForceDampingConstant",
            "factor": self.factor,
        }

    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != "ComputeForceDampingConstant":
            raise ValueError(f"Cannot deserialize {data['method']} as ComputeForceDampingConstant")
        return ComputeForceDampingConstant(factor=data["factor"])

    @property
    def factor(self):
        return self._factor

    def __call__(self, normalized_muscle_fiber_velocity: MX) -> MX:
        return self._factor


class ComputeForceDampingLinear:
    def __init__(self, factor: float = 0.1):
        self._factor = factor

    @property
    def copy(self) -> Self:
        return ComputeForceDampingLinear(factor=self.factor)

    def serialize(self) -> dict:
        return {
            "method": "ComputeForceDampingLinear",
            "factor": self.factor,
        }

    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != "ComputeForceDampingLinear":
            raise ValueError(f"Cannot deserialize {data['method']} as ComputeForceDampingLinear")
        return ComputeForceDampingLinear(factor=data["factor"])

    @property
    def factor(self):
        return self._factor

    def __call__(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized force from the damping

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The normalized force corresponding to the given muscle velocity
        """
        return self._factor * normalized_muscle_fiber_velocity


class ComputeForceDampingMethods(Enum):
    Constant = ComputeForceDampingConstant
    Linear = ComputeForceDampingLinear

    def __call__(self, *args, **kwargs) -> ComputeForceDamping:
        return self.value(*args, **kwargs)

    @staticmethod
    def deserialize(data: dict) -> ComputeForceDamping:
        method = data["method"]
        for method_enum in ComputeForceDampingMethods:
            if method_enum.value.__name__ == method:
                return method_enum.value.deserialize(data)
        raise ValueError(f"Cannot deserialize {method} as ComputeForceDampingMethods")

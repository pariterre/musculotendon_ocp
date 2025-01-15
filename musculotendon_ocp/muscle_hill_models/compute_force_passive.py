from enum import Enum
from typing import Self

from casadi import MX, exp

from .muscle_hill_model_abstract import ComputeForcePassive

"""
Implementations of the ComputeForcePassive protocol
"""


class ComputeForcePassiveHillType:
    def __init__(self, kpe: float = 4.0, e0: float = 0.6):
        self.kpe = kpe
        self.e0 = e0

    @property
    def copy(self) -> Self:
        return ComputeForcePassiveHillType(kpe=self.kpe, e0=self.e0)

    def serialize(self) -> dict:
        return {"method": type(self).__name__, "kpe": self.kpe, "e0": self.e0}

    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != ComputeForcePassiveHillType.__name__:
            raise ValueError(f"Cannot deserialize {data['method']} as ComputeForcePassiveHillType")
        return ComputeForcePassiveHillType(kpe=data["kpe"], e0=data["e0"])

    def __call__(self, normalized_muscle_fiber_length: MX) -> MX:
        return (exp(self.kpe * (normalized_muscle_fiber_length - 1) / self.e0) - 1) / (exp(self.kpe) - 1)


class ComputeForcePassiveAlwaysPositiveHillType(ComputeForcePassiveHillType):
    def __call__(self, normalized_muscle_fiber_length: MX) -> MX:
        return (
            super(ComputeForcePassiveAlwaysPositiveHillType, self).__call__(normalized_muscle_fiber_length)
            - self.offset
        )

    @property
    def copy(self) -> Self:
        return ComputeForcePassiveAlwaysPositiveHillType(kpe=self.kpe, e0=self.e0)

    def serialize(self) -> dict:
        return {"method": type(self).__name__, "kpe": self.kpe, "e0": self.e0}

    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != ComputeForcePassiveAlwaysPositiveHillType.__name__:
            raise ValueError(f"Cannot deserialize {data['method']} as ComputeForcePassiveAlwaysPositiveHillType")
        return ComputeForcePassiveAlwaysPositiveHillType(kpe=data["kpe"], e0=data["e0"])

    @property
    def offset(self) -> float:
        """
        Get the offset to ensure the force is always positive, by offsetting the force by the minimum value
        """
        return super(ComputeForcePassiveAlwaysPositiveHillType, self).__call__(0.0)


class ComputeForcePassiveMethods(Enum):
    HillType = ComputeForcePassiveHillType
    AlwaysPositiveHillType = ComputeForcePassiveAlwaysPositiveHillType

    def __call__(self, *args, **kwargs) -> ComputeForcePassive:
        return self.value(*args, **kwargs)

    @staticmethod
    def deserialize(data: dict) -> ComputeForcePassive:
        method = data["method"]
        for method_enum in ComputeForcePassiveMethods:
            if method_enum.value.__name__ == method:
                return method_enum.value.deserialize(data)
        raise ValueError(f"Cannot deserialize {method} as ComputeForcePassiveMethods")

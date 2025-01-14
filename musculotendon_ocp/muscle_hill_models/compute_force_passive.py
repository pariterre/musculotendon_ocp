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
